## File 7: landslide/landslide_susceptibility.R

I removed the duplicated map-export block and moved the path to the top.

r
# LANDSLIDE SUSCEPTIBILITY MAPPING – XGBOOST + AHP
library(terra)
library(sf)
library(dplyr)
library(xgboost)
library(RColorBrewer)

# 1. Working directory (CHANGE THIS to your data folder)
path <- "C:/somnath/NDMA/uttarakhand/new output/"

# 2. Raster predictors
raster_files <- list(
  dem = paste0(path, "DEM.tif"),
  slope = paste0(path, "Slope.tif"),
  aspect = paste0(path, "Aspect.tif"),
  soil_moisture = paste0(path, "SMAP_soil_moisture_Jan_Mar_2025 (1).tif"),
  dist_river = paste0(path, "distance_from_river.tif"),
  dist_road = paste0(path, "distance_from_road.tif"),
  dist_fault = paste0(path, "distance_from_fault.tif"),
  dist_settlement = paste0(path, "distance_from_settlement.tif"),
  earthquake_zone = paste0(path, "mask_earthquake_density.tif"),
  drainage_density = paste0(path, "drainage_density2.tif"),
  ndvi = paste0(path, "NDVI_Uttarakhand.tif"),
  ndbi = paste0(path, "NDBI_Uttarakhand.tif"),
  rainfall_intensity = paste0(path, "Rainfall_20yr_Mean.tif")
)
layers <- lapply(raster_files, rast)
template <- layers[[1]]

# 3. Vector layers
geo_vec  <- st_read(paste0(path, "geology2.shp"))
lith_vec <- st_read(paste0(path, "lithology2.shp"))
soil_vec <- st_read(paste0(path, "DSMW_clipped.shp"))

geo_vec  <- st_transform(geo_vec,  crs(template))
lith_vec <- st_transform(lith_vec, crs(template))
soil_vec <- st_transform(soil_vec, crs(template))

layers$geology   <- rasterize(vect(geo_vec),  template, field = 1)
layers$lithology <- rasterize(vect(lith_vec), template, field = 1)
layers$soil_type <- rasterize(vect(soil_vec), template, field = 1)

# 4. Align layers
for (i in names(layers)) {
  if (!compareGeom(layers[[i]], template, stopOnError = FALSE)) {
    layers[[i]] <- resample(layers[[i]], template, method = "near")
  }
}
predictor_stack <- rast(layers)

# 5. Training points
ls_points <- st_read(paste0(path, "landslide_point.shp")) %>% st_transform(crs(template))
ls_points$ls <- 1
nonls_points <- st_read(paste0(path, "non-landslide_point.shp")) %>% st_transform(crs(template))
nonls_points$ls <- 0
all_points <- rbind(ls_points, nonls_points)

# 6. Extract raster values
vals <- extract(predictor_stack, vect(all_points))
data <- cbind(ls = all_points$ls, vals[, -1])
data <- na.omit(data)

X <- as.matrix(data[, -1])
y <- data[, 1]
dtrain <- xgb.DMatrix(data = X, label = y)

# 7. Train XGBoost
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 6,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8
)
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 300, verbose = 1)

# 8. Spatial prediction
xgb_fun <- function(model, data) predict(model, as.matrix(data))

suscept_raster <- terra::predict(
  predictor_stack,
  model = xgb_model,
  fun = xgb_fun,
  na.rm = TRUE,
  filename = paste0(path, "XGBoost_LSM_Prob5.tif"),
  overwrite = TRUE
)

# 9. Classification
class_matrix <- matrix(c(
  -Inf, 0.2, 1,
  0.2, 0.4, 2,
  0.4, 0.6, 3,
  0.6, 0.8, 4,
  0.8, Inf, 5
), ncol = 3, byrow = TRUE)

sus_class <- classify(suscept_raster, class_matrix)
writeRaster(sus_class, paste0(path, "XGBoost_LSM_Classes5.tif"), overwrite = TRUE)

# 10. AHP (currently equal weights)
factors <- names(predictor_stack)
n <- length(factors)
ahp_matrix <- matrix(1, n, n)
colnames(ahp_matrix) <- factors
rownames(ahp_matrix) <- factors

col_sum <- colSums(ahp_matrix)
norm_matrix <- sweep(ahp_matrix, 2, col_sum, "/")
weights <- rowMeans(norm_matrix)
weights <- weights / sum(weights)

ahp_map <- sum(predictor_stack[[factors]] * weights)
writeRaster(ahp_map, paste0(path, "AHP_LSM5.tif"), overwrite = TRUE)
plot(ahp_map)

ahp_class <- classify(ahp_map, class_matrix)
writeRaster(ahp_class, paste0(path, "AHP_LSM_Class5.tif"), overwrite = TRUE)

# 11. Map export with legend, north arrow, scale bar
png(paste0(path, "LSM_Map.png"), width = 1800, height = 1400)
pal <- rev(brewer.pal(5, "RdYlGn"))

plot(sus_class, col = pal, main = "Landslide Susceptibility Map",
     axes = TRUE, legend = FALSE)

legend("bottomleft",
       legend = c("Very Low", "Low", "Moderate", "High", "Very High"),
       fill = pal, cex = 1.4, bty = "n")

usr <- par("usr")
xN  <- usr[1] + 0.05 * (usr[2] - usr[1])
yN1 <- usr[4] - 0.12 * (usr[4] - usr[3])
yN2 <- usr[4] - 0.04 * (usr[4] - usr[3])
arrows(xN, yN1, xN, yN2, length = 0.15, lwd = 3)
text(xN, yN2 + 0.02 * (usr[4] - usr[3]), "N", cex = 1.8, font = 2)

scale_len <- 20000  # 20 km (only valid if raster CRS is in metres)
ext_vals <- ext(sus_class)
x_start <- ext_vals[2] - (scale_len * 1.5)
y_start <- ext_vals[3] + (scale_len * 0.25)
segments(x_start, y_start, x_start + scale_len, y_start, lwd = 4)
text(x_start + scale_len / 2, y_start + 1000, "20 km", cex = 1.3)

dev.off()