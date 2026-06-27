#!/bin/bash
#=========================
# Example preprocessing workflow for topographic data
# Requirements: GDAL, CDO

# 1. Clip topography raster to target bounding box
gdalwarp -te <minX> <minY> <maxX> <maxY> input.tif SRTM_slope_cut.tif

# 2. Convert GeoTIFF to NetCDF
gdal_translate -of netCDF SRTM_slope_cut.tif slope_raw.nc

# 3. Remap to target grid
cdo remapcon2,target_latlon.txt slope_raw.nc slope_conserv.nc

