---
title: Data
hide:
- navigation
---

# Data

The supported models rely on data from the [Copernicus's CDS](https://cds.climate.copernicus.eu/), [Eurostat's NUTS definition](https://ec.europa.eu/eurostat/en/web/products-manuals-and-guidelines/w/ks-gq-23-010), and [ISIMIP's population data](https://data.isimip.org/).

## Copernicus Data

The [CDS's ERA5-Land monthly](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means?tab=overview) dataset is currently being used for now. You can either download the data directly from CDS website or use the provided Python script, [`inout module`](reference/inout.md).

For the latter option, please set up the CDS API as outlined below and take note of the naming convention used for the downloaded files.

### Set up CDS API
To use  [CDS](https://cds.climate.copernicus.eu/) API for downloading data, you need to first create an account on CDS to obtain your personal access token.

Create a `.cdsapirc` file containing your personal access token by following [this instruction](https://cds.climate.copernicus.eu/how-to-api).

### Naming convention
The filenames of the downloaded netCDF files follow this structure:
```text linenums="0"
source_name_list_of_years_list_of_months_list_of_vars[_montly][_area]_raw.ext
```

* `source_name` is `"era5_data"`,
* All years are firstly sorted.
    * If years are continuous values, `list_of_years`is a concatenate of `min` and `max` values. Otherwise, `list_of_years` is a join of all years.
    * However, if there are more than 5 years, we only keep the first 5 years and replace the rest by `"_etc"`
* `list_of_months`can be `"all"`, representing a whole year, or a join of all months
* Each variable has an abbreviation derived by the first letter of each word in the variable name (e.g. `tp` for `total precipitation`).
    * All abbreviations are then concatenated
    * If this concatenated string is longer than 30 characters, we only keep the first 2 characters and replace the the rest by `"_etc"`
* If the file was downloaded from a monthly dataset, `"_monthly"` is presented in the file name
* If the downloaded data is only for an area of the grid (instead of the whole map), `"_area"` is inserted into the file name
* If the part before `"_raw"` is longer than 100 characters, only the first 100 characters are kept and the rest is replaced by `"_etc"`
* `"_raw"` is added at the end to indicate that the file is raw data
* Extension `ext` of the file can be `.nc` or `.grib`

## Eurostat's NUTS definition 
The regions are set [here](https://ec.europa.eu/eurostat/en/web/products-manuals-and-guidelines/w/ks-gq-23-010) and corresponding shapefiles can be downloaded [here](https://ec.europa.eu/eurostat/web/gisco/geodata/statistical-units/territorial-units-statistics).

For downloading, please choose:

* The latest year from NUTS year,
* File format: `SHP`,
* Geometry type: `Polygons (RG)`,
* Scale: `20M`
* CRS: `EPSG: 4326`

???+ note
    * After downloading the file, unzip it to access the root folder containing the NUTS data (e.g. folder named `NUTS_RG_20M_2024_4326.shp`)
        * Inside the unzipped folder, there are five different shapefiles, which are all required to display and extract the NUTS regions data.
        ```
        shape data folder
        |____.shp file: geometry data (e.g. polygons)
        |____.shx file: index for geometry data
        |____.dbf file: attribute data for each NUTS region (e.g NUTS name, NUTS ID)
        |____.prj file: information on CRS
        |____.cpg file: character encoding data
        ```
    * These NUTS definition files are for Europe only.
    * If a country does not have NUTS level $x \in [1,3]$, the corresponding data for these levels is excluded from the shapefiles.

#### `NUTS_ID` explanation:
* Structure of `NUTS_ID`: `<country><level>`
* `country`: 2 letters, representing name of a country, e.g. DE
* `level`: 0 to 3 letters or numbers, signifying the level of the NUTS region

## ISIMIP Data
To download population data, please perform the following steps:

* go to [ISIMIP website](https://data.isimip.org/)
* search `population` from the search bar
* choose simulation round `ISIMIP3a`
* click `Input Data` -> `Direct human forcing` -> `Population data` -> `histsoc`
* choose `population_histsoc_30arcmin_annual`
* download file `population_histsoc_30arcmin_annual_1901_2021.nc`
