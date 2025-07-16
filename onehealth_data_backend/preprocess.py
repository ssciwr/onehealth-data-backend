from typing import TypeVar, Union, Callable, Dict, Any, Tuple
import xarray as xr
import numpy as np
import warnings
from pathlib import Path


T = TypeVar("T", bound=Union[np.float64, xr.DataArray])
warn_positive_resolution = "New resolution must be a positive number."


def convert_360_to_180(longitude: T) -> T:
    """Convert longitude from 0-360 to -180-180.

    Args:
        longitude (T): Longitude in 0-360 range.

    Returns:
        T: Longitude in -180-180 range.
    """
    return (longitude + 180) % 360 - 180


def adjust_longitude_360_to_180(
    dataset: xr.Dataset,
    limited_area: bool = False,
    lon_name: str = "longitude",
) -> xr.Dataset:
    """Adjust longitude from 0-360 to -180-180.

    Args:
        dataset (xr.Dataset): Dataset with longitude in 0-360 range.
        limited_area (bool): Flag indicating if the dataset is a limited area.
            Default is False.
        lon_name (str): Name of the longitude variable in the dataset.
            Default is "longitude".

    Returns:
        xr.Dataset: Dataset with longitude adjusted to -180-180 range.
    """
    if lon_name not in dataset.coords:
        raise ValueError(f"Longitude coordinate '{lon_name}' not found in the dataset.")
    # record attributes
    lon_attrs = dataset[lon_name].attrs.copy()

    # adjust longitude
    dataset = dataset.assign_coords(
        {lon_name: convert_360_to_180(dataset[lon_name])}
    ).sortby(lon_name)
    dataset[lon_name].attrs = lon_attrs

    # update attributes of data variables
    for var in dataset.data_vars.keys():
        if limited_area:
            # get old attribute values
            old_lon_first_grid = dataset[var].attrs.get(
                "GRIB_longitudeOfFirstGridPointInDegrees"
            )
            old_lon_last_grid = dataset[var].attrs.get(
                "GRIB_longitudeOfLastGridPointInDegrees"
            )
            dataset[var].attrs.update(
                {
                    "GRIB_longitudeOfFirstGridPointInDegrees": convert_360_to_180(
                        old_lon_first_grid
                    ),
                    "GRIB_longitudeOfLastGridPointInDegrees": convert_360_to_180(
                        old_lon_last_grid
                    ),
                }
            )
        else:
            dataset[var].attrs.update(
                {
                    "GRIB_longitudeOfFirstGridPointInDegrees": np.float64(-179.9),
                    "GRIB_longitudeOfLastGridPointInDegrees": np.float64(180.0),
                }
            )

    return dataset


def convert_to_celsius(temperature_kelvin: T) -> T:
    """Convert temperature from Kelvin to Celsius.

    Args:
        temperature_kelvin (T): Temperature in Kelvin,
            accessed through t2m variable in the dataset.

    Returns:
        T: Temperature in Celsius.
    """
    return temperature_kelvin - 273.15


def convert_to_celsius_with_attributes(
    dataset: xr.Dataset,
    inplace: bool = False,
    var_name: str = "t2m",
) -> xr.Dataset:
    """Convert temperature from Kelvin to Celsius and keep attributes.

    Args:
        dataset (xr.Dataset): Dataset containing temperature in Kelvin.
        inplace (bool): If True, modify the original dataset.
            If False, return a new dataset. Default is False.
        var_name (str): Name of the temperature variable in the dataset.
            Default is "t2m".

    Returns:
        xr.Dataset: Dataset with temperature converted to Celsius.
    """
    if var_name not in dataset.data_vars:
        raise ValueError(f"Variable '{var_name}' not found in the dataset.")
    if not inplace:
        dataset = dataset.copy(deep=True)

    # record attributes
    var_attrs = dataset[var_name].attrs.copy()

    # Convert temperature variable
    dataset[var_name] = convert_to_celsius(dataset[var_name])

    # Update attributes
    dataset[var_name].attrs = var_attrs
    dataset[var_name].attrs.update(
        {
            "GRIB_units": "C",
            "units": "C",
        }
    )

    return dataset


def rename_coords(dataset: xr.Dataset, coords_mapping: dict) -> xr.Dataset:
    """Rename coordinates in the dataset based on a mapping.

    Args:
        dataset (xr.Dataset): Dataset with coordinates to rename.
        coords_mapping (dict): Mapping of old coordinate names to new names.

    Returns:
        xr.Dataset: A new dataset with renamed coordinates.
    """
    coords_mapping_check = (
        isinstance(coords_mapping, dict)
        and bool(coords_mapping)
        and all(
            isinstance(old_name, str) and isinstance(new_name, str)
            for old_name, new_name in coords_mapping.items()
        )
    )
    if not coords_mapping_check:
        raise ValueError(
            "coords_mapping must be a non-empty dictionary of {old_name: new_name} pairs."
        )

    for old_name, new_name in coords_mapping.items():
        if old_name in dataset.coords:
            dataset = dataset.rename({old_name: new_name})
        else:
            warnings.warn(
                f"Coordinate '{old_name}' not found in the dataset and will be skipped.",
                UserWarning,
            )

    return dataset


def convert_m_to_mm(precipitation: T) -> T:
    """Convert precipitation from meters to millimeters.

    Args:
        precipitation (T): Precipitation in meters.

    Returns:
        T: Precipitation in millimeters.
    """
    return precipitation * 1000.0


def convert_m_to_mm_with_attributes(
    dataset: xr.Dataset, inplace: bool = False, var_name: str = "tp"
) -> xr.Dataset:
    """Convert precipitation from meters to millimeters and keep attributes.

    Args:
        dataset (xr.Dataset): Dataset containing precipitation in meters.
        inplace (bool): If True, modify the original dataset.
            If False, return a new dataset. Default is False.
        var_name (str): Name of the precipitation variable in the dataset.
            Default is "tp".

    Returns:
        xr.Dataset: Dataset with precipitation converted to millimeters.
    """
    if var_name not in dataset.data_vars:
        raise ValueError(f"Variable '{var_name}' not found in the dataset.")
    if not inplace:
        dataset = dataset.copy(deep=True)

    # record attributes
    var_attrs = dataset[var_name].attrs.copy()

    # Convert precipitation variable
    dataset[var_name] = convert_m_to_mm(dataset[var_name])

    # Update attributes
    dataset[var_name].attrs = var_attrs
    dataset[var_name].attrs.update(
        {
            "GRIB_units": "mm",
            "units": "mm",
        }
    )

    return dataset


def downsample_resolution(
    dataset: xr.Dataset,
    new_resolution: float = 0.5,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    agg_funcs: Dict[str, str] | None = None,
    agg_map: Dict[str, Callable[[Any], float]] | None = None,
) -> xr.Dataset:
    """Downsample the resolution of a dataset.

    Args:
        dataset (xr.Dataset): Dataset to change resolution.
        new_resolution (float): New resolution in degrees. Default is 0.5.
        lat_name (str): Name of the latitude coordinate. Default is "latitude".
        lon_name (str): Name of the longitude coordinate. Default is "longitude".
        agg_funcs (Dict[str, str] | None): Aggregation functions for each variable.
            If None, default aggregation (i.e. mean) is used. Default is None.
        agg_map (Dict[str, Callable[[Any], float]] | None): Mapping of string
            to aggregation functions.
            If None, default mapping is used. Default is None.

    Returns:
        xr.Dataset: Dataset with changed resolution.
    """
    if lat_name not in dataset.coords or lon_name not in dataset.coords:
        raise ValueError(
            f"Coordinate names '{lat_name}' and '{lon_name}' are incorrect."
        )
    if new_resolution <= 0:
        raise ValueError(warn_positive_resolution)

    old_resolution = np.round((dataset[lon_name][1] - dataset[lon_name][0]).item(), 2)

    if new_resolution <= old_resolution:
        raise ValueError(
            f"To downsample, degree of new resolution {new_resolution} "
            "should be greater than {old_resolution}."
        )

    weight = int(np.ceil(new_resolution / old_resolution))
    dim_kwargs = {
        lon_name: weight,
        lat_name: weight,
    }

    if agg_map is None:
        agg_map = {
            "mean": np.mean,
            "sum": np.sum,
            "max": np.max,
            "min": np.min,
        }
    if agg_funcs is None:
        agg_funcs = dict.fromkeys(dataset.data_vars, "mean")
    elif not isinstance(agg_funcs, dict):
        raise ValueError(
            "agg_funcs must be a dictionary of variable names and aggregation functions."
        )

    result = {}
    for var in dataset.data_vars:
        func_str = agg_funcs.get(var, "mean")
        func = agg_map.get(func_str, np.mean)

        # apply coarsening and reduction per variable
        result[var] = dataset[var].coarsen(**dim_kwargs, boundary="trim").reduce(func)
        result[var].attrs = dataset[var].attrs.copy()

    # copy attributes of the dataset
    result_dataset = xr.Dataset(result)
    result_dataset.attrs = dataset.attrs.copy()

    return result_dataset


def align_lon_lat_with_popu_data(
    dataset: xr.Dataset,
    expected_longitude_max: np.float64 = np.float64(179.75),
    lat_name: str = "latitude",
    lon_name: str = "longitude",
) -> xr.Dataset:
    """Align longitude and latitude coordinates with population data\
    of the same resolution.
    This function is specifically designed to ensure that the
    longitude and latitude coordinates in the dataset match the expected
    values used in population data, which are:
    - Longitude: -179.75 to 179.75, 720 points
    - Latitude: 89.75 to -89.75, 360 points

    Args:
        dataset (xr.Dataset): Dataset with longitude and latitude coordinates.
        expected_longitude_max (np.float64): Expected maximum longitude
            after adjustment. Default is np.float64(179.75).
        lat_name (str): Name of the latitude coordinate. Default is "latitude".
        lon_name (str): Name of the longitude coordinate. Default is "longitude".

    Returns:
        xr.Dataset: Dataset with adjusted longitude and latitude coordinates.
    """
    if lat_name not in dataset.coords or lon_name not in dataset.coords:
        raise ValueError(
            f"Coordinate names '{lat_name}' and '{lon_name}' are incorrect."
        )

    old_longitude_min = dataset[lon_name].min().values
    old_longitude_max = dataset[lon_name].max().values

    # TODO: find a more general solution
    special_case = (
        np.isclose(expected_longitude_max, np.float64(179.75))
        and np.isclose(old_longitude_min, np.float64(-179.7))
        and np.isclose(old_longitude_max, np.float64(179.8))
    )
    if special_case:
        offset = expected_longitude_max - old_longitude_max

        # adjust coord values
        dataset = dataset.assign_coords(
            {
                lon_name: (dataset[lon_name] + offset).round(2),
                lat_name: (dataset[lat_name] + offset).round(2),
            }
        )

    return dataset


def upsample_resolution(
    dataset: xr.Dataset,
    new_resolution: float = 0.1,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    method_map: Dict[str, str] | None = None,
) -> xr.Dataset:
    """Upsample the resolution of a dataset.

    Args:
        dataset (xr.Dataset): Dataset to change resolution.
        new_resolution (float): New resolution in degrees. Default is 0.1.
        lat_name (str): Name of the latitude coordinate. Default is "latitude".
        lon_name (str): Name of the longitude coordinate. Default is "longitude".
        method_map (Dict[str, str] | None): Mapping of variable names to
            interpolation methods. If None, linear interpolation is used.
            Default is None.

    Returns:
        xr.Dataset: Dataset with changed resolution.
    """
    if lat_name not in dataset.coords or lon_name not in dataset.coords:
        raise ValueError(
            f"Coordinate names '{lat_name}' and '{lon_name}' are incorrect."
        )
    if new_resolution <= 0:
        raise ValueError(warn_positive_resolution)

    old_resolution = np.round((dataset[lon_name][1] - dataset[lon_name][0]).item(), 2)

    if new_resolution >= old_resolution:
        raise ValueError(
            f"To upsample, degree of new resolution {new_resolution} "
            "should be smaller than {old_resolution}."
        )

    lat_min, lat_max = (
        dataset[lat_name].min().values,
        dataset[lat_name].max().values,
    )
    lon_min, lon_max = (
        dataset[lon_name].min().values,
        dataset[lon_name].max().values,
    )
    updated_lat = np.arange(lat_min, lat_max + new_resolution, new_resolution)
    updated_lon = np.arange(lon_min, lon_max + new_resolution, new_resolution)
    updated_coords = {
        lat_name: updated_lat,
        lon_name: updated_lon,
    }

    if method_map is None:
        method_map = dict.fromkeys(dataset.data_vars, "linear")
    elif not isinstance(method_map, dict):
        raise ValueError(
            "method_map must be a dictionary of variable names and interpolation methods."
        )

    # interpolate each variable
    result = {}
    for var in dataset.data_vars:
        method = method_map.get(var, "linear")
        result[var] = dataset[var].interp(**updated_coords, method=method)
        result[var].attrs = dataset[var].attrs.copy()

    # create a new dataset with the interpolated variables
    result_dataset = xr.Dataset(result)
    result_dataset.attrs = dataset.attrs.copy()

    return result_dataset


def resample_resolution(
    dataset: xr.Dataset,
    new_resolution: float = 0.5,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    agg_funcs: Dict[str, str] | None = None,
    agg_map: Dict[str, Callable[[Any], float]] | None = None,
    expected_longitude_max: np.float64 = np.float64(179.75),
    method_map: Dict[str, str] | None = None,
) -> xr.Dataset:
    """Resample the grid of a dataset to a new resolution.

    Args:
        dataset (xr.Dataset): Dataset to resample.
        new_resolution (float): New resolution in degrees. Default is 0.5.
        lat_name (str): Name of the latitude coordinate. Default is "latitude".
        lon_name (str): Name of the longitude coordinate. Default is "longitude".
        agg_funcs (Dict[str, str] | None): Aggregation functions for each variable.
            If None, default aggregation (i.e. mean) is used. Default is None.
        agg_map (Dict[str, Callable[[Any], float]] | None): Mapping of string
            to aggregation functions. If None, default mapping is used. Default is None.
        expected_longitude_max (np.float64): Expected maximum longitude
            after adjustment. Default is np.float64(179.75).
        method_map (Dict[str, str] | None): Mapping of variable names to
            interpolation methods. If None, linear interpolation is used. Default is None.

    Returns:
        xr.Dataset: Resampled dataset with changed resolution.
    """
    if lat_name not in dataset.coords or lon_name not in dataset.coords:
        raise ValueError(
            f"Coordinate names '{lat_name}' and '{lon_name}' are incorrect."
        )

    if new_resolution <= 0:
        raise ValueError(warn_positive_resolution)

    old_resolution = np.round((dataset[lon_name][1] - dataset[lon_name][0]).item(), 2)

    if new_resolution > old_resolution:
        dataset = downsample_resolution(
            dataset,
            new_resolution=new_resolution,
            lat_name=lat_name,
            lon_name=lon_name,
            agg_funcs=agg_funcs,
            agg_map=agg_map,
        )
        return align_lon_lat_with_popu_data(
            dataset,
            expected_longitude_max=expected_longitude_max,
            lat_name=lat_name,
            lon_name=lon_name,
        )

    return upsample_resolution(
        dataset,
        new_resolution=new_resolution,
        lat_name=lat_name,
        lon_name=lon_name,
        method_map=method_map,
    )


def truncate_data_from_time(
    dataset: xr.Dataset,
    start_date: Union[str, np.datetime64],
    var_name: str = "time",
) -> xr.Dataset:
    """Truncate data from a specific start date.

    Args:
        dataset (xr.Dataset): Dataset to truncate.
        start_date (Union[str, np.datetime64]): Start date for truncation.
            Format as "YYYY-MM-DD" or as a numpy datetime64 object.
        var_name (str): Name of the time variable in the dataset. Default is "time".

    Returns:
        xr.Dataset: Dataset truncated from the specified start date.
    """
    end_date = dataset[var_name].max().values
    if isinstance(start_date, str):
        start_date = np.datetime64(start_date, "ns")
    return dataset.sel({var_name: slice(start_date, end_date)})


def _replace_decimal_point(degree: float) -> str:
    """Replace the decimal point in a degree string with 'p'
    if the degree is greater than or equal to 1.0,
    or remove it if the degree is less than 1.0.

    Args:
        degree (float): Degree value to convert.

    Returns:
        str: String representation of the degree without decimal point.
    """
    if not isinstance(degree, (float)):
        raise ValueError("Resolution degree must be a float.")
    if degree < 1.0:
        return str(degree).replace(".", "")
    else:
        return str(degree).replace(".", "p")


def _apply_preprocessing(
    dataset: xr.Dataset,
    file_name_base: str,
    settings: Dict[str, Any],
) -> Tuple[xr.Dataset, str]:
    """Apply preprocessing steps to the dataset based on settings.

    Args:
        dataset (xr.Dataset): Dataset to preprocess.
        file_name_base (str): Base name for the output file.
        settings (Dict[str, Any]): Settings for preprocessing.

    Returns:
        Tuple[xr.Dataset, str]: Preprocessed dataset and updated file name.
    """
    # get settings
    unify_coords = settings.get("unify_coords", False)
    unify_coords_fname = settings.get("unify_coords_fname")
    uni_coords = settings.get("uni_coords")

    adjust_longitude = settings.get("adjust_longitude", False)
    adjust_longitude_vname = settings.get("adjust_longitude_vname")
    adjust_longitude_fname = settings.get("adjust_longitude_fname")

    convert_kelvin_to_celsius = settings.get("convert_kelvin_to_celsius", False)
    convert_kelvin_to_celsius_vname = settings.get("convert_kelvin_to_celsius_vname")
    convert_kelvin_to_celsius_fname = settings.get("convert_kelvin_to_celsius_fname")

    convert_m_to_mm_precipitation = settings.get("convert_m_to_mm_precipitation", False)
    convert_m_to_mm_precipitation_vname = settings.get(
        "convert_m_to_mm_precipitation_vname"
    )
    convert_m_to_mm_precipitation_fname = settings.get(
        "convert_m_to_mm_precipitation_fname"
    )

    resample_grid = settings.get("resample_grid", False)
    resample_grid_vname = settings.get("resample_grid_vname")
    lat_name = resample_grid_vname[0] if resample_grid_vname else None
    lon_name = resample_grid_vname[1] if resample_grid_vname else None
    resample_grid_fname = settings.get("resample_grid_fname")
    resample_degree = settings.get("resample_degree")

    truncate_date = settings.get("truncate_date", False)
    truncate_date_from = settings.get("truncate_date_from")
    truncate_date_vname = settings.get("truncate_date_vname")

    if unify_coords:
        print("Renaming coordinates to unify them across datasets...")
        dataset = rename_coords(dataset, uni_coords)
        file_name_base += f"_{unify_coords_fname}"

    if adjust_longitude and adjust_longitude_vname in dataset.coords:
        print("Adjusting longitude from 0-360 to -180-180...")
        dataset = adjust_longitude_360_to_180(
            dataset, lon_name=adjust_longitude_vname
        )  # only consider full map for now, i.e. limited_area=False
        file_name_base += f"_{adjust_longitude_fname}"

    if (
        convert_kelvin_to_celsius
        and convert_kelvin_to_celsius_vname in dataset.data_vars
    ):
        print("Converting temperature from Kelvin to Celsius...")
        dataset = convert_to_celsius_with_attributes(
            dataset, var_name=convert_kelvin_to_celsius_vname
        )
        file_name_base += f"_{convert_kelvin_to_celsius_fname}"

    if (
        convert_m_to_mm_precipitation
        and convert_m_to_mm_precipitation_vname in dataset.data_vars
    ):
        print("Converting precipitation from meters to millimeters...")
        dataset = convert_m_to_mm_with_attributes(
            dataset, var_name=convert_m_to_mm_precipitation_vname
        )
        file_name_base += f"_{convert_m_to_mm_precipitation_fname}"

    if resample_grid and lat_name in dataset.coords and lon_name in dataset.coords:
        print("Resampling grid to a new resolution...")
        dataset = resample_resolution(
            dataset,
            new_resolution=resample_degree,
            lat_name=lat_name,
            lon_name=lon_name,
        )  # agg_funcs, agg_map, and method_map are omitted for simplicity
        degree_str = _replace_decimal_point(resample_degree)
        file_name_base += f"_{degree_str}{resample_grid_fname}"

    if truncate_date and truncate_date_vname in dataset.coords:
        print("Truncating data from a specific start date...")
        dataset = truncate_data_from_time(
            dataset, start_date=truncate_date_from, var_name=truncate_date_vname
        )
        max_time = dataset[truncate_date_vname].max().values
        max_year = np.datetime64(max_time, "Y")
        file_name_base += f"_{truncate_date_from[:4]}_{max_year}"

    return dataset, file_name_base


def preprocess_data_file(
    netcdf_file: Path,
    settings: Dict[str, Any],
) -> xr.Dataset:
    """Preprocess the dataset based on provided settings.
    Processed data is saved to the same directory with updated filename,
    defined by the settings.

    Args:
        netcdf_file (Path): Path to the NetCDF file to preprocess.
        settings (Dict[str, Any]): Settings for preprocessing.

    Returns:
        xr.Dataset: Preprocessed dataset.
    """
    invalid_file = (
        not netcdf_file or not netcdf_file.exists() or netcdf_file.stat().st_size == 0
    )
    if invalid_file:
        raise ValueError("netcdf_file must be a valid file path.")

    if not settings:
        raise ValueError("settings must be a non-empty dictionary.")

    folder_path = netcdf_file.parent
    file_name = netcdf_file.stem
    file_name = file_name[: -len("_raw")] if file_name.endswith("_raw") else file_name
    file_ext = netcdf_file.suffix

    with xr.open_dataset(netcdf_file) as dataset:
        dataset, file_name_base = _apply_preprocessing(dataset, file_name, settings)
        # save the processed dataset
        output_file = folder_path / f"{file_name_base}{file_ext}"
        dataset.to_netcdf(output_file, mode="w", format="NETCDF4")
        print(f"Processed dataset saved to: {output_file}")
        return dataset
