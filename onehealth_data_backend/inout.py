import cdsapi
from pathlib import Path
import xarray as xr


def download_data(output_file: Path, dataset: str, request: dict):
    """Download data from Copernicus's CDS using the cdsapi.

    Args:
        output_file (Path): The path to the output file where data will be saved.
        dataset (str): The name of the dataset to download.
        request (dict): A dictionary containing the request parameters.
    """
    if not output_file:
        raise ValueError("Output file path must be provided.")

    if not dataset or not isinstance(dataset, str):
        raise ValueError("Dataset name must be a non-empty string.")

    if not request or not isinstance(request, dict):
        raise ValueError("Request information must be a dictionary.")

    if not output_file.exists():
        # create the directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()
    client.retrieve(dataset, request, target=str(output_file))
    print("Data downloaded successfully to {}".format(output_file))


def save_to_netcdf(data: xr.DataArray, filename: str, encoding: dict = None):
    """Save data to a NetCDF file.

    Args:
        data (xr.DataArray): Data to be saved.
        filename (str): The name of the output NetCDF file.
        encoding (dict): Encoding options for the NetCDF file.
    """

    if not filename:
        raise ValueError("Filename must be provided.")

    data.to_netcdf(filename, encoding=encoding)  # TODO: check structure of encoding
    print("Data saved to {}".format(filename))


def get_filename(
    ds_name: str,
    data_format: str,
    years: list,
    months: list,
    has_area: bool,
    base_name: str = "era5_data",
    variable: list = ["2m_temperature"],
) -> str:
    """Get file name based on dataset name, base name, years, months and area.

    Args:
        ds_name (str): Dataset name.
        data_format (str): Data format (e.g., "netcdf", "grib").
        years (list): List of years.
        months (list): List of months.
        has_area (bool): Flag indicating if area is included.
        base_name (str): Base name for the file.
            Default is "era5_data".
        variable (list): List of variables.
            Default is ["2m_temperature"].

    Returns:
        str: Generated file name.
    """
    # TODO: refactor to smaller functions
    year_nums = sorted(int(year) for year in years)
    are_continuous_years = (
        len(year_nums) == (max(year_nums) - min(year_nums) + 1)
    ) and len(year_nums) > 1
    if are_continuous_years:
        year_str = "_".join([str(min(year_nums)), str(max(year_nums))])
    elif len(year_nums) > 5:
        year_str = "_".join(str(y) for y in year_nums[:5]) + "_etc"
    else:
        year_str = "_".join(str(y) for y in year_nums)

    if len(set(months)) != 12:
        month_str = "_".join(months)
    else:
        month_str = "all"

    var_str = "_".join(
        ["".join(word[0] for word in var.split("_")) for var in variable]
    )
    if len(var_str) > 30:
        var_str = var_str[:2] + "_etc"  # e.g. 2t_etc

    if "monthly" in ds_name:
        ds_type = "_monthly"
    else:
        ds_type = ""

    file_name = base_name + "_{}_{}_{}".format(year_str, month_str, var_str) + ds_type

    if has_area:
        file_name = file_name + "_area"

    if len(file_name) > 100:
        file_name = file_name[:100] + "_etc"

    # add raw to file name
    file_name = file_name + "_raw"

    file_ext = "grib" if data_format == "grib" else "nc"
    file_name = file_name + "." + file_ext

    return file_name
