import cdsapi
from pathlib import Path
import xarray as xr
from typing import List, Dict, Any


def download_data(output_file: Path, dataset: str, request: Dict[str, Any]):
    """Download data from Copernicus's CDS using the cdsapi.

    Args:
        output_file (Path): The path to the output file where data will be saved.
        dataset (str): The name of the dataset to download.
        request (Dict[str, Any]): A dictionary containing the request parameters.
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


def save_to_netcdf(data: xr.DataArray, filename: str, encoding: Dict = None):
    """Save data to a NetCDF file.

    Args:
        data (xr.DataArray): Data to be saved.
        filename (str): The name of the output NetCDF file.
        encoding (Dict): Encoding options for the NetCDF file.
    """

    if not filename:
        raise ValueError("Filename must be provided.")

    data.to_netcdf(filename, encoding=encoding)  # TODO: check structure of encoding
    print("Data saved to {}".format(filename))


def get_filename(
    ds_name: str,
    data_format: str,
    years: List[str] | None,
    months: List[str] | None,
    days: List[str] | None = None,
    times: List[str] | None = None,
    has_area: bool = False,
    base_name: str = "era5_data",
    variables: List[str] = ["2m_temperature"],
) -> str:
    """Get file name based on dataset name, base name, years, months and area.

    Args:
        ds_name (str): Dataset name.
        data_format (str): Data format (e.g., "netcdf", "grib").
        years (List[str] | None): List of years.
        months (List[str] | None): List of months.
        days (List[str] | None): List of days.
        times (List[str] | None): List of times.
        has_area (bool): Flag indicating if area is included.
        base_name (str): Base name for the file.
            Default is "era5_data".
        variables (List[str]): List of variables.
            Default is ["2m_temperature"].

    Returns:
        str: Generated file name.
    """

    def format_ymdt(numbers: List[str] | None) -> str:
        """Format years, months, days or times into a string representation.
        If numbers is None, return empty string.
        If numbers are continuous, return start and end values (e.g., '01_12').
        If more than 5 numbers, return first 5 numbers followed by '_etc'.
        Otherwise, return all numbers joined by underscores, each formatted as two digits.
        """
        if numbers is None:
            return ""

        num_list = sorted(int(num) for num in numbers)
        are_continuous = (len(num_list) == (max(num_list) - min(num_list) + 1)) and (
            len(num_list) > 1
        )
        if are_continuous:
            return f"{min(num_list):02d}-{max(num_list):02d}"
        elif len(num_list) > 5:
            return "_".join(f"{n:02d}" for n in num_list[:5]) + "_etc"
        else:
            return "_".join(f"{n:02d}" for n in num_list)

    def format_months(months: List[str] | None) -> str:
        """Format months into a string representation.
        If months is None, return empty string.
        If all months are selected, return 'allm'.
        If months are continuous, return start and end values.
        If more than 5 months, return first 5 months followed by '_etc'.
        Otherwise, return all months joined by underscores.
        """
        if months is None:
            return ""
        elif len(set(months)) == 12:
            return "allm"
        else:
            return format_ymdt(months)

    def format_days(days: List[str] | None) -> str:
        """Format days into a string representation.
        If days is None, return empty string.
        If all days are selected, return 'alld'.
        If days are continuous, return start and end values.
        If more than 5 days, return first 5 days followed by '_etc'.
        Otherwise, return all days joined by underscores.
        """
        if days is None:
            return ""
        elif len(set(days)) == 31:  # TODO how about month with 30 or 28 days?
            return "alld"
        else:
            return format_ymdt(days)

    def format_times(times: List[str] | None) -> str:
        """Format times into a string representation.
        If times is None, return empty string.
        If only "00:00" is selected, return "midnight".
        If all times are selected, return 'allt'.
        If more than 5 times, return first 5 times followed by '_etc'.
        Otherwise, return all times in two-digit format joined by underscores.
        """
        if times is None:
            return ""
        elif "00:00" in times and len(times) == 1:
            return "midnight"
        elif len(set(times)) == 24:
            return "allt"
        else:
            return format_ymdt([t.split(":")[0] for t in times])

    def format_variables(variables: List[str]) -> str:
        """Format variables into a string representation.
        Join all first letters of variable names with underscores.
        E.g. ['2m_temperature', 'total_precipitation'] -> '2t_tp'.
        If the total length exceeds 30 characters,
            limit to first 2 characters then add '_etc'.
        """
        var_str = "_".join(
            ["".join(word[0] for word in var.split("_")) for var in variables]
        )
        if len(var_str) > 30:
            var_str = var_str[:2] + "_etc"  # e.g. 2t_etc
        return var_str

    def format_ds_type(ds_name: str, time_str: str) -> str:
        """Format dataset type based on dataset name.
        If 'monthly' is in the dataset name, return 'monthly'.
        if 'monthly' is not in the dataset name and time_str is '_midnight',
            return 'daily'.
        Otherwise, return empty string.
        """
        if "monthly" in ds_name:
            return "monthly"
        elif time_str == "_midnight":
            return "daily"
        else:
            return ""

    def file_extension(data_format: str) -> str:
        """Get file extension based on data format.
        If data format is 'grib', return 'grib'.
        Otherwise, return 'nc'.
        """
        if data_format == "grib":
            return "grib"
        else:
            return "nc"

    def area_format(has_area: bool) -> str:
        """Get area string based on has_area flag.
        If has_area is True, return 'area'.
        Otherwise, return empty string.
        """
        return "area" if has_area else ""

    def add_prefix_if_not_empty(s: str, prefix: str = "_") -> str:
        """Prefix a string with a given prefix if the string is not empty."""
        return f"{prefix}{s}" if s else ""

    def truncate_string(s: str, max_length: int = 100) -> str:
        """Truncate a string to a maximum length, adding '_etc' if truncated."""
        return s if len(s) <= max_length else s[:max_length] + "_etc"

    year_str = add_prefix_if_not_empty(format_ymdt(years))
    month_str = add_prefix_if_not_empty(format_months(months))
    day_str = add_prefix_if_not_empty(format_days(days))
    time_str = add_prefix_if_not_empty(format_times(times))
    var_str = add_prefix_if_not_empty(format_variables(variables))
    ds_type = add_prefix_if_not_empty(format_ds_type(ds_name, time_str))
    area_str = add_prefix_if_not_empty(area_format(has_area))

    file_name = f"{base_name}{year_str}{month_str}{day_str}{time_str}{var_str}{ds_type}{area_str}"

    file_name = truncate_string(file_name, max_length=100)

    # add raw to file name
    file_name = file_name + "_raw"

    file_ext = file_extension(data_format)
    file_name = file_name + "." + file_ext

    return file_name
