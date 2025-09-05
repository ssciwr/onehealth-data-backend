from onehealth_data_backend import inout
import pytest
import xarray as xr
import numpy as np


def test_download_data_invalid():
    # empty output file path
    with pytest.raises(ValueError):
        inout.download_data(None, "test_dataset", {"param": "value"})

    # empty dataset name
    with pytest.raises(ValueError):
        inout.download_data("test_output.nc", "", {"param": "value"})

    # invalid dataset name
    with pytest.raises(ValueError):
        inout.download_data("test_output.nc", 123, {"param": "value"})

    # empty request information
    with pytest.raises(ValueError):
        inout.download_data("test_output.nc", "test_dataset", None)

    # invalid request information
    with pytest.raises(ValueError):
        inout.download_data("test_output.nc", "test_dataset", "invalid_request")


def test_download_data_valid(tmp_path):
    output_file = tmp_path / "test" / "test_output.nc"
    dataset = "reanalysis-era5-land-monthly-means"
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["2m_temperature"],
        "year": ["2025"],
        "month": ["03"],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [0, -1, 0, 1],  # [N, W, S, E]
    }
    inout.download_data(output_file, dataset, request)
    assert output_file.exists()
    assert output_file.parent.exists()
    # Clean up
    output_file.unlink()


@pytest.fixture()
def get_data():
    data = np.random.rand(2, 3) * 1000 + 273.15
    data_array = xr.DataArray(
        data,
        dims=["latitude", "longitude"],
        coords={"latitude": [0, 1], "longitude": [0, 1, 2]},
    )
    return data_array


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_save_to_netcdf(get_data, tmp_path):
    with pytest.raises(ValueError):
        inout.save_to_netcdf(get_data, None)

    file_name = tmp_path / "test_output_celsius.nc"
    inout.save_to_netcdf(get_data, file_name)
    assert file_name.exists()
    # Clean up
    file_name.unlink()


def test_format_ymdt():
    numbers = None
    assert inout._format_ymdt(numbers) == ""

    numbers = []
    assert inout._format_ymdt(numbers) == ""

    numbers = [str(i).zfill(2) for i in range(1, 7)]
    assert inout._format_ymdt(numbers) == "01-06"

    numbers = [str(i).zfill(2) for i in range(1, 13, 2)]
    assert inout._format_ymdt(numbers) == "01_03_05_07_09_etc"

    numbers = [str(i).zfill(2) for i in range(1, 4, 2)]
    assert inout._format_ymdt(numbers) == "01_03"


def test_format_months():
    months = None
    assert inout._format_months(months) == ""

    months = []
    assert inout._format_months(months) == ""

    months = [str(i).zfill(2) for i in range(1, 13)]
    assert inout._format_months(months) == "allm"

    months = [str(i).zfill(2) for i in range(1, 7)]
    assert inout._format_months(months) == "01-06"

    months = [str(i).zfill(2) for i in range(1, 13, 2)]
    assert inout._format_months(months) == "01_03_05_07_09_etc"

    months = [str(i).zfill(2) for i in range(1, 4, 2)]
    assert inout._format_months(months) == "01_03"


def test_format_days():
    days = None
    assert inout._format_days(days) == ""

    days = []
    assert inout._format_days(days) == ""

    days = [str(i).zfill(2) for i in range(1, 32)]
    assert inout._format_days(days) == "alld"

    days = [str(i).zfill(2) for i in range(1, 16)]
    assert inout._format_days(days) == "01-15"

    days = [str(i).zfill(2) for i in range(1, 32, 2)]
    assert inout._format_days(days) == "01_03_05_07_09_etc"

    days = [str(i).zfill(2) for i in range(1, 4, 2)]
    assert inout._format_days(days) == "01_03"


def test_format_times():
    times = None
    assert inout._format_times(times) == ""

    times = []
    assert inout._format_times(times) == ""

    times = ["00:00"]
    assert inout._format_times(times) == "midnight"

    times = [f"{str(i).zfill(2)}:00" for i in range(0, 24)]
    assert inout._format_times(times) == "allt"

    times = [f"{str(i).zfill(2)}:00" for i in range(0, 12)]
    assert inout._format_times(times) == "00-11"

    times = [f"{str(i).zfill(2)}:00" for i in range(0, 24, 2)]
    assert inout._format_times(times) == "00_02_04_06_08_etc"

    times = ["00:00", "02:00", "04:00"]
    assert inout._format_times(times) == "00_02_04"


def test_format_variables():
    variables = []
    assert inout._format_variables(variables) == ""

    variables = ["2m_temperature"]
    assert inout._format_variables(variables) == "2t"

    variables = ["2m_temperature", "total_precipitation"]
    assert inout._format_variables(variables) == "2t_tp"

    variables = [
        "2m_temperature",
        "total_precipitation",
        "surface_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
        "total_cloud_cover",
        "low_cloud_cover",
        "medium_cloud_cover",
        "high_cloud_cover",
    ]
    assert inout._format_variables(variables) == "2t_etc"


def test_format_ds_type():
    ds_name = "reanalysis-era5-land-monthly-means"
    time_str = "_midnight"
    assert inout._format_ds_type(ds_name, time_str) == "monthly"

    ds_name = "reanalysis-era5-land"
    time_str = "_midnight"
    assert inout._format_ds_type(ds_name, time_str) == "daily"

    ds_name = "reanalysis-era5-land"
    time_str = "_00-11"
    assert inout._format_ds_type(ds_name, time_str) == ""


def test_file_extension():
    data_format = "netcdf"
    assert inout._file_extension(data_format) == "nc"

    data_format = "grib"
    assert inout._file_extension(data_format) == "grib"

    data_format = "unknown"
    with pytest.raises(ValueError):
        inout._file_extension(data_format)


def test_area_format():
    has_area = True
    assert inout._area_format(has_area) == "area"

    has_area = False
    assert inout._area_format(has_area) == ""


def test_add_prefix_if_not_empty():
    prefix = "_"
    string = "test"
    assert inout._add_prefix_if_not_empty(string, prefix) == "_test"

    prefix = "_"
    string = ""
    assert inout._add_prefix_if_not_empty(string, prefix) == ""


def test_truncate_string():
    string = "a" * 50
    max_length = 2
    assert inout._truncate_string(string, max_length) == "aa_etc"

    string = "short_string"
    max_length = 30
    assert inout._truncate_string(string, max_length) == "short_string"

    string = "a" * 100
    max_length = 100
    assert inout._truncate_string(string, max_length) == "a" * 100


def test_get_filename_var():
    file_name = inout.get_filename(
        "reanalysis-era5-land-monthly-means",
        "netcdf",
        ["2025"],
        ["01", "02"],
        None,
        None,
        True,
        "era5_data",
        ["2m_temperature"],
    )
    assert file_name == "era5_data_2025_01-02_2t_monthly_area_raw.nc"

    file_name = inout.get_filename(
        "reanalysis-era5-land-monthly-means",
        "netcdf",
        ["2025"],
        [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
        ],
        None,
        None,
        True,
        "era5_data",
        ["2m_temperature"],
    )
    assert file_name == "era5_data_2025_allm_2t_monthly_area_raw.nc"

    file_name = inout.get_filename(
        "reanalysis-era5-land",
        "netcdf",
        ["2025"],
        ["01"],
        None,
        None,
        True,
        "era5_data",
        ["2m_temperature"],
    )
    assert file_name == "era5_data_2025_01_2t_area_raw.nc"

    file_name = inout.get_filename(
        "reanalysis-era5-land",
        "netcdf",
        ["2025"],
        ["01", "02"],
        None,
        None,
        False,
        "era5_data",
        ["2m_temperature"],
    )
    assert file_name == "era5_data_2025_01-02_2t_raw.nc"

    file_name = inout.get_filename(
        "reanalysis-era5-land",
        "grib",
        ["2025"],
        ["01", "02"],
        None,
        None,
        True,
        "era5_data",
        ["2m_temperature"],
    )
    assert file_name == "era5_data_2025_01-02_2t_area_raw.grib"


def test_get_filename_vars():
    file_name = inout.get_filename(
        "reanalysis-era5-land-monthly-means",
        "netcdf",
        ["2025"],
        ["01", "02"],
        None,
        None,
        True,
        "era5_data",
        ["2m_temperature", "total_precipitation"],
    )
    assert file_name == "era5_data_2025_01-02_2t_tp_monthly_area_raw.nc"


def test_get_filename_long():
    # long vars
    var_names = [
        "2m_temperature",
        "total_precipitation",
        "surface_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
        "total_cloud_cover",
        "low_cloud_cover",
        "medium_cloud_cover",
        "high_cloud_cover",
    ]

    file_name = inout.get_filename(
        "reanalysis-era5-land-monthly-means",
        "netcdf",
        ["2025"],
        ["01", "02"],
        None,
        None,
        True,
        "era5_data",
        var_names,
    )
    assert file_name == "era5_data_2025_01-02_2t_etc_monthly_area_raw.nc"

    # long years and long vars
    years = [str(i) for i in range(1900, 2030)]
    file_name = inout.get_filename(
        "reanalysis-era5-land-monthly-means",
        "netcdf",
        years,
        ["01", "02"],
        None,
        None,
        True,
        "era5_data",
        var_names,
    )
    assert file_name == "era5_data_1900-2029_01-02_2t_etc_monthly_area_raw.nc"

    # non-continuous years and long vars
    file_name = inout.get_filename(
        "reanalysis-era5-land-monthly-means",
        "netcdf",
        ["2020", "2023", "2021"],
        ["01", "02"],
        None,
        None,
        True,
        "era5_data",
        var_names,
    )
    assert file_name == "era5_data_2020_2021_2023_01-02_2t_etc_monthly_area_raw.nc"

    # non-continuous years with more than 5 years
    years = [str(i) for i in range(2020, 2040, 2)]
    file_name = inout.get_filename(
        "reanalysis-era5-land-monthly-means",
        "netcdf",
        years,
        ["01", "02"],
        None,
        None,
        True,
        "era5_data",
        var_names,
    )
    assert (
        file_name
        == "era5_data_2020_2022_2024_2026_2028_etc_01-02_2t_etc_monthly_area_raw.nc"
    )

    # more than 100 chars
    years = [str(i) for i in range(1900, 2030)]
    file_name = inout.get_filename(
        "reanalysis-era5-land-monthly-means",
        "netcdf",
        years,
        ["01", "02"],
        None,
        None,
        True,
        "era5_data_plus_something_very_long_to_make_the_name_longer_than_100_chars",
        var_names,
    )
    assert (
        file_name
        == "era5_data_plus_something_very_long_to_make_the_name_longer_than_100_chars_"
        "1900-2029_01-02_2t_etc_mon_etc_raw.nc"
    )


def test_get_filename_none_cases():
    file_name = inout.get_filename(
        ds_name="reanalysis-era5-land-monthly-means",
        data_format="netcdf",
        years=None,
        months=None,
        days=None,
        times=None,
        has_area=False,
        base_name="era5_data",
        variables=["2m_temperature"],
    )
    assert file_name == "era5_data_2t_monthly_raw.nc"


def test_get_filename_days_times_dstype():
    file_name = inout.get_filename(
        ds_name="reanalysis-era5-land",
        data_format="netcdf",
        years=["2025"],
        months=["01"],
        days=[str(i).zfill(2) for i in range(1, 32)],
        times=[f"{str(i).zfill(2)}:00" for i in range(0, 24)],
        has_area=False,
        base_name="era5_data",
        variables=["2m_temperature"],
    )
    assert file_name == "era5_data_2025_01_alld_allt_2t_raw.nc"

    file_name = inout.get_filename(
        ds_name="reanalysis-era5-land",
        data_format="netcdf",
        years=["2025"],
        months=["01"],
        days=[str(i).zfill(2) for i in range(1, 11)],
        times=[f"{str(i).zfill(2)}:00" for i in range(0, 11)],
        has_area=False,
        base_name="era5_data",
        variables=["2m_temperature"],
    )
    assert file_name == "era5_data_2025_01_01-10_00-10_2t_raw.nc"

    file_name = inout.get_filename(
        ds_name="reanalysis-era5-land",
        data_format="netcdf",
        years=["2025"],
        months=["01"],
        days=["01", "05"],
        times=["00:00", "02:00"],
        has_area=False,
        base_name="era5_data",
        variables=["2m_temperature"],
    )
    assert file_name == "era5_data_2025_01_01_05_00_02_2t_raw.nc"

    file_name = inout.get_filename(
        ds_name="reanalysis-era5-land",
        data_format="netcdf",
        years=["2025"],
        months=["01"],
        days=["01", "05", "10", "15", "20", "25"],
        times=["00:00", "02:00", "04:00", "06:00", "08:00", "10:00"],
        has_area=False,
        base_name="era5_data",
        variables=["2m_temperature"],
    )
    assert (
        file_name == "era5_data_2025_01_01_05_10_15_20_etc_00_02_04_06_08_etc_2t_raw.nc"
    )

    file_name = inout.get_filename(
        ds_name="reanalysis-era5-land",
        data_format="netcdf",
        years=["2025"],
        months=["01"],
        days=[str(i).zfill(2) for i in range(1, 32)],
        times=["00:00"],
        has_area=True,
        base_name="era5_data",
        variables=["2m_temperature"],
    )
    assert file_name == "era5_data_2025_01_alld_midnight_2t_daily_area_raw.nc"
