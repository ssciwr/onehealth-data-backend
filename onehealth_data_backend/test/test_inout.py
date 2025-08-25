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
    assert file_name == "era5_data_2025_01_02_2t_monthly_area_raw.nc"

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
    assert file_name == "era5_data_2025_01_02_2t_raw.nc"

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
    assert file_name == "era5_data_2025_01_02_2t_area_raw.grib"


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
    assert file_name == "era5_data_2025_01_02_2t_tp_monthly_area_raw.nc"


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
    assert file_name == "era5_data_2025_01_02_2t_etc_monthly_area_raw.nc"

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
    assert file_name == "era5_data_1900_2029_01_02_2t_etc_monthly_area_raw.nc"

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
    assert file_name == "era5_data_2020_2021_2023_01_02_2t_etc_monthly_area_raw.nc"

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
        == "era5_data_2020_2022_2024_2026_2028_etc_01_02_2t_etc_monthly_area_raw.nc"
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
        "1900_2029_01_02_2t_etc_mon_etc_raw.nc"
    )


def test_get_filename_none_cases():
    # TODO
    pass
