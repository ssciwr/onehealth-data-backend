from pathlib import Path
from importlib import resources
import json
import jsonschema
import warnings
from datetime import datetime
import socket


def is_non_empty_file(file_path: Path) -> bool:
    """Check if a file exists and is not empty.

    Args:
        file_path (Path): The path to the file.

    Returns:
        bool: True if the file exists and is not empty, False otherwise.
    """
    invalid_file = (
        not file_path or not file_path.exists() or file_path.stat().st_size == 0
    )
    if invalid_file:
        return False

    return True


def is_valid_settings(settings: dict) -> bool:
    """Check if the settings are valid.
    Args:
        settings (dict): The settings.

    Returns:
        bool: True if the settings are valid, False otherwise.
    """
    pkg = resources.files("onehealth_data_backend")
    setting_schema_path = Path(pkg / "setting_schema.json")
    setting_schema = json.load(open(setting_schema_path, "r", encoding="utf-8"))

    try:
        jsonschema.validate(instance=settings, schema=setting_schema)
        return True
    except jsonschema.ValidationError as e:
        print(e)
        return False


def _update_new_settings(settings: dict, new_settings: dict) -> bool:
    """Update the settings directly with the new settings.

    Args:
        settings (dict): The settings.
        new_settings (dict): The new settings.

    Returns:
        bool: True if the settings are updated, False otherwise.
    """
    updated = False
    if not settings:
        raise ValueError("Current settings are empty")

    for key, new_value in new_settings.items():
        # check if the new value is different from the old value
        # if the setting schema has more nested structures, deepdiff should be used
        # here just simple check
        updatable = key in settings and settings[key] != new_value
        if key not in settings:
            warnings.warn(
                "Key {} not found in the settings and will be skipped.".format(key),
                UserWarning,
            )
        if updatable:
            old_value = settings[key]
            settings[key] = new_value
            if is_valid_settings(settings):
                updated = True
            else:
                warnings.warn(
                    "The new value for key {} is not valid in the settings. "
                    "Reverting to the old value: {}".format(key, old_value),
                    UserWarning,
                )
                settings[key] = old_value

    return updated


def save_settings_to_file(settings: dict, dir_path: str = None):
    """Save the settings to a file.
    If dir_path is None, save to the current directory.

    Args:
        settings (dict): The settings.
        dir_path (str, optional): The path to save the settings file.
            Defaults to None.
    """
    now = datetime.now()
    timestamp = (
        now.strftime("%Y%m%d_%H%M%S.") + now.strftime("%f")[:3]
    )  # first 3 digits of milliseconds
    hostname = socket.gethostname()
    file_name = "updated_settings_{}_{}.json".format(timestamp, hostname)
    file_path = ""

    if dir_path is None:
        file_path = Path.cwd() / file_name
    else:
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            file_path = Path(dir_path) / file_name
        except FileExistsError:
            raise ValueError(
                "The path {} already exists and is not a directory".format(dir_path)
            )

    # save the settings to a file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)

    print("The settings have been saved to {}".format(file_path))


def get_settings(
    setting_path: str = "default",
    new_settings: dict = {},
    updated_setting_dir: str = None,
    save_updated_settings: bool = True,
) -> dict:
    """Get the settings for preprocessing steps.
    If the setting path is "default", return the default settings.
    If the setting path is not default, read the settings from the file.
    If the new settings are provided, overwrite the default/loaded settings.

    Args:
        setting_path (str): Path to the settings file.
            Defaults to "default".
        new_settings (dict): New settings to overwrite the existing settings.
            Defaults to {}.
        updated_setting_dir (str): Directory to save the updated settings file.
            Defaults to None.
        save_updated_settings (bool): Whether to save the updated settings to a file.

    Returns:
        dict: The settings.
    """
    settings = {}
    pkg = resources.files("onehealth_data_backend")
    default_setting_path = Path(pkg / "default_settings.json")

    def load_json(file_path: Path) -> dict:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    try:
        settings = (
            load_json(default_setting_path)
            if setting_path == "default"
            else load_json(Path(setting_path))
        )
        if setting_path != "default" and not is_valid_settings(settings):
            warnings.warn(
                "Invalid settings file. Using default settings instead.",
                UserWarning,
            )
            settings = load_json(default_setting_path)
    except Exception:
        warnings.warn(
            "Error in loading the settings file. Using default settings instead.",
            UserWarning,
        )
        settings = load_json(default_setting_path)

    # update the settings with the new settings
    updated = _update_new_settings(settings, new_settings)

    if updated and save_updated_settings:
        save_settings_to_file(settings, updated_setting_dir)

    return settings
