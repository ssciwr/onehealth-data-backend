# Export the version defined in project metadata
try:
    from importlib.metadata import version

    __version__ = version("heiplanet-data")
except ImportError:
    __version__ = "unknown"
