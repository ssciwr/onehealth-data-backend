---
title: Issues
hide:
- navigation
---

# Issues

## Address DeprecationWarning of the mkdocs-jupyter plugin
While using `mkdocs serve` you might get warning as:
> DeprecationWarning: Jupyter is migrating its paths to use standard platformdirs given by the platformdirs library.  To remove this warning and see the appropriate new directories, set the environment variable `JUPYTER_PLATFORM_DIRS=1` and then run `jupyter --paths`.

To address this warning *temporarily* for every session:

=== "bash"
    ```shell
    export JUPYTER_PLATFORM_DIRS=1
    mkdocs serve
    ```

=== "powershell"
    ```shell
    $env:JUPYTER_PLATFORM_DIRS=1
    mkdocs serve
    ```

Or *permanently* by saving the variable value into your conda environment

```shell
conda activate your_env
conda env config vars set JUPYTER_PLATFORM_DIRS=1 #(1)!
```
{ .annotate }

1.  reactivate `your_env` would be needed

## Other issues
Other issues are listed [here](https://github.com/ssciwr/heiplanet-data/issues)