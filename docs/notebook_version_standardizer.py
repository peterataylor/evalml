import click
import json
import os

DOCS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source")


def _get_python_version(notebook):
    with open(notebook, "r") as f:
        source = json.load(f)
        version = source["metadata"]["language_info"]["version"]
    return version


def _standardize_python_version(notebook, desired_version="3.8.6"):
    with open(notebook, "r") as f:
        source = json.load(f)
        source["metadata"]["language_info"]["version"] = desired_version
    json.dump(source, open(notebook, "w"), indent=1)


def _get_ipython_notebooks(docs_source):
    directories_to_skip = ["_templates", "generated", ".ipynb_checkpoints"]
    notebooks = []
    for root, _, filenames in os.walk(docs_source):
        if any(dir_ in root for dir_ in directories_to_skip):
            continue
        for filename in filenames:
            if filename.endswith(".ipynb"):
                notebooks.append(os.path.join(root, filename))
    return notebooks


def _get_notebooks_with_different_versions(notebooks, desired_version="3.8.6"):
    different_versions = []
    for notebook in notebooks:
        version = _get_python_version(notebook)
        if version != desired_version:
            different_versions.append(notebook)
    return different_versions


def _standardize_versions(notebooks, desired_version="3.8.6"):
    for notebook in notebooks:
        _standardize_python_version(notebook, desired_version)


def _check_execution_and_output(notebook):
    with open(notebook, "r") as f:
        source = json.load(f)
        for cells in source["cells"]:
            if cells["cell_type"] == "code" and cells["execution_count"] != None:
                return False
    return True


def _fix_execution_and_output(notebook):
    with open(notebook, "r") as f:
        source = json.load(f)
        for cells in source["cells"]:
            if cells["cell_type"] == "code" and cells["execution_count"] != 'null':
                cells["execution_count"] = None
                cells["outputs"] = []
        source["metadata"]["kernelspec"]["display_name"] = "Python 3"
        source["metadata"]["kernelspec"]["name"] = "python3"
    json.dump(source, open(notebook, "w"), indent=1)


def _get_notebooks_with_executions(notebooks):
    executed = []
    for notebook in notebooks:
        if not _check_execution_and_output(notebook):
            executed.append(notebook)
    return executed


def _standardize_outputs(notebooks):
    for notebook in notebooks:
        _fix_execution_and_output(notebook)


@click.group()
def cli():
    """no-op"""


@cli.command()
@click.option(
    "--desired-version",
    default="3.8.6",
    help="python version that all notebooks should match",
)
def check_versions(desired_version):
    notebooks = _get_ipython_notebooks(DOCS_PATH)
    different_versions = _get_notebooks_with_different_versions(
        notebooks, desired_version
    )
    if different_versions:
        different_versions = ["\t" + notebook for notebook in different_versions]
        different_versions = "\n".join(different_versions)
        raise SystemExit(
            f"The following notebooks don't match {desired_version}:\n {different_versions}\n"
            "Please run make lint-fix to fix this."
        )


@cli.command()
@click.option(
    "--desired-version",
    default="3.8.6",
    help="python version that all notebooks should match",
)
def standardize(desired_version):
    notebooks = _get_ipython_notebooks(DOCS_PATH)
    different_versions = _get_notebooks_with_different_versions(
        notebooks, desired_version
    )
    executed_notebooks = _get_notebooks_with_executions(notebooks)
    if different_versions:
        _standardize_versions(different_versions, desired_version)
        different_versions = ["\t" + notebook for notebook in different_versions]
        different_versions = "\n".join(different_versions)
        click.echo(
            f"Set the notebook version to {desired_version} for:\n {different_versions}"
        )
    if executed_notebooks:
        _standardize_outputs(executed_notebooks)
        executed_notebooks = ["\t" + notebook for notebook in executed_notebooks]
        executed_notebooks = "\n".join(executed_notebooks)
        click.echo(
            f"Removed the outputs for:\n {executed_notebooks}"
        )


@cli.command()
def check_execution():
    notebooks = _get_ipython_notebooks(DOCS_PATH)
    executed_notebooks = _get_notebooks_with_executions(notebooks)
    if executed_notebooks:
        executed_notebooks = ["\t" + notebook for notebook in executed_notebooks]
        different_versions = "\n".join(executed_notebooks)
        raise SystemExit(
            f"The following notebooks have executed outputs:\n {different_versions}\n"
            "Please run make lint-fix to fix this."
        )


if __name__ == "__main__":
    cli()
