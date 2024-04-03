import re
from pathlib import Path
from typing import Union

import numpy as np
from parsl.app.app import python_app
from parsl.app.futures import DataFuture
from parsl.dataflow.futures import AppFuture

from psiflow.data import FlowAtoms
from psiflow.utils import unpack_i


def read_output(filename):  # from i-PI
    # Regex pattern to match header lines and capture relevant parts
    header_pattern = re.compile(
        r"#\s*(column|cols\.)\s+(\d+)(?:-(\d+))?\s*-->\s*([^\s\{]+)(?:\{([^\}]+)\})?\s*:\s*(.*)"
    )

    # Reading the file
    with open(filename, "r") as file:
        lines = file.readlines()

    header_lines = [line for line in lines if line.startswith("#")]
    data_lines = [line for line in lines if not line.startswith("#") and line.strip()]

    # Interprets properties
    properties = {}
    for line in header_lines:
        match = header_pattern.match(line)
        if match:
            # Extracting matched groups
            (
                col_type,
                start_col,
                end_col,
                property_name,
                units,
                description,
            ) = match.groups()
            col_info = f"{start_col}-{end_col}" if end_col else start_col
            properties[col_info] = {
                "name": property_name,
                "units": units,
                "description": description,
            }

    # Parse data
    values_dict = {}
    info_dict = {}
    for prop_info in properties.values():
        # Initialize list to hold values for each property
        values_dict[prop_info["name"]] = []
        # Save units and description
        info_dict[prop_info["name"]] = (prop_info["units"], prop_info["description"])

    for line in data_lines:
        values = line.split()
        for column_info, prop_info in properties.items():
            if "-" in column_info:  # Multi-column property
                start_col, end_col = map(
                    int, column_info.split("-")
                )  # 1-based indexing
                prop_values = values[
                    start_col - 1 : end_col
                ]  # Adjust to 0-based indexing
            else:  # Single column property
                col_index = int(column_info) - 1  # Adjust to 0-based indexing
                prop_values = [values[col_index]]

            values_dict[prop_info["name"]].append([float(val) for val in prop_values])

    for prop_name, prop_values in values_dict.items():
        values_dict[prop_name] = np.array(
            prop_values
        ).squeeze()  # make 1-col into a flat array

    return values_dict, info_dict


def _parse_data(
    keys: list[str],
    inputs: list = [],
) -> dict[str, np.ndarray]:
    from psiflow.sampling.output import read_output

    values, _ = read_output(inputs[0].filepath)
    bare_keys = []
    for key in keys:
        if "{" in key:
            bare_key = key.split("{")[0]
        else:
            bare_key = key
        bare_keys.append(bare_key)
    return [values[key] for key in bare_keys]


parse_data = python_app(_parse_data, executors=["default_threads"])


def _parse(
    state: FlowAtoms,
    inputs: list = [],
) -> int:
    time = state.info["time"]
    temperature = state.info["temperature"]

    # determine status based on stdout
    with open(inputs[0], "r") as f:
        content = f.read()
    if "force exceeded" in content:
        status = 2  # max_force exception
    elif "@SOFTEXIT: Kill signal received" in content:
        status = 1  # timeout
    elif "@ SIMULATION: Exiting cleanly" in content:
        status = 0  # everything OK
    else:
        status = -1
    return time, temperature, status


parse = python_app(_parse, executors=["default_threads"])


class SimulationOutput:
    """Gathers simulation output

    status is an integer which represents an exit code of the run:

    -1: unknown error
     0: run completed successfully
     1: run terminated early due to time limit
     2: run terminated early due to max force exception

    """

    def __init__(self, fields: list[str]):
        self._data = {key: None for key in fields}

        self.state = None
        self.stdout = None
        self.status = None
        self.time = None
        self.temperature = None
        self.trajectory = None

    def __getitem__(self, key: str):
        if key not in self._data:
            raise ValueError("output {} not available".format(key))
        return self._data[key]

    def parse(
        self,
        result: AppFuture,  # result from ipi execution
        state: AppFuture,
    ):
        self.state = state
        self.stdout = result.stdout
        parsed = parse(state, inputs=[result.stdout, result.stderr])
        self.time = unpack_i(parsed, 0)
        self.temperature = unpack_i(parsed, 1)
        self.status = unpack_i(parsed, 2)

    def parse_data(self, data_future: DataFuture):
        data = parse_data(
            list(self._data.keys()),
            inputs=[data_future],
        )
        for i, key in enumerate(self._data.keys()):
            self._data[key] = unpack_i(data, i)

    def save(self, path: Union[str, Path]):
        if type(path) is str:
            path = Path(path)
        assert not path.exists()
        raise NotImplementedError
