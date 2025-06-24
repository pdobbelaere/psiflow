import re
import shutil
from typing import Optional, Union, Sequence

import numpy as np
import typeguard
from ase.data import atomic_numbers, chemical_symbols
from parsl.app.app import python_app
from parsl.dataflow.futures import AppFuture

from psiflow.geometry import Geometry, NullState, _assign_identifier, create_outputs, GeometryLike, NULLSTATE, get_unique_numbers, get_atomic_energy
from psiflow.quantities import quantities, Type


def iter_read_frames(file: str) -> list[str]:
    """"""
    frame_regex = re.compile(r"^\d+$")
    with open(file, "r") as f:
        for line in f:
            if frame_regex.match(_ := line.strip()):
                natoms = int(_)
                yield [line] + [f.readline() for _i in range(natoms + 1)]


@typeguard.typechecked
def _write_frames(
    *states: GeometryLike | list[GeometryLike],
    outputs: list = [],
) -> None:
    """
    Write Geometry instances to a file.

    Args:
        *states: Variable number of Geometry instances to write.
        outputs: List of Parsl futures. The first element should be a DataFuture
                 representing the output file path.

    Returns:
        None

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    assert len(outputs) == 1
    data = []
    for d in states:
        if isinstance(d, list):
            data.extend(d)
        else:
            data.append(d)
    with open(outputs[0], "w") as f:
        f.write(''.join([geom.to_string() for geom in data]))


write_frames = python_app(_write_frames, executors=["default_threads"])


@typeguard.typechecked
def _read_frames(
    indices: Union[None, slice, list[int], int] = None,
    inputs: list = [],
    outputs: list = [],
) -> Optional[list[GeometryLike]]:
    """
    Read Geometry instances from a file.

    Args:
        indices: Indices of frames to read. Can be None (read all), a slice, a list of integers, or a single integer.
        inputs: List of Parsl futures. The first element should be a DataFuture
                representing the input file path containing the geometry data.
        outputs: List of Parsl futures. If provided, the first element should be
                 a DataFuture representing the output file path where the selected
                 geometries will be written.

    Returns:
        Optional[list[Geometry]]: List of read Geometry instances if no output
                                  is specified, otherwise None.

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    # TODO: reads twice + indices always wrap modulo nframes which might be unexpected
    assert len(inputs) == 1 and len(outputs) < 2
    length = _count_frames(inputs=inputs)
    _all = range(length)
    if isinstance(indices, slice):
        indices = list(_all[indices])
    elif isinstance(indices, int):
        indices = [list(_all)[indices]]

    if isinstance(indices, list):
        if length > 0:
            indices = [i % length for i in indices]  # for negative indices and wrapping
        indices_ = set(indices)  # for *much* faster 'i in indices'
    else:
        assert indices is None
        indices_ = None

    data = []
    frame_index = 0
    for geom_str_list in iter_read_frames(inputs[0]):
        # currently at position frame_index, check if to be read
        if indices_ is None or frame_index in indices_:
            data.append("".join(geom_str_list))
        else:
            data.append(None)
        frame_index += 1

    if indices is not None:  # sort states accordingly and filter None
        data = [data[i] for i in indices]

    if len(outputs) > 0:
        with open(outputs[0], "w") as f:
            f.write(''.join(data))
    else:
        geometries = [Geometry.from_string(s) for s in data]
        return geometries


read_frames = python_app(_read_frames, executors=["default_threads"])


@typeguard.typechecked
def _extract_quantities(
    quantity_names: Sequence[str],
    *geometries: Geometry,
    atom_indices: Optional[list[int]] = None,
    elements: Optional[list[str]] = None,
    inputs: list = [],
) -> tuple[np.ndarray, ...]:
    """
    Extract specified quantities from Geometry instances.

    Args:
        quantity_names: Tuple of quantity names to extract.
        *data: Geometry instances.
        atom_indices: List of atom indices to consider.
        elements: List of element symbols to consider.
        inputs: List of Parsl futures. If provided, the first element should be a DataFuture
                representing the input file path containing geometry data.

    Returns:
        tuple[np.ndarray, ...]: Tuple of arrays containing extracted quantities.

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    if not geometries:
        data = _read_frames(inputs=inputs)
    else:
        assert len(inputs) == 0
        data = list(geometries)
    max_natoms = int(np.array([len(geometry) for geometry in data if geometry != NULLSTATE]).max())

    # already sets default values for missing data
    array_dict, unknown_quantities = create_outputs(quantity_names, data)
    # we do not know about default values for unregistered quantities
    array_dict_unknown = {name: np.array([None] * len(data)) for name in unknown_quantities}

    for i, geometry in enumerate(data):
        if geometry == NULLSTATE:           # TODO: what if you want NullState metadata?
            continue
        natoms = len(geometry)
        mask = get_index_element_mask(
            geometry.numbers,
            atom_indices,
            elements,
            natoms_padded=max_natoms,
        )

        for name, array in array_dict.items():
            quantity = quantities.all[name]
            if quantity.per_atom:
                array[i, mask] = geometry.per_atom[name][mask[:natoms]]
            elif quantity.type == Type.METADATA and name in geometry.metadata:
                array[i] = geometry[name]
            elif hasattr(geometry, name):
                array[i] = getattr(geometry, name)

        for name, array in array_dict_unknown.items():      # TODO: some warning message?
            array[i] = geometry.metadata.get(name)

    array_dict |= array_dict_unknown
    return *[array_dict[name] for name in quantity_names],


extract_quantities = python_app(_extract_quantities, executors=["default_threads"])


@typeguard.typechecked
def _insert_quantities(
    quantity_names: Sequence[str],
    arrays: Sequence[np.ndarray],
    *data: Geometry,
    inputs: list = [],
    outputs: list = [],
) -> None:
    """
    Insert quantities into Geometry instances.

    Args:
        quantity_names: Tuple of quantity names to insert.
        arrays: List of arrays containing the quantities to insert.
        data: Geometry instances to update.
        inputs: List of Parsl futures. If provided, the first element should be a DataFuture
                representing the input file path containing geometry data.
        outputs: List of Parsl futures. If provided, the first element should be a DataFuture
                 representing the output file path where updated geometries will be written.

    Returns:
        None

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    assert len(outputs) == 1
    for name in quantity_names:
        if name not in quantities.all:
            raise ValueError(f'Unknown quantity "{name}"')
    if not data:
        data = tuple(_read_frames(inputs=inputs))
    else:
        assert len(inputs) == 0
    assert all(arr.shape[0] == len(data) for arr in arrays)

    max_natoms = int(np.array([len(geometry) for geometry in data if geometry != NULLSTATE]).max())
    for i, geometry in enumerate(data):
        if geometry == NULLSTATE:
            continue
        mask = get_index_element_mask(
            geometry.numbers,
            None,
            None,
            natoms_padded=max_natoms,
        )
        for j, name in enumerate(quantity_names):
            quantity = quantities.all[name]
            if quantity.per_atom:
                geometry.per_atom[name] = arrays[j][i, mask]
            else:
                value = arrays[j][i]
                if np.isnan(value).all():
                    value = quantity.default
                elif value.size == 1:
                    value = value.item()
                if quantity.type == Type.METADATA:  # TODO: you cannot set metadata unless the quantity is registered?
                    geometry[name] = arrays[j][i]
                else:
                    setattr(geometry, name, value)

    if len(outputs) > 0:
        _write_frames(*data, outputs=[outputs[0]])


insert_quantities = python_app(_insert_quantities, executors=["default_threads"])


@typeguard.typechecked
def _check_distances(state: Geometry, threshold: float) -> GeometryLike:
    """
    Check if all interatomic distances in a Geometry are above a threshold.

    Args:
        state: Geometry instance to check.
        threshold: Minimum allowed interatomic distance.

    Returns:
        Geometry: The input Geometry if all distances are above the threshold, otherwise NullState.

    Note:
        This function is wrapped as a Parsl app and executed using the default_htex executor.
    """
    from ase.geometry.geometry import find_mic
    nrows = int(len(state) * (len(state) - 1) / 2)
    deltas = np.zeros((nrows, 3))
    count = 0
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
            deltas[count] = state.per_atom.positions[i] - state.per_atom.positions[j]
            count += 1
    assert count == nrows
    if state.periodic:
        deltas, _ = find_mic(deltas, state.cell)
    check = np.all(np.linalg.norm(deltas, axis=1) > threshold)
    return state if check else NULLSTATE


check_distances = python_app(_check_distances, executors=["default_htex"])


@typeguard.typechecked
def _assign_identifiers(
    identifier: Optional[int],
    inputs: list = [],
    outputs: list = [],
) -> int:
    """
    Assign identifiers to Geometry instances in a file.

    Args:
        identifier: Starting identifier value.
        inputs: List of Parsl futures. The first element should be a DataFuture
                representing the input file path containing geometry data.
        outputs: List of Parsl futures. The first element should be a DataFuture
                 representing the output file path where updated geometries will be written.

    Returns:
        int: Next available identifier.

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    assert len(inputs) == 1 and len(outputs) == 1
    data = _read_frames(inputs=[inputs[0]])
    if identifier is None:  # look for maximum assigned identifier
        identifiers = [geom['identifier'] for geom in data if 'identifier' in geom.metadata]
        identifier = max(identifiers, default=-1) + 1
    for geometry in data:  # assign those which were not yet assigned
        geometry, identifier = _assign_identifier(geometry, identifier)
    _write_frames(*data, outputs=[outputs[0]])
    return identifier


assign_identifiers = python_app(_assign_identifiers, executors=["default_threads"])


@typeguard.typechecked
def _join_frames(
    inputs: list = [],
    outputs: list = [],
):
    """
    Join multiple frame files into a single file.

    Args:
        inputs: List of Parsl futures. Each element should be a DataFuture
                representing an input file path containing geometry data.
        outputs: List of Parsl futures. The first element should be a DataFuture
                 representing the output file path where joined frames will be written.

    Returns:
        None

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    assert len(outputs) == 1

    with open(outputs[0], "wb") as destination:
        for input_file in inputs:
            with open(input_file, "rb") as source:
                shutil.copyfileobj(source, destination)


join_frames = python_app(_join_frames, executors=["default_threads"])


@typeguard.typechecked
def _count_frames(inputs: list = []) -> int:
    """
    Count the number of frames in a file.

    Args:
        inputs: List of Parsl futures. The first element should be a DataFuture
                representing the input file path containing geometry data.

    Returns:
        int: Number of frames in the file.

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    assert len(inputs) == 1
    nframes = 0
    for _ in iter_read_frames(inputs[0]):
        nframes += 1
    return nframes


count_frames = python_app(_count_frames, executors=["default_threads"])


@typeguard.typechecked
def _reset_frames(inputs: list = [], outputs: list = []) -> None:
    """
    Reset all frames in a file.

    Args:
        inputs: List of Parsl futures. The first element should be a DataFuture
                representing the input file path containing geometry data.
        outputs: List of Parsl futures. The first element should be a DataFuture
                 representing the output file path where reset frames will be written.

    Returns:
        None

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    assert len(inputs) == len(outputs) == 1
    data = _read_frames(inputs=[inputs[0]])
    for geometry in data:
        if geometry != NULLSTATE:
            geometry.reset()
    _write_frames(*data, outputs=[outputs[0]])


reset_frames = python_app(_reset_frames, executors=["default_threads"])


@typeguard.typechecked
def _clean_frames(inputs: list = [], outputs: list = []) -> None:
    """
    Clean all frames in a file.

    Args:
        inputs: List of Parsl futures. The first element should be a DataFuture
                representing the input file path containing geometry data.
        outputs: List of Parsl futures. The first element should be a DataFuture
                 representing the output file path where cleaned frames will be written.

    Returns:
        None

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    assert len(inputs) == len(outputs) == 1
    data = _read_frames(inputs=[inputs[0]])
    for geometry in data:
        if geometry != NULLSTATE:
            geometry.clean()
    _write_frames(*data, outputs=[outputs[0]])


clean_frames = python_app(_clean_frames, executors=["default_threads"])


@typeguard.typechecked
def _apply_offset(
    subtract: bool,
    inputs: list = [],
    outputs: list = [],
    **atomic_energies: float,
) -> None:
    """
    Apply an energy offset to all frames in a file.

    Args:
        subtract: Whether to subtract or add the offset.
        inputs: List of Parsl futures. The first element should be a DataFuture
                representing the input file path containing geometry data.
        outputs: List of Parsl futures. The first element should be a DataFuture
                 representing the output file path where updated frames will be written.
        **atomic_energies: Atomic energies for each element.

    Returns:
        None

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    assert len(inputs) == 1 and len(outputs) == 1
    data = _read_frames(inputs=[inputs[0]])
    numbers = [atomic_numbers[e] for e in atomic_energies.keys()]
    all_numbers = get_unique_numbers(data)
    all_numbers.discard(0)      # from potential NullState
    assert all(n in numbers for n in all_numbers)

    for geometry in data:
        if geometry == NullState:
            continue
        assert geometry.energy is not None, 'Geometry does not have an energy value'
        energy = get_atomic_energy(geometry, atomic_energies)
        if subtract:
            geometry.energy -= energy
        else:
            geometry.energy += energy
    _write_frames(*data, outputs=[outputs[0]])


apply_offset = python_app(_apply_offset, executors=["default_threads"])


@typeguard.typechecked
def _get_elements(inputs: list = []) -> set[str]:
    """
    Get the set of elements present in all frames of a file.

    Args:
        inputs: List of Parsl futures. The first element should be a DataFuture
                representing the input file path containing geometry data.

    Returns:
        set[str]: Set of element symbols.

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    data = _read_frames(inputs=[inputs[0]])
    return set([chemical_symbols[n] for n in get_unique_numbers(data)])


get_elements = python_app(_get_elements, executors=["default_threads"])


@typeguard.typechecked
def _align_axes(inputs: list = [], outputs: list = []) -> None:
    """
    Align axes for all frames in a file.

    Args:
        inputs: List of Parsl futures. The first element should be a DataFuture
                representing the input file path containing geometry data.
        outputs: List of Parsl futures. The first element should be a DataFuture
                 representing the output file path where aligned frames will be written.

    Returns:
        None

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    assert len(inputs) == len(outputs) == 1
    data = _read_frames(inputs=[inputs[0]])
    for geometry in data:
        if geometry != NULLSTATE:
            geometry.align_axes()
    _write_frames(*data, outputs=[outputs[0]])


align_axes = python_app(_align_axes, executors=["default_threads"])


@typeguard.typechecked
def _not_null(inputs: list = [], outputs: list = []) -> list[bool]:
    """
    Check which frames in a file are not null states.

    Args:
        inputs: List of Parsl futures. The first element should be a DataFuture
                representing the input file path containing geometry data.
        outputs: List of Parsl futures. If provided, the first element should be a DataFuture
                 representing the output file path where non-null frames will be written.

    Returns:
        list[bool]: List of boolean values indicating non-null states.

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    assert len(inputs) == 1 and len(outputs) < 2
    data, mask = [], []
    for geom_str_list in iter_read_frames(inputs[0]):
        natoms, header = int(geom_str_list[0].strip()), geom_str_list[1]
        if natoms == 1 and header.startswith(NullState.key):
            mask.append(False)
        else:
            mask.append(True)
            data.append(''.join(geom_str_list))
    if len(outputs) > 0:
        with open(outputs[0], "w") as f:
            f.write(''.join(data))
    return mask


not_null = python_app(_not_null, executors=["default_threads"])


@typeguard.typechecked
def _app_filter(
    quantity_name: str,
    inputs: list = [],
    outputs: list = [],
) -> None:
    """
    Filter frames based on a specified quantity.

    Args:
        quantity_name: The quantity to filter on.
        inputs: List of Parsl futures. The first element should be a DataFuture
                representing the input file path containing geometry data.
        outputs: List of Parsl futures. The first element should be a DataFuture
                 representing the output file path where filtered frames will be written.

    Returns:
        None

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    assert len(inputs) == 1 and len(outputs) == 1
    assert quantity_name in quantities.all
    data = [geom for geom in _read_frames(inputs=[inputs[0]]) if geom != NULLSTATE]
    quantity = quantities.all[quantity_name]
    default = quantity.default

    out = []
    for geom in data:
        if quantity.type == Type.METADATA and quantity.name in geom.metadata:
            out.append(geom)
            continue
        elif quantity.per_atom:
            value = geom.per_atom[quantity.name]
        else:
            value = getattr(geom, quantity.name)
        if default is not None and np.isnan(default):
            # comparison with NaN is always false
            skip = np.isnan(value)
        else:
            skip = (value == default)
        if isinstance(skip, np.ndarray):
            skip = skip.all()
        if not skip:
            out.append(geom)

    _write_frames(*out, outputs=[outputs[0]])


app_filter = python_app(
    _app_filter, executors=["default_threads"]
)  # filter is protected


@typeguard.typechecked
def _shuffle(
    inputs: list = [],
    outputs: list = [],
) -> None:
    """
    Shuffle the order of frames in a file.

    Args:
        inputs: List of Parsl futures. The first element should be a DataFuture
                representing the input file path containing geometry data.
        outputs: List of Parsl futures. The first element should be a DataFuture
                 representing the output file path where shuffled frames will be written.

    Returns:
        None

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    assert len(inputs) == 1 and len(outputs) == 1
    data = _read_frames(inputs=[inputs[0]])
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    shuffled = [data[int(i)] for i in indices]
    _write_frames(*shuffled, outputs=[outputs[0]])


shuffle = python_app(_shuffle, executors=["default_threads"])


@typeguard.typechecked
def _get_train_valid_indices(
    effective_nstates: int,
    train_valid_split: float,
    shuffle: bool,
) -> tuple[list[int], list[int]]:
    """
    Generate indices for train and validation splits.

    Args:
        effective_nstates: Total number of states.
        train_valid_split: Fraction of states to use for training.
        shuffle: Whether to shuffle the indices.

    Returns:
        tuple[list[int], list[int]]: Lists of indices for training and validation sets.

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    ntrain = int(np.floor(effective_nstates * train_valid_split))
    nvalid = effective_nstates - ntrain
    assert ntrain > 0 and nvalid > 0
    order = np.arange(effective_nstates, dtype=int)
    if shuffle:
        np.random.shuffle(order)
    train, valid = order[:ntrain], order[ntrain: (ntrain + nvalid)]
    return [int(i) for i in train], [int(i) for i in valid]


get_train_valid_indices = python_app(_get_train_valid_indices, executors=["default_threads"])


@typeguard.typechecked
def get_index_element_mask(
    numbers: np.ndarray,
    atom_indices: Optional[list[int]],
    elements: Optional[list[str]],
    natoms_padded: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a mask for atom indices and elements.

    Args:
        numbers: Array of atomic numbers.
        atom_indices: List of atom indices to include.
        elements: List of element symbols to include.
        natoms_padded: Total number of atoms including padding.

    Returns:
        np.ndarray: Boolean mask array.
    """
    mask = np.array([True] * len(numbers))

    if elements is not None:
        numbers_to_include = [atomic_numbers[e] for e in elements]
        mask_elements = np.array([False] * len(numbers))
        for number in numbers_to_include:
            mask_elements = np.logical_or(mask_elements, (numbers == number))
        mask = np.logical_and(mask, mask_elements)

    if natoms_padded is not None:
        assert natoms_padded >= len(numbers)
        padding = natoms_padded - len(numbers)
        mask = np.concatenate((mask, np.array([False] * padding)), axis=0).astype(bool)

    if atom_indices is not None:  # below padding
        mask_indices = np.array([False] * len(mask))
        mask_indices[np.array(atom_indices)] = True
        mask = np.logical_and(mask, mask_indices)

    return mask


@typeguard.typechecked
def _compute_rmse(
    array0: np.ndarray,
    array1: np.ndarray,
    reduce: bool = True,
) -> Union[float, np.ndarray]:
    """
    Compute the Root Mean Square Error (RMSE) between two arrays.

    Args:
        array0: First array.
        array1: Second array.
        reduce: Whether to reduce the result to a single value.

    Returns:
        Union[float, np.ndarray]: RMSE value(s).

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    # TODO: can be simplified greatly with np.nanmean and such
    assert array0.shape == array1.shape
    assert np.all(np.isnan(array0) == np.isnan(array1))

    se = (array0 - array1) ** 2
    se = se.reshape(se.shape[0], -1)

    if reduce:  # across both dimensions
        mask = np.logical_not(np.isnan(se))
        return float(np.sqrt(np.mean(se[mask])))
    else:  # retain first dimension
        if se.ndim == 1:
            return se
        else:
            values = np.empty(len(se))
            for i in range(len(se)):
                if np.all(np.isnan(se[i])):
                    values[i] = np.nan
                else:
                    mask = np.logical_not(np.isnan(se[i]))
                    value = np.sqrt(np.mean(se[i][mask]))
                    values[i] = value
            return values


compute_rmse = python_app(_compute_rmse, executors=["default_threads"])


@typeguard.typechecked
def _compute_mae(
    array0,
    array1,
    reduce: bool = True,
) -> Union[float, np.ndarray]:
    """
    Compute the Mean Absolute Error (MAE) between two arrays.

    Args:
        array0: First array.
        array1: Second array.
        reduce: Whether to reduce the result to a single value.

    Returns:
        Union[float, np.ndarray]: MAE value(s).

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    assert array0.shape == array1.shape
    mask0 = np.logical_not(np.isnan(array0))
    mask1 = np.logical_not(np.isnan(array1))
    assert np.all(mask0 == mask1)
    ae = np.abs(array0 - array1)
    to_reduce = tuple(range(1, array0.ndim))
    mask = np.logical_not(np.all(np.isnan(ae), axis=to_reduce))
    ae = ae[mask0].reshape(np.sum(1 * mask), -1)
    if reduce:  # across both dimensions
        return float(np.sqrt(np.mean(ae)))
    else:  # retain first dimension
        return np.sqrt(np.mean(ae, axis=1))


compute_mae = python_app(_compute_mae, executors=["default_threads"])


@typeguard.typechecked
def _batch_frames(
    batch_size: int,
    inputs: list = [],
    outputs: list = [],
) -> None:
    """
    Split frames into batches.

    Args:
        batch_size: Number of frames per batch.
        inputs: List of Parsl futures. The first element should be a DataFuture
                representing the input file path containing geometry data.
        outputs: List of Parsl futures. Each element should be a DataFuture
                 representing an output file path for each batch.

    Note:
        This function is wrapped as a Parsl app and executed using the default_threads executor.
    """
    assert len(inputs) == 1
    data = []
    batch_index = 0
    for idx, geom_str_list in enumerate(iter_read_frames(inputs[0])):
        data += geom_str_list
        if (idx + 1) % batch_size == 0:
            with open(outputs[batch_index], "w") as f:
                f.write(''.join(data))
                data = []                   # clear frames
            batch_index += 1
    if len(data) > 0:                       # write final partial batch
        with open(outputs[batch_index], "w") as f:
            f.write(''.join(data))
        batch_index += 1
    assert batch_index == len(outputs)


batch_frames = python_app(_batch_frames, executors=["default_threads"])
