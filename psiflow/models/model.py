from __future__ import annotations  # necessary for type-guarding class methods

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Union

import parsl
import typeguard
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.data import Dataset
from psiflow.utils import copy_data_future, log_message, resolve_and_check, save_yaml

logger = logging.getLogger(__name__)  # logging per module


@typeguard.typechecked
class Model:
    """Base Container for a trainable interaction potential"""

    def __init__(self) -> None:
        self.config = None
        self.model_future = None
        self.atomic_energies = {}

    def add_atomic_energy(self, element: str, energy: Union[float, AppFuture]) -> None:
        assert self.model_future is None, (
            "cannot add atomic energies after model has "
            "been initialized; reset model, add energy, and reinitialize"
        )
        if element in self.atomic_energies:
            if isinstance(energy, AppFuture):
                energy = energy.result()
            if isinstance(self.atomic_energies[element], AppFuture):
                existing = self.atomic_energies[element].result()
            assert energy == existing, (
                "model already has atomic energy "
                "for element {} ({}), which is different from {}"
                "".format(element, existing, energy)
            )
        self.atomic_energies[element] = energy

    def train(self, training: Dataset, validation: Dataset) -> None:
        log_message(
            "training model using {} states for training and {} for validation",
            training.length(),
            validation.length(),
        )
        inputs = [self.model_future]
        if self.do_offset:
            inputs += [
                training.subtract_offset(**self.atomic_energies).data_future,
                validation.subtract_offset(**self.atomic_energies).data_future,
            ]
        else:
            inputs += [
                training.data_future,
                validation.data_future,
            ]
        future = self._train(
            asdict(self.config),
            stdout=parsl.AUTO_LOGNAME,
            stderr=parsl.AUTO_LOGNAME,
            inputs=inputs,
            outputs=[psiflow.context().new_file("model_", ".pth")],
        )
        self.model_future = future.outputs[0]

    def initialize(self, dataset: Dataset) -> None:
        """Initializes the model based on a dataset"""
        assert self.model_future is None
        if self.do_offset:
            inputs = [dataset.subtract_offset(**self.atomic_energies).data_future]
        else:
            inputs = [dataset.data_future]
        future = self._initialize(
            asdict(self.config),
            stdout=parsl.AUTO_LOGNAME,
            stderr=parsl.AUTO_LOGNAME,
            inputs=inputs,
            outputs=[psiflow.context().new_file("model_", ".pth")],
        )
        self.model_future = future.outputs[0]

    def reset(self) -> None:
        self.model_future = None

    def save(
        self,
        path: Union[Path, str],
    ) -> None:
        path = resolve_and_check(Path(path))
        path.mkdir(exist_ok=True)

        name = self.__class__.__name__
        path_config = path / "{}.yaml".format(name)

        atomic_energies = {
            "atomic_energies_" + key: value
            for key, value in self.atomic_energies.items()
        }
        save_yaml(
            asdict(self.config),
            outputs=[File(str(path_config))],
            **atomic_energies,
        )
        if self.model_future is not None:
            path_model = path / "{}.pth".format(name)
            copy_data_future(
                inputs=[self.model_future],
                outputs=[File(str(path_model))],
            )

    def copy(self) -> Model:
        model = self.__class__(**asdict(self.config))
        for element, energy in self.atomic_energies.items():
            model.add_atomic_energy(element, energy)
        if self.model_future is not None:
            model.model_future = copy_data_future(
                inputs=[self.model_future],
                outputs=[psiflow.context().new_file("model_", ".pth")],
            ).outputs[0]
        return model

    @property
    def do_offset(self) -> bool:
        return len(self.atomic_energies) > 0

    @property
    def seed(self) -> int:
        raise NotImplementedError

    @seed.setter
    def seed(self, arg) -> None:
        raise NotImplementedError