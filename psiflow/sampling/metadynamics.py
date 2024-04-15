from __future__ import annotations  # necessary for type-guarding class methods

from pathlib import Path
from typing import Optional, Union

import typeguard
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture

import psiflow
from psiflow.hamiltonians._plumed import remove_comments_printflush, set_path_in_plumed
from psiflow.utils import copy_app_future


@typeguard.typechecked
@psiflow.serializable
class Metadynamics:
    _plumed_input: str
    external: Optional[psiflow._DataFuture]

    def __init__(
        self,
        plumed_input: str,
        external: Union[None, str, Path, psiflow._DataFuture] = None,
    ):
        assert "METAD" in plumed_input
        if "RESTART" not in plumed_input:
            plumed_input = "RESTART\n" + plumed_input
        _plumed_input = remove_comments_printflush(plumed_input)

        if type(external) in [str, Path]:
            external = File(str(external))
        if external is None:
            external = psiflow.context().new_file("hills_", ".txt.")
        else:
            assert external.filepath in _plumed_input
        _plumed_input = set_path_in_plumed(
            _plumed_input,
            "METAD",
            "PLACEHOLDER",
        )
        self._plumed_input = _plumed_input
        self.external = external

    def plumed_input(self):
        plumed_input = self._plumed_input
        if self.external is not None:
            plumed_input = plumed_input.replace("PLACEHOLDER", self.external.filepath)
        return plumed_input

    def input(self) -> AppFuture:
        return copy_app_future(self.plumed_input(), inputs=[self.external])

    def wait_for(self, result: AppFuture) -> None:
        self.external = copy_app_future(
            0,
            inputs=[result, self.external],
            outputs=[File(self.external.filepath)],
        ).outputs[0]

    def reset(self) -> None:
        self.external = psiflow.context().new_file("hills_", ".txt")

    def __eq__(self, other) -> bool:
        if type(other) is not Metadynamics:
            return False
        return self.plumed_input() == other.plumed_input()

    def copy(self) -> Metadynamics:
        return Metadynamics(str(self.plumed_input()))