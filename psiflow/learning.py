from __future__ import annotations # necessary for type-guarding class methods
from typing import Optional, Tuple
import typeguard
from dataclasses import dataclass
from typing import Optional
import logging

from parsl.app.app import python_app

from psiflow.utils import get_train_valid_indices
from psiflow.data import Dataset
from psiflow.experiment import FlowManager
from psiflow.models import BaseModel
from psiflow.reference import BaseReference
from psiflow.sampling import RandomWalker, PlumedBias
from psiflow.ensemble import Ensemble


logger = logging.getLogger(__name__) # logging per module
logger.setLevel(logging.INFO)


@typeguard.typechecked
@dataclass
class BaseLearning:
    train_valid_split: float = 0.9
    pretraining_nstates: int = 50
    pretraining_amplitude_pos: float = 0.05
    pretraining_amplitude_box: float = 0.05

    def run_pretraining(
            self,
            flow_manager: FlowManager,
            model: BaseModel,
            reference: BaseReference,
            initial_data: Dataset,
            ):
        nstates = self.pretraining_nstates
        amplitude_pos = self.pretraining_amplitude_pos
        amplitude_box = self.pretraining_amplitude_box
        logger.info('performing random pretraining')
        nstates_per_state = (nstates // initial_data.length().result()) + 1
        random_data = Dataset(reference.context, [])
        for i in range(initial_data.length().result()):
            walker = RandomWalker( # create walker for state i in initial data
                    reference.context,
                    initial_data[i],
                    amplitude_pos=amplitude_pos,
                    amplitude_box=amplitude_box,
                    seed=i
                    )
            ensemble = Ensemble.from_walker(walker, nstates_per_state)
            random_data = random_data + ensemble.sample(nstates_per_state)
        data = reference.evaluate(random_data)
        data_success = data.get(indices=data.success)
        train, valid = get_train_valid_indices(
                data_success.length(), # can be less than nstates
                self.train_valid_split,
                )
        data_train = data_success.get(indices=train)
        data_valid = data_success.get(indices=valid)
        data_train.log('data_train')
        data_valid.log('data_valid')
        model.initialize(data_train)
        model.train(data_train, data_valid)
        flow_manager.save(
                name='random_pretraining',
                model=model,
                ensemble=ensemble,
                data_train=data_train,
                data_valid=data_valid,
                data_failed=data.get(indices=data.failed),
                )
        log = flow_manager.log_wandb( # log training
                run_name='random_pretraining',
                model=model,
                ensemble=ensemble,
                data_train=data_train,
                data_valid=data_valid,
                )
        log.result() # necessary to force execution of logging app
        return data_train, data_valid


@typeguard.typechecked
@dataclass
class OnlineLearning(BaseLearning):
    nstates: int = 30
    niterations: int = 10
    retrain_model_per_iteration : bool = True

    def run(
            self,
            flow_manager: FlowManager,
            model: BaseModel,
            reference: BaseReference,
            ensemble: Ensemble,
            data_train: Optional[Dataset] = None,
            data_valid: Optional[Dataset] = None,
            checks: Optional[list] = None,
            ) -> Tuple[Dataset, Dataset]:
        if data_train is None:
            data_train = Dataset(model.context, [])
        if data_valid is None:
            data_valid = Dataset(model.context, [])
        for i in range(self.niterations):
            if flow_manager.output_exists(str(i)):
                continue # skip iterations in case of restarted run
            model.deploy()
            dataset = ensemble.sample(
                    self.nstates,
                    model=model,
                    checks=checks,
                    )
            data = reference.evaluate(dataset)
            data_success = data.get(indices=data.success)
            train, valid = get_train_valid_indices(
                    data_success.length(), # can be less than nstates
                    self.train_valid_split,
                    )
            data_train.append(data_success.get(indices=train))
            data_valid.append(data_success.get(indices=valid))
            if self.retrain_model_per_iteration:
                logger.info('reinitialize model (scale/shift/avg_num_neighbors) on new training data')
                model.reset()
                model.initialize(data_train)
            data_train.log('data_train')
            data_valid.log('data_valid')
            model.train(data_train, data_valid)
            flow_manager.save(
                    name=str(i),
                    model=model,
                    ensemble=ensemble,
                    data_train=data_train,
                    data_valid=data_valid,
                    data_failed=data.get(indices=data.failed),
                    )
            log = flow_manager.log_wandb( # log training
                    run_name=str(i),
                    model=model,
                    ensemble=ensemble,
                    data_train=data_train,
                    data_valid=data_valid,
                    bias=ensemble.biases[0], # possibly None
                    )
            log.result() # necessary to force execution of logging app
        return data_train, data_valid
