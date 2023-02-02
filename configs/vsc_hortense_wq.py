from parsl.channels import LocalChannel

from psiflow.utils import SlurmProvider # fixed SlurmProvider

from psiflow.models import MACEModel, NequIPModel
from psiflow.reference import CP2KReference
from psiflow.execution import ModelEvaluationExecution, ModelTrainingExecution, \
        ReferenceEvaluationExecution
from psiflow.execution import generate_parsl_config


# psiflow definitions
model_evaluate = ModelEvaluationExecution(
        executor='model',
        device='cpu',
        ncores=4,
        dtype='float32',
        )
model_training = ModelTrainingExecution( # forced cuda/float32
        executor='training',
        ncores=12, # number of cores per GPU on gpu_rome_a100 partition
        walltime=1, # in minutes
        )
reference_evaluate = ReferenceEvaluationExecution(
        executor='reference',
        device='cpu',
        ncores=32,
        omp_num_threads=1,
        mpi_command=lambda x: f'mympirun', # use vsc wrapper
        cp2k_exec='cp2k.psmp',
        walltime=1, # in minutes
        )
definitions = {
        MACEModel: [model_evaluate, model_training],
        NequIPModel: [model_evaluate, model_training],
        CP2KReference: [reference_evaluate],
        }


providers = {}


# define provider for default executor (HTEX)
channel = LocalChannel(script_dir=str(path_internal / 'local_script_dir'))
worker_init =  'ml PLUMED/2.7.2-foss-2021a\n'
worker_init += 'ml psiflow-develop/10Jan2023-CPU\n'
provider = SlurmProvider(
        partition='cpu_rome',
        account='2022_050',
        channel=channel,
        nodes_per_block=1,
        cores_per_node=16,
        init_blocks=1,
        min_blocks=1,
        max_blocks=1,
        parallelism=1,
        walltime='02:00:00',
        worker_init=worker_init,
        exclusive=False,
        )
providers['default'] = provider


# define provider for executing model evaluations (e.g. MD)
worker_init =  'ml cctools/7.4.16-GCCcore-10.3.0\n'
worker_init += 'ml PLUMED/2.7.2-foss-2021a\n'
worker_init += 'ml psiflow-develop/10Jan2023-CPU\n'
worker_init += 'export OMP_NUM_THREADS={}\n'.format(model_evaluate.ncores)
provider = SlurmProvider(
        partition='cpu_rome',
        account='2022_050',
        channel=channel,
        nodes_per_block=1,
        cores_per_node=8,
        init_blocks=0,
        min_blocks=0,
        max_blocks=512,
        parallelism=1,
        walltime='02:00:00',
        worker_init=worker_init,
        exclusive=False,
        )
providers['model'] = provider


# define provider for executing model training
worker_init =  'ml cctools/7.4.16-GCCcore-10.3.0\n'
worker_init += 'ml PLUMED/2.7.2-foss-2021a\n'
worker_init += 'ml psiflow-develop/10Jan2023-CUDA-11.3.1\n'
worker_init += 'unset SLURM_CPUS_PER_TASK\n'
worker_init += 'export SLURM_NTASKS_PER_NODE={}\n'.format(model_training.ncores)
worker_init += 'export SLURM_TASKS_PER_NODE={}\n'.format(model_training.ncores)
worker_init += 'export SLURM_NTASKS={}\n'.format(model_training.ncores)
worker_init += 'export SLURM_NPROCS={}\n'.format(model_training.ncores)
worker_init += 'export OMP_NUM_THREADS={}\n'.format(model_training.ncores)
provider = SlurmProvider(
        partition='gpu_rome_a100',
        account='2022_050',
        channel=channel,
        nodes_per_block=1,
        cores_per_node=12,
        init_blocks=0,
        min_blocks=0,
        max_blocks=4,
        parallelism=1.0,
        walltime='01:05:00',
        worker_init=worker_init,
        exclusive=False,
        scheduler_options='#SBATCH --gpus=1\n#SBATCH --cpus-per-gpu=12\n#SBATCH --export=None', # request gpu
        )
providers['training'] = provider


# to get MPI to recognize the available slots correctly, it's necessary
# to override the slurm variables as set by the jobscript, as these are
# based on the number of parsl tasks, NOT on the number of MPI tasks for
# cp2k. Essentially, this means we have to reproduce the environment as
# if we launched a job using 'qsub -l nodes=1:ppn=cores_per_singlepoint'
worker_init =  'ml cctools/7.4.16-GCCcore-10.3.0\n'
worker_init += 'ml vsc-mympirun\n'
worker_init += 'ml CP2K/8.2-foss-2021a\n'
worker_init += 'ml psiflow-develop/10Jan2023-CPU\n'
worker_init += 'unset SLURM_CPUS_PER_TASK\n'
worker_init += 'export SLURM_NTASKS_PER_NODE={}\n'.format(reference_evaluate.ncores)
worker_init += 'export SLURM_TASKS_PER_NODE={}\n'.format(reference_evaluate.ncores)
worker_init += 'export SLURM_NTASKS={}\n'.format(reference_evaluate.ncores)
worker_init += 'export SLURM_NPROCS={}\n'.format(reference_evaluate.ncores)
#worker_init += 'export OMP_NUM_THREADS=1\n'
provider = SlurmProvider(
        partition='cpu_rome',
        account='2022_050',
        channel=channel,
        nodes_per_block=1,
        cores_per_node=reference_evaluate.ncores, # 1 worker per block
        init_blocks=0,
        min_blocks=0,
        max_blocks=16,
        parallelism=1,
        walltime='01:00:00',
        worker_init=worker_init,
        exclusive=False,
        )


def get_config(path_parsl_internal):
    config = generate_parsl_config(
            path_parsl_internal,
            definitions,
            providers,
            use_work_queue=True,
            )
    return config, definitions
