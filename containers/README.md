# Container setup

**Note**: up until version 4.0.0, containers were created using Docker, and then converted to Singularity/Apptainer images. 
The Dockerfiles are still available in this repository ('legacy' prefix), but are no longer maintained. 

Psiflow provides pre-built containers for portability across systems, but they are in no means necessary. 
If your cluster has software specifically compiled for its hardware, it might be preferable to use that.
This is especially true for quantum chemistry codes.

## Available containers

Every container Psiflow uses is available on the GitHub Container Registry 
([here](https://github.com/orgs/molmod/packages)). 
You can specify a container by its name in the `config.yaml` as:
```
container_uri: oras://ghcr.io/molmod/{NAME}:{TAG}
```
This will download the container and cache it for further invocations.
Alternatively, you can directly specify a `PATH` to the specific image to use. 

An overview of existing containers:
- **psiflow**: use for ModelEvaluation and ModelTraining tasks. \
It contains psiflow, Parsl, MACE, PyTorch, i-PI and PLUMED. It comes in CUDA and ROCM flavours for different GPU hardware.
Changes to the psiflow source code (e.g. new features, bug fixes) will not be automatically reflected in this container.
To guarantee compatibility, recompile with all new commits (see [below](#creating-custom-containers)).

- **cp2k**: use for CP2K Reference calculations.

- **gpaw**: use for GPAW Reference calculations. \
Currently contains a psiflow dependency, which will be removed in future versions.

[//]: # (- **orca**: )

## Creating custom containers

If you develop psiflow, or the latest PR has a feature you need but the psiflow container is not yet up-to-date,
you can quickly patch it for personal use. To do so, you need an [Apptainer](https://apptainer.org/)/[Singularity](https://sylabs.io/singularity/) installation 
(which is probably available on your HPC, otherwise you cannot run containers anyway) and a [Definition File](https://docs.sylabs.io/guides/latest/user-guide/definition_files.html),
which specifies a build recipe. The building process is a simple one-liner:
```bash
apptainer/singularity build {OUTPUT_IMAGE_NAME}.sif {DEFINITION_FILE}
```

A very basic definition file is shown below:

```
Bootstrap: oras
From: ghcr.io/molmod/psiflow:v4.0.0_cu118
Stage: build

%arguments
    PSIFLOW_REPO="https://github.com/molmod/psiflow@main"
        
%post
    pip uninstall -y psiflow
    pip install --no-cache-dir git+{{ PSIFLOW_REPO }}
        
%test
    pip list
    
%help
    A patched version of the psiflow container.
```

Here, we start from the outdated `psiflow:v4.0.0_cu118` container and 
replace its psiflow installation with the latest version from the `main` branch.
Of course, you can make any changes you want in the `%post` section 
(e.g., updating other packages, installing your own software). 
Alternatively, you can build from scratch. 

Reach out if this gives you any troubles (although we are no container experts either).
