FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    build-essential \
    cmake \
    wget \
    git \
    zlib1g-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    libgsl-dev \
    perl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip

# Install PLUMED
ARG PLUMED_VERSION
RUN wget https://github.com/plumed/plumed2/archive/refs/tags/v${PLUMED_VERSION}.tar.gz \
    && tar -xzf v${PLUMED_VERSION}.tar.gz \
    && cd plumed2-${PLUMED_VERSION} \
    && ./configure --prefix=/usr/local/plumed \
    && make -j$(nproc) \
    && make install \
    && cd .. \
    && rm -rf plumed2-${PLUMED_VERSION} v${PLUMED_VERSION}.tar.gz

# Ensure cctools can find the Python environment
ENV PYTHONPATH="/opt/venv/lib/python3.10/site-packages:$PYTHONPATH"
ENV PATH="/opt/venv/bin:$PATH"

# Install cctools
ARG CCTOOLS_VERSION
RUN wget https://github.com/cooperative-computing-lab/cctools/archive/refs/tags/release/${CCTOOLS_VERSION}.tar.gz \
    && tar -xzf ${CCTOOLS_VERSION}.tar.gz \
    && cd cctools-release-${CCTOOLS_VERSION} \
    && ./configure --prefix=/usr/local/cctools \
    && make -j$(nproc) \
    && make install \
    && cd .. \
    && rm -rf cctools-release-${CCTOOLS_VERSION} ${CCTOOLS_VERSION}.tar.gz

# Set environment variables for PLUMED and cctools
ENV PATH="/usr/local/plumed/bin:/usr/local/cctools/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/plumed/lib:/usr/local/cctools/lib:$LD_LIBRARY_PATH"

# Create entrypoint script
RUN echo '#!/bin/bash\nsource /opt/venv/bin/activate\nexec "$@"' > /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ARG PSIFLOW_VERSION
ARG PARSL_VERSION
ARG GPU_LIBRARY
RUN /bin/bash -c -o pipefail \
    "source /opt/venv/bin/activate && \
     pip install --no-cache-dir wandb plotly plumed && \
     pip install --no-cache-dir git+https://github.com/lab-cosmo/i-pi.git@feat/socket_prefix && \
     pip install --no-cache-dir git+https://github.com/molmod/psiflow.git@${PSIFLOW_VERSION} && \
     pip install --no-cache-dir torch==2.1 --index-url https://download.pytorch.org/whl/${GPU_LIBRARY} && \
     pip install --no-cache-dir git+https://github.com/acesuit/mace.git@v0.3.3"

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command
CMD ["bash"]
