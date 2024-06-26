# syntax = docker/dockerfile:experimental
# based onhttps://github.com/pytorch/pytorch/blob/master/Dockerfile
# 
# NOTE: To build this you will need a docker version > 18.06 with
#       experimental enabled and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference: 
#           https://docs.docker.com/develop/develop-images/build_enhancements/

ARG BENCHMARK=retinanet
ARG IMPL=pytorch-cpu
ARG BASE_IMAGE=rockylinux:8.7
FROM ${BASE_IMAGE} AS dev-base
RUN --mount=type=cache,id=yum-dev,target=/var/cache/yum \
    DEBIAN_FRONTEND=noninteractive dnf install -y \
    ca-certificates \
    git \
    vim \
    numactl \
    cmake \
    sudo \
    wget \
    mesa-libGL \
    gcc-toolset-11-gcc \
    gcc-toolset-11-gcc-c++ \
    && rm -rf /var/lib/yum/lists/*

RUN git config --global user.name "user" && git config --global user.email "user@example.com"
RUN echo "alias ll='ls -l'" >> /root/.bashrc
RUN echo "source /opt/rh/gcc-toolset-11/enable" >> /root/.bashrc
RUN mkdir /opt/ccache
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda
ARG PYTHON_VERSION=3.9
RUN wget -O ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} && \
    /opt/conda/bin/conda config --add channels intel && \
    /opt/conda/bin/conda install -c conda-forge cython==0.29.33 -y && \
    /opt/conda/bin/conda install -c conda-forge setuptools==65.5.1 -y && \
    /opt/conda/bin/conda install -c conda-forge cmake==3.26.4 -y && \
    /opt/conda/bin/conda install -c intel mkl=2022.0.1 mkl-include=2022.0.1 intel-openmp=2022.2.1 -y && \
    /opt/conda/bin/conda install -c conda-forge llvm-openmp=8.0.1 jemalloc=5.2.1 -y && \
    /opt/conda/bin/conda install -c conda-forge numpy==1.23.5 -y && \
    /opt/conda/bin/conda install -c conda-forge pycocotools==2.0.2 absl-py==1.4.0 wheel==0.38.1 future==0.18.3 -y && \
    /opt/conda/bin/conda install -c conda-forge pillow==9.5.0 pybind11==2.11.1 -y && \
    /opt/conda/bin/conda clean -ya

FROM dev-base AS build
COPY --from=conda /opt/conda /opt/conda
ARG PYTORCH_VERSION=v1.12.0-rc7
RUN --mount=type=cache,target=/opt/ccache \
    source /opt/rh/gcc-toolset-11/enable && \
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"} && \
    #pip install Pillow==9.5.0 pycocotools==2.0.2 opencv-python==4.7.0.72 absl-py==1.4.0 fiftyone==0.16.5 wheel==0.38.1 future==0.18.3 pybind11 && \
    pip install opencv-python==4.7.0.72 fiftyone==0.16.5 pybind11 && \
    git clone https://github.com/pytorch/pytorch.git pytorch && cd pytorch && \
    git checkout ${PYTORCH_VERSION} && git submodule sync && git submodule update --init --recursive && \
    git fetch origin pull/89925/head && \
    git cherry-pick 78cad998e505b667d25ac42f8aaa24409f5031e1 && \
    python setup.py install && cd ..

FROM dev-base as mp-base
ARG BENCHMARK
ARG IMPL
COPY --from=build /opt/conda /opt/conda
ENV CONDA_PREFIX "/opt/conda"
WORKDIR /opt/workdir
COPY ./code/${BENCHMARK} code/${BENCHMARK}
COPY ./code/run_clean.sh code/${BENCHMARK}/${IMPL}/run_clean.sh
COPY ./code/user_config.py code/user_config.py
#COPY ./calibration/retinanet calibration/retinanet
ARG IPEX_VERSION=mlperf/retinanet
ARG INFERENCE=master
ARG VISION_VERSION=8e078971b8aebdeb1746fea58851e3754f103053
ARG OPENCV_VERSION=1ef8175c494dbf9e11fee5913272a2bcc2f6002f
RUN source /opt/rh/gcc-toolset-11/enable && \
    cd code/${BENCHMARK}/${IMPL} && mkdir retinanet-env && cd retinanet-env && \
    git clone https://github.com/pytorch/vision && \
    cd vision && git checkout ${VISION_VERSION} && python setup.py install && cd .. && \
    git clone https://github.com/intel/intel-extension-for-pytorch.git ipex-cpu-dev && \
    cd ipex-cpu-dev && git checkout ${IPEX_VERSION} && \
    git submodule sync && git submodule update --init --recursive && \
    git apply /opt/workdir/code/${BENCHMARK}/${IMPL}/runtime_ignore_dequant_check.patch && \
    python setup.py install && \
    export IPEX_PATH=${PWD}/build/Release/packages/intel_extension_for_pytorch && cd .. && \
    git clone https://github.com/mlcommons/inference.git mlperf_inference && \
    cd mlperf_inference && git checkout ${INFERENCE} && cp mlperf.conf ../../mlperf.conf && \
    cd loadgen && mkdir build && cd build && cmake .. && make && cd .. && \
    CFLAGS="-std=c++14" python setup.py install && export LOADGEN_DIR=${PWD} && cd ../.. && \
    git clone https://github.com/opencv/opencv.git && cd opencv && git checkout ${OPENCV_VERSION} && \
    mkdir build && cd build && cmake -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -DBUILD_LIST=highgui,imgcodecs,imgproc .. && make -j$(nproc) && \
    export OPENCV_DIR=${PWD} && cd ../.. && \
    git clone https://github.com/Tencent/rapidjson.git && cd rapidjson && git checkout e4bde977 && \
    export RAPIDJSON_INCLUDE_DIR=${PWD}/include && cd .. && \
    git clone https://github.com/gflags/gflags.git && cd gflags && git checkout a738fdf93 && mkdir build && cd build && \
    cmake .. && make && export GFLAGS_DIR=${PWD} && cd ../.. && \
    export LD_LIBRARY_PATH=${OPENCV_DIR}/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH} && \
    export LIBRARY_PATH=${CONDA_PREFIX}/lib:${LIBRARY_PATH} && \
    rm -rf /opt/conda/lib/cmake/mkl && \
    export TORCH_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` && \
    cd .. && \
    cmake -DCMAKE_PREFIX_PATH=${TORCH_PATH} \
        -DLOADGEN_DIR=${LOADGEN_DIR} \
        -DOpenCV_DIR=${OPENCV_DIR} \
        -DRapidJSON_INCLUDE_DIR=${RAPIDJSON_INCLUDE_DIR} \
        -Dgflags_DIR=${GFLAGS_DIR} \
        -DINTEL_EXTENSION_FOR_PYTORCH_PATH=${IPEX_PATH} \
        -B${PWD}/build \
        -H${PWD}/src && \
    cmake --build ${PWD}/build --config Release

ENV MALLOC_CONF "oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
ENV LD_PRELOAD "/opt/conda/lib/libjemalloc.so":"/opt/conda/lib/libiomp5.so":${LD_PRELOAD}
ENV ENV_DEPS_DIR "/opt/workdir/code/${BENCHMARK}/${IMPL}"
ENV CONDA_PREFIX "/opt/conda"
