#!/bin/bash

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

declare -a CVDS=( "0" )
if [ ${NUM_GPUS} -ge 2 ]; then
  CVDS+=( "0,1" )
fi
if [ ${NUM_GPUS} -ge 4 ]; then
  CVDS+=( "0,1,2,3" )
fi
if [ ${NUM_GPUS} -ge 8 ]; then
  CVDS+=( "0,1,2,3,4,5,6,7" )
fi

echo "CUDA_VISIBLE_DEVICES = ${CVDS[*]}"

for CVD in "${CVDS[@]}"; do
    for scenario in Offline Server; do
        for gpu_batch_size in 1 2 4 8; do
            echo "build scenario=${scenario} bs=${gpu_batch_size} gpus=${CVD}"
            LD_LIBRARY_PATH=$CONDA_PREFIX/lib
                CPATH=$CONDA_PREFIX/include \
                CXXPATH=$CONDA_PREFIX/include \
                MLPERF_SCRATCH_PATH=$PWD CUDA_VISIBLE_DEVICES=${CVD} \
                make generate_engines RUN_ARGS="--benchmarks=stable-diffusion-xl --scenarios=Server --gpu_batch_size=${gpu_batch_size}" \
                && echo "Generating engine done" || echo "Generating engine failed"
            done;
    done;
done;

