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
    for scenario in Server Offline; do
        for gpu_batch_size in 1 2 4 8; do
            for qps in 0.2 0.3 0.5 0.7 0.9 1.1 1.3 1.5; do
                echo "run scenario=${scenario} bs=${gpu_batch_size} gpus=${CVD} qps=${qps}"
                LOGDIR="build/logs/benchmark/scenario-${scenario}-bs-${gpu_batch_size}-gpus-${CVD}-qps-${qps}"
                mkdir -p $LOGDIR
                LOG_OUT=${LOGDIR}/stdout.log
                LOG_ERR=${LOGDIR}/stderr.log
                echo "stdout: ${LOG_OUT}"
                echo "stderr: ${LOG_ERR}"
                LD_LIBRARY_PATH=$CONDA_PREFIX/lib \
                    MLPERF_SCRATCH_PATH=$PWD \
                    CUDA_VISIBLE_DEVICES=${CVD} \
                    make run_harness RUN_ARGS="--benchmarks=stable-diffusion-xl --scenarios=${scenario} --test_mode=PerformanceOnly --fast --log_dir=${LOGDIR} --gpu_batch_size=${gpu_batch_size} --server_target_qps=${qps}" \
                    > $LOG_OUT 2> $LOG_ERR \
                    && echo "Running benchmark done" || echo "Running benchmark failed"
            done;
        done;
    done;
done;

