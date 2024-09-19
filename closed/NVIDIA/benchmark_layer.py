import torch
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import nvtx

import os
import re

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq

BATCH_SIZE = 8192
INPUT_SHAPE = (BATCH_SIZE, 640)
OUTPUT_SHAPE = (BATCH_SIZE, 5120)
DUMMY_INPUT = torch.randn(*INPUT_SHAPE)
FLOPS = BATCH_SIZE * INPUT_SHAPE[-1] * OUTPUT_SHAPE[-1] * 2

# BATCH_SIZE = 2
# INPUT_SHAPE = (2, 1280, 64, 64)
# OUTPUT_SHAPE = (2, 1280, 64, 64)
# WEIGHT_SHAPE = (1280, 1280, 3, 3)
# DUMMY_INPUT = torch.randn(*INPUT_SHAPE)
# FLOPS = 2 * BATCH_SIZE * WEIGHT_SHAPE[0] * WEIGHT_SHAPE[1] * WEIGHT_SHAPE[2] * WEIGHT_SHAPE[3] * OUTPUT_SHAPE[2] * OUTPUT_SHAPE[3]

IS_FP8 = False

from modelopt.torch.quantization import utils as quant_utils
from modelopt.torch.quantization.calib.max import MaxCalibrator


class BenchmarkLayer(torch.nn.Module):
    def __init__(self):
        super(BenchmarkLayer, self).__init__()
        self.layer = torch.nn.Linear(INPUT_SHAPE[-1], OUTPUT_SHAPE[-1])
        # self.layer = torch.nn.Conv2d(WEIGHT_SHAPE[0], WEIGHT_SHAPE[1], (WEIGHT_SHAPE[2], WEIGHT_SHAPE[3]), 1, 1)

    def forward(self, x):
        x = self.layer(x)
        return x


def build_engine(onnx_file_path, data, cache_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder_flag = builder.create_builder_config()

        if IS_FP8:
            # enable fp8
            builder_flag.set_flag(trt.BuilderFlag.FP8)
        else:
            # enable int8
            builder_flag.set_flag(trt.BuilderFlag.INT8)

        builder_flag.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        builder_flag.builder_optimization_level = 5
        # print(builder_flag.get_memory_pool_limit(trt.MemoryPoolType.TACTIC_SHARED_MEMORY))
        # print(builder_flag.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE))
        # print(builder_flag.hardware_compatibility_level)
        # builder_flag.hardware_compatibility_level = trt.HardwareCompatibilityLevel.AMPERE_PLUS

        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        plan = builder.build_serialized_network(network, builder_flag)
        return plan


def quantize_lvl(unet, quant_level=2.5):
    """
    We should disable the unwanted quantizer when exporting the onnx
    Because in the current modelopt setting, it will load the quantizer amax for all the layers even
    if we didn't add that unwanted layer into the config during the calibration
    """
    for name, module in unet.named_modules():
        if isinstance(module, (torch.nn.Conv2d, )):
            module.input_quantizer.enable()
            module.weight_quantizer.enable()
        elif isinstance(module, (torch.nn.Linear, )):
            module.input_quantizer.enable()
            module.weight_quantizer.enable()
            # if (
            #     (quant_level >= 2 and "ff.net" in name)
            #     or (quant_level >= 2.5 and ("to_q" in name or "to_k" in name or "to_v" in name))
            #     or quant_level == 3
            # ):
            #     module.input_quantizer.enable()
            #     module.weight_quantizer.enable()
            # else:
            #     module.input_quantizer.disable()
            #     module.weight_quantizer.disable()


if __name__ == "__main__":
    cuda.init()
    device = cuda.Device(0)  # enter your Gpu id here
    ctx = device.make_context()

    module = BenchmarkLayer()

    def forward_loop(*_):
        for _ in range(10):
            module(DUMMY_INPUT)
    mtq.FP8_DEFAULT_CFG["quant_cfg"]["default"]["axis"] = None
    if IS_FP8:
        quant_config = {
            'quant_cfg': {
                '*weight_quantizer': {
                    'num_bits': (4, 3), 'axis': None
                },
                '*input_quantizer': {
                    'num_bits': (4, 3), 'axis': None
                },
                '*lm_head*': {'enable': False},
                '*block_sparse_moe.gate*': {'enable': False},
                '*output_layer*': {'enable': False},
                'default': {'num_bits': (4, 3), 'axis': None}
            },
            'algorithm': 'max',
        }
    else:
        quant_config = {
            "quant_cfg": {
                "default": {"num_bits": 8, "axis": None},
            },
            "algorithm": {"method": "smoothquant", "alpha": 1.0},
        }

    mtq.quantize(module, quant_config, forward_loop=forward_loop)
    quantize_lvl(module, 2.5)

    # export onnx
    torch.onnx.export(module, DUMMY_INPUT, "benchmark_layer.onnx", verbose=True)

    # tensorrt
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    # benchmark layer
    data = DUMMY_INPUT.numpy()

    plan = build_engine("benchmark_layer.onnx", data, "tmp")
    with trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(plan)
        inspector = engine.create_engine_inspector()
        print(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
        context = engine.create_execution_context()

    # prepare input
    input_data = np.random.rand(*INPUT_SHAPE).astype(np.float32)
    output_data = np.empty(OUTPUT_SHAPE, dtype=np.float32)

    num_warmup = 1
    num_iter = 2
    num_bindings = 8

    # inference
    d_inputs = [cuda.mem_alloc(input_data.nbytes) for _ in range(num_bindings)]
    d_outputs = [cuda.mem_alloc(output_data.nbytes) for _ in range(num_bindings)]
    bindings = [(int(d_input), int(d_output)) for d_input, d_output in zip(d_inputs, d_outputs)]
    stream = cuda.Stream()
    for d_input in d_inputs:
        cuda.memcpy_htod_async(d_input, input_data, stream)
    bind_iter = 0
    for _ in range(num_warmup):
        context.execute_async_v2(bindings=bindings[bind_iter], stream_handle=stream.handle)
        bind_iter += 1
        if bind_iter == num_bindings:
            bind_iter = 0
    start_time = cuda.Event()
    start_time.record()

    rng = nvtx.start_range(message="DoProfile")
    for _ in range(num_iter):
        context.execute_async_v2(bindings=bindings[bind_iter], stream_handle=stream.handle)
        bind_iter += 1
        if bind_iter == num_bindings:
            bind_iter = 0

    end_time = cuda.Event()
    end_time.record()
    cuda.memcpy_dtoh_async(output_data, d_outputs[0], stream)
    stream.synchronize()
    nvtx.end_range(rng)

    duration = start_time.time_till(end_time)
    duration_per_iter = duration / num_iter

    print(output_data)
    print("Latency: {:.2f} ms".format(duration_per_iter))

    print("FLOPS: {:.2f} TFLOPS".format(FLOPS / duration_per_iter / 1e9))

    ctx.pop()
