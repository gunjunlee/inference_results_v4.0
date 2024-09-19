# %%
import sys
sys.path.append("code/stable-diffusion-xl/tensorrt")
if "" in sys.path:
    sys.path.remove("")
sys.path.insert(0, "")
if "code" in sys.modules:
    del sys.modules["code"]
print(sys.path)


# %%

# from backend import SDXLEngine, SDXLBufferManager
# from utilities import calculate_max_engine_device_memory
from infer import TRTTester
from utilities import CUASSERT
from utilities import PipelineConfig
from utilities import nvtx_profile_start
from utilities import nvtx_profile_stop

import nvtx
import torch
import tensorrt as trt
from time import time
from pathlib import Path
from cuda import cudart


def ppath(f):
    try:
        import inspect
        def _ppath(f):
            if isinstance(f, type):
                return inspect.getfile(f) + ":" + str(inspect.getsourcelines(f)[1])
            return _ppath(f.__class__)
        print("code -g " + _ppath(f))
    except Exception as e:
        print(f"cannot get path for {f}: {e}")


class Tester(TRTTester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nvtx_profile = False
        self.markers = {}

    def _denoise_latent(self, actual_batch_size):
        # Prepare predetermined input tensors
        if self.seed:
            generator = torch.Generator(device=self.device).manual_seed(self.seed)
            latents_shape = self.models['unet'].get_shape_dict(actual_batch_size)['sample']
            latents = torch.randn(latents_shape, device=self.device, dtype=self.latent_dtype, generator=generator)  # TODO modify dtype after we switch to quantized engine
        else:
            latents = self.init_noise_latent
            latents = torch.concat([latents] * actual_batch_size)
        latents = latents * self.scheduler.init_noise_sigma()
        encoder_hidden_states = torch.concat([
            self.buffers['clip1_hidden_states'],
            self.buffers['clip2_hidden_states'].to(self.latent_dtype)
        ], dim=-1)
        text_embeds = self.buffers['clip2_text_embeddings'].to(self.latent_dtype)

        self.buffers['unet_encoder_hidden_states'] = encoder_hidden_states
        self.buffers['unet_text_embeds'] = text_embeds
        self.buffers['unet_time_ids'] = self._get_time_ids(actual_batch_size)

        for step_index, timestep in enumerate(self.scheduler.timesteps):
            # Expand the latents because we have prompt and negative prompt guidance
            latents_expanded = self.scheduler.scale_model_input(torch.concat([latents] * 2), step_index, timestep)

            # Prepare runtime dependent input tensors
            self.buffers['unet_sample'] = latents_expanded.to(self.latent_dtype)
            self.buffers['unet_timestep'] = timestep.to(self.latent_dtype).to("cuda")

            for tensor_name, tensor_shape in self.models['unet'].get_shape_dict(actual_batch_size).items():
                self.engines['unet'].stage_tensor(tensor_name, self.buffers[f'unet_{tensor_name}'], tensor_shape)

            if self.nvtx_profile:
                nvtx_profile_start(
                    f"Unet timestep {timestep} step {step_index}",
                    self.markers,
                    color="blue"
                )
            self.engines['unet'].infer(self.infer_stream, batch_size=actual_batch_size)
            if self.nvtx_profile:
                nvtx_profile_stop(
                    f"Unet timestep {timestep} step {step_index}",
                    self.markers
                )


            # TODO: yihengz check if we actually need sync the stream
            CUASSERT(cudart.cudaStreamSynchronize(self.infer_stream))  # make sure Unet kernel execution are finished

            # Perform guidance
            noise_pred = self.buffers['unet_latent']

            noise_pred_negative_prompt = noise_pred[0:actual_batch_size]  # negative prompt in batch dimension [0:BS]
            noise_pred_prompt = noise_pred[actual_batch_size:actual_batch_size * 2]  # prompt in batch dimension [BS:]

            noise_pred = noise_pred_negative_prompt + PipelineConfig.GUIDANCE * (noise_pred_prompt - noise_pred_negative_prompt)

            latents = self.scheduler.step(noise_pred, latents, step_index)

        latents = 1. / PipelineConfig.VAE_SCALING_FACTOR * latents
        # Transfer the Unet output to vae buffer
        self.buffers['vae_latent'] = latents


if __name__ == "__main__":
    build_dir = Path("build")
    engine_dir = build_dir/ "engines/GeForceRTX_4090x1/stable-diffusion-xl/Offline/"
    preprocessed_data_dir = build_dir / "preprocessed_data/coco2014-tokenized-sdxl/5k_dataset_final/"
    batch_size = 1
    num_samples = 1
    unet_precision = "int8"
    latent_dtype = "fp16"
    denoising_steps = 1
    use_graphs = False
    seed = 0
    verbose = True
    debug = True
    tester = Tester(
        engine_dir=engine_dir,
        preprocessed_data_dir=preprocessed_data_dir,
        batch_size=batch_size,
        num_samples=num_samples,
        unet_precision=unet_precision,
        latent_dtype=latent_dtype,
        denoising_steps=denoising_steps,
        use_graphs=use_graphs,
        seed=seed,
        verbose=verbose,
        debug=debug
    )

    unet = tester.engines["unet"]
    unet.context.report_to_profiler(), unet.context.profiler

    ppath(trt.IProfiler)

    class MyProfiler(trt.IProfiler):
        def __init__(self):
            trt.IProfiler.__init__(self)

        def report_layer_time(self, layer_name, ms):
            print(f"Layer {layer_name} took {ms} ms")

    unet.context.profiler = MyProfiler()
    unet.engine, unet.context

    for _ in range(1):
        tester.generate_images()

    start_time = time()
    tester.nvtx_profile = True
    tester.generate_images()
    end_time = time()
    print(f"Time taken: {end_time - start_time:.3f} s")
    print(f"FPS: {num_samples / (end_time - start_time):.3f}")
