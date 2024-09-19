import re

import torch
from diffusers import DiffusionPipeline

from model_inspector import ProfilingInterpreter


if __name__ == "__main__":
    dtype = torch.float16
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16",
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipe.to(device)

    latent = torch.randn(2, 4, 128, 128, device=device, dtype=dtype)
    t = torch.tensor(10., device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(2, 77, 2048, device=device, dtype=dtype)
    added_cond_kwargs = {
        "text_embeds": torch.randn(2, 1280, device=device, dtype=dtype),
        "time_ids": torch.tensor([[1024., 1024.,    0.,    0., 1024., 1024.], [1024., 1024.,    0.,    0., 1024., 1024.]], device=device, dtype=dtype),
    }

    interp = ProfilingInterpreter(pipe.unet, input_example=((latent, t, encoder_hidden_states, ), {"added_cond_kwargs": added_cond_kwargs}))
    _ = interp.run(latent, t, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)
    table = interp.table
    table.to_csv("sd.csv", index=False, sep="\t")
    flops_table = table[table.flops > 0]
    flops_table.to_csv("sd_flops.csv", index=False, sep="\t")

    node_names = flops_table.name.tolist()

    attn_pattern = re.compile(r"(up|down)_blocks.\d+.attentions.\d+.transformer_blocks.\d+.attn\d+")
    attn_nodes = [node for node in node_names if attn_pattern.fullmatch(node)]

    print(flops_table)
