# AoclTorchLocal
Status: Alpha. Do not use, under heavy development.

### Description
Amd Gpu is Rocm only, and Directml is Onnx, so the solution is: enhance Avx2 with special Aocl code. Users obviously require Aocl to be installed, its driver updates or someth for Amd processors, but its significantly faster. Obviously the user could use gguf for text on vulkan now, but the objective is to more, simply and effectively, host ckpt for, image and 3d, generation on Avx2.

## Development:
- Build torch with vulkan backend [here](https://pytorch.org/tutorials/prototype/vulkan_workflow.html). This needs investigation after AOCL-AVX2 level is working correctly, as it may be a waste of time currently.
