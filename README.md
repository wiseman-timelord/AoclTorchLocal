# AoclTorchLocal
Status: Alpha. Do not use, under heavy development.

### Description
Amd Gpu is Rocm only, and Directml is Onnx, so the solution is: enhance Avx2 with special Aocl code. Users obviously require Aocl to be installed, its driver updates or someth for Amd processors, but its significantly faster. Obviously the user could use gguf for most things, but the objective is to more, simply and effectively, host ckpt for, image and 3d, generation on Avx2.
