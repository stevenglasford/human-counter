import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA version:", torch.version.cuda)

# This forces a CUDA kernel call to ensure GPU is usable
x = torch.rand(1024, 1024).cuda()
y = torch.mm(x, x)
print("Matrix multiply successful on GPU:", y.is_cuda)
