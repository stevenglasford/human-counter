import torch
from torch.utils.cpp_extension import CppExtension

print("Using GPU:", torch.cuda.get_device_name(0))
print("CUDA Version:", torch.version.cuda)
print("Compiled architectures:", CppExtension.get_cuda_arch_list())
