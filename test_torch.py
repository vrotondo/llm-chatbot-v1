import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable CUDA
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ':16:False'

print("Testing torch import with CUDA disabled...")
import torch
print("torch imported successfully")
print("torch version:", torch.__version__)
