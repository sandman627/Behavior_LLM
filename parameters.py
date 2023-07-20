import torch




# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GPU_NUM = 0
device = torch.device(f"cuda:{GPU_NUM}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device=device)
print("Current CUDA device : ", torch.cuda.current_device)