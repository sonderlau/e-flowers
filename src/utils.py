import torch

def get_available_device(assigned='cuda'):
    """返回可用的设备

    Args:
        assigned (str, optional): 设备名称. Defaults to 'cuda'.

    Returns:
        torch.device: 设备
    """
    if assigned:
        return torch.device(assigned)
    
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')