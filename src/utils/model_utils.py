import os
import torch
import numpy as np

def load_params(params, path):
    for filename in os.listdir(path):
        if filename.endswith('.pth'):
            params.append(torch.load(os.path.join(path, filename)))

    return params

def square_subsequent_mask(size):
    mask = torch.triu(torch.ones(size,size)*float('-inf'),diagonal=1)
    return mask

def pad_to_shape(array, shape):
    #print(shape)
    array = np.array(array)
    padded_array = np.zeros(shape, dtype=array.dtype)
    slices = tuple(slice(0, min(dim, shape)) for i, dim in enumerate(array.shape))
    padded_array[slices] = array[slices]
    return padded_array


def pad_critic(tensor, target_shape, fill_value=0):

    tensor = torch.as_tensor(tensor)
    padding = [(target - size) for size, target in zip(tensor.shape[::-1], target_shape[::-1])]
    padding = sum([(0, pad) for pad in padding], ())
    return torch.nn.functional.pad(tensor, padding, value=fill_value)


def pad_list_to_length(lst, target_length, tensor_shape, fill_value=0):

    assert len(lst) <= target_length, "List length exceeds target length"

 
    pad_tensor = torch.full(tensor_shape, fill_value, dtype=lst[0].dtype, device=lst[0].device)

 
    return lst + [pad_tensor] * (target_length - len(lst))