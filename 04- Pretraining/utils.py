import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken

def text_to_token_ids(text,tokenizer):

    encoded=tokenizer.encode(text,allowed_special={'<|endoftext|>'})

    encoded_tensor=torch.tensor(encoded).unsqueeze(0) #we convert the list above to pytorch tensor cause the model expects tensor(adds batch dimension)

    return encoded_tensor



def token_ids_to_text(token_ids,tokenizer):

    flat=token_ids.squeeze(0) 

    return tokenizer.decode(flat.tolist()) #needs a normal Python list.