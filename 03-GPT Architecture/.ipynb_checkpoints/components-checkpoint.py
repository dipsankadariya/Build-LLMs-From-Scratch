import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken


class GPTDatasetV1(Dataset):
    """
    Prepare training samples for the GPT model.

    Workflow:
        1. Tokenize the entire input text into a single sequence of token IDs.
        2. Use a sliding window to split the sequence into overlapping input
           chunks of length `max_length`.
        3. Create the target chunk by shifting the input chunk one token to
           the right so the model learns to predict the next token.
        4. Convert each input-target pair into PyTorch tensors and store them.
        5. Return one `(input_ids, target_ids)` pair whenever the DataLoader
           requests a sample.
    """

    def __init__(self, txt, tokenizer, max_length, stride):
        super().__init__()

        self.input_ids = []
        self.target_ids = []

        # tokenize the input text into token ids
        token_ids = tokenizer.encode(
            txt,
            allowed_special={"<|endoftext|>"}
        )

        # create overlapping input-target pairs using a sliding window
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# the GPTDatasetV1 (stores all input-target pairs) so wee need to crete a helper class that groups samples into mini-batches

def create_dataloader(txt,batch_size=4,max_length=256,stride=128,shuffle=True,drop_last=True,num_workers=0):
    #initialize the tokenizer
    tokenizer=tiktoken.get_encoding("gpt2")
    #create the dataset using the claass(instantiating)
    dataset=GPTDatasetV1(txt,tokenizer,max_length,stride)

    #Create the dataloader
    dataloader=DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers
    )

    return dataloader


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self,input_dim,output_dim,context_length,dropout,num_heads,qkv_bias=False):
        super().__init__()
        assert output_dim % num_heads== 0 , "output_dim must be divisible by num_heads"
        self.output_dim=output_dim
        self.num_heads=num_heads
        self.head_dim = output_dim // num_heads
        self.W_query=nn.Linear(input_dim,output_dim,bias=qkv_bias)
        self.W_key=nn.Linear(input_dim,output_dim,bias=qkv_bias)
        self.W_value=nn.Linear(input_dim,output_dim,bias=qkv_bias)
        self.dropout=nn.Dropout(dropout)
        self.register_buffer("mask",torch.triu(torch.ones(context_length,context_length),diagonal=1))
        self.output_proj= nn.Linear(output_dim,output_dim)


    def forward(self, x):
        b, num_tokens, input_dim = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.reshape(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.reshape(b, num_tokens, self.num_heads, self.head_dim)
        values = values.reshape(b, num_tokens, self.num_heads, self.head_dim)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        mask = self.mask[:num_tokens, :num_tokens]
        mask = mask.bool()

        attn_scores = queries @ keys.transpose(2, 3)

        # apply mask
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = torch.softmax(
            attn_scores / (self.head_dim ** 0.5),
            dim=-1
        )
        # apply dropout
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        context_vec = context_vec.transpose(1, 2)
        context_vec = context_vec.contiguous().view(
            b,
            num_tokens,
            self.output_dim
        )

        context_vec = self.output_proj(context_vec)
        return context_vec