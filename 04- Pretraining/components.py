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

def create_dataloader_v1(txt,batch_size=4,max_length=256,stride=128,shuffle=True,drop_last=True,num_workers=0):
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
        self.out_proj = nn.Linear(output_dim, output_dim)


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

        context_vec = self.out_proj(context_vec)
        return context_vec


class LayerNorm(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.eps=1e-5
        self.scale=nn.Parameter(torch.ones(emb_dim))
        self.shift=nn.Parameter(torch.zeros(emb_dim))

    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        var=x.var(dim=-1,keepdim=True, unbiased=False)
        norm_x= (x-mean)/torch.sqrt(var+ self.eps)
        return self.scale* norm_x + self.shift


import math

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(
                math.sqrt(2.0 / math.pi) *
                (x + 0.044715 * torch.pow(x, 3))
            )
        )
        

class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]),
            GELU(),
            nn.Linear(4*cfg["emb_dim"],cfg["emb_dim"])
        )

    def forward(self,x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.norm1= LayerNorm(cfg["emb_dim"])

        self.att=MaskedMultiHeadAttention(
            input_dim=cfg["emb_dim"],
            output_dim=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )

        self.norm2=LayerNorm(cfg["emb_dim"])
        self.ff=FeedForward(cfg)
        self.dropout_shortcut= nn.Dropout(cfg["drop_rate"])

    def  forward(self,x):
        shortcut=x
        x=self.norm1(x)
        x=self.att(x)
        x=self.dropout_shortcut(x)
        x=x+shortcut

        shortcut=x
        x=self.norm2(x)
        x=self.ff(x)
        x=self.dropout_shortcut(x)
        x=x+shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb=nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        self.pos_emb=nn.Embedding(cfg["context_length"],cfg["emb_dim"])
        self.drop_emb=nn.Dropout(cfg["drop_rate"])
        self.trf_blocks=nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm=LayerNorm(cfg["emb_dim"])
        self.out_head=nn.Linear(cfg["emb_dim"],cfg["vocab_size"],bias=False)

    def forward(self,in_idx):
        batch_size,seq_len=in_idx.shape
        tok_embeds=self.tok_emb(in_idx)
        pos_embeds=self.pos_emb(torch.arange(seq_len,device=in_idx.device))

        x=tok_embeds+pos_embeds
        x=self.drop_emb(x)
        x=self.trf_blocks(x)
        x=self.final_norm(x)
        logits=self.out_head(x)

        return logits

def generate_text_simple(model, input_tokens, max_new_tokens, context_size):
    generated_tokens = input_tokens

    for _ in range(max_new_tokens):
        model_input = generated_tokens[:, -context_size:]

        with torch.no_grad():
            output_logits = model(model_input)

        last_position_logits = output_logits[:, -1, :]

        probabilities = torch.softmax(
            last_position_logits,
            dim=-1
        )

        predicted_token = torch.argmax(
            probabilities,
            dim=-1,
            keepdim=True
        )

        generated_tokens = torch.cat(
            (generated_tokens, predicted_token),
            dim=1
        )

    return generated_tokens


def generate(model, input_tokens, max_new_tokens, context_size,
             temperature=1.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        input_tokens = input_tokens[:, -context_size:]
        with torch.no_grad():
            logits = model(input_tokens)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits
            )
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and idx_next.item() == eos_id:
            break
        input_tokens = torch.cat((input_tokens, idx_next), dim=1)
    return input_tokens