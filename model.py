import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    """
    Input embeddings for the transformer model.
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model.
    """
    
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        # dropout is to make the model less overfeat
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arrange(0, seq_len, dtype=torch.float).unsqueeze(1)
        dev_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        # Apply the sin to even positions
        pe[:,0::2] = torch.sin(position * dev_term)
        pe[:,1::2] = torch.cos(position * dev_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # position encoding
        return self.dropout(x)

        
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        # eps: to avoid division by zero, we don't want very big or very small numbers
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiply
        self.bias = nn.Parameter(torch.zeros(1)) # add
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    """
    A feed forward block in the transformer model.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.linear2 = nn.Linear(d_ff, d_model) # W2 and B2
    
    def forward(self, x):
        # (Batch, seq_len, d_model) -> (Batch, seq_len, d_ff) -> (Batch, seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttentionBlock(nn.Module):
    """
    A multi-head attention block in the transformer model.
    """
    def __init__(self, d_model: int, h: int, dropout: float):
        # d_model: model dimension
        # h: number of heads
        # dropout: dropout rate
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv
 
        self.w_o = nn.Linear(d_model, d_model) # Wo, the output matrix
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Compute the attention scores and apply them to the values.
        """
        d_k = query.shape[-1]

        # (Batch, h, seq_len, d_k)  -->  (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k, v, mask):
        """
        Compute the multi-head attention.
        """
        query = self.w_q(q) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        key = self.w_k(k) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        value = self.w_v(v) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)

        # Split the embeddings into multiple heads
        # (Batch, seq_len, d_model) -> (Batch, seq_len, h, d_k) -> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, h, seq_len, d_k) ->transpose-> (Batch, seq_len, h, d_k) ->view-> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        # Apply the output matrix (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm()
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    """
    A single encoder block in the transformer.
    """
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, lambda x: self.feed_forward_block(x))
        return x


class Encoder(nn.Module):
    """
    The entire encoder stack.
    """
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm()
    
    def forward(self, x, mask):
        # N layers
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    """
    A single decoder block in the transformer.
    """
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.Module([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        source: (batch, seq_len, d_model), encoder output
        target: (batch, seq_len, d_model), decoder input
        """
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, lambda x: self.feed_forward_block(x))
        return x


class Decoder(nn.Module):
    """
    The entire decoder stack.
    """
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # N layers
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)