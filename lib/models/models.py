import torch 
import torch.nn as nn 


class EmbeddingLayer(nn.Module): 
    """
    A torch module representing the embedding layer of the transformer. 
    The input time serie, for each timestep, is projected into space of dimension d, 
    and the positional embedding is then encodded using a summation 

    Attributes
    -----
    w: int 
        Number of timesteps
    
    m : int 
        number of exogenous variables

    d_dim : int 
        Dimension size  
        
    References 
    -----
        George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning
    """
    def __init__(self, w, m, d_dim): 
        super().__init__()
        self.w = w 
        self.m = m
        self.d_dim = d_dim 

        self.fc = nn.Linear(m, d_dim) 
        self.pos_embedding = nn.Embedding(w, d_dim)

    def forward(self, x): 
        """
        Compute the input projection of the sequence.

        Parameters
        -----
        x: torch.tensor (bs, w, m)
            Batch input sequence with w timesteps and m exogenous features.

        Output 
        -----
        u: torch.tensor (bs, w, d_dim).
            projections with positional embedding output.
        """

        x1 = self.fc(x) # (bs, w, d_dim)
        idx = torch.range(0, self.w-1, dtype=torch.int, device=x.device)
        out = x1 + self.pos_embedding(idx) # (bs, w, d_dim) 
        return out


class TransformerEncoder(nn.Module):
    """
    Transformer encoder layer According to attention is all you need. 

    References 
    -----
        Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). 
        Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).
    """
    def __init__(self, d_dim, num_heads, num_layers, ff_hidden_dim, dropout=0.1):
        super().__init__()
        
        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_dim,
            nhead=num_heads,
            dim_feedforward=ff_hidden_dim,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, src, src_mask=None):
        # Encoder
        output = self.encoder(src, src_key_padding_mask=src_mask)
        return output


class Head(nn.Module): 
    """
    Adaptive regressino or classification head. 
    """
    def __init__(self, w, input_dim, out_dim): 
        super().__init__()
        self.fc = nn.Linear(input_dim * w, out_dim)

    def forward(self, x): 
        """
        x: torch.tensor 
            (bs, w, d) 

        out: torch.tensor 
            (bs, w, out_dim)
        """
        out = self.fc(x.view(x.shape[0], -1))
        return out
    
class ReconstructionHead(nn.Module): 
    def __init__(self, input_dim, out_dim): 
        super().__init__()
        self.fc = nn.Linear(input_dim, out_dim)

    def forward(self, x): 
        """
        x: torch.tensor 
            (bs, w, input_dim)
        """

        out = self.fc(x) 
        return out # (bs, w, out_dim)

class TFMTSRL(nn.Module): 
    """
    Complete embedding + encoder + head model

    Attributes
    -----
    w: int 
        Number of timesteps
    
    m : int 
        number of exogenous variables

    d_dim : int 
        Dimension size  

    mode: str
        "supervised" or "unsupervised"

        
    References 
    -----
        George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning
    """


    def __init__(
        self, 
        w, 
        m, 
        d_dim, 
        out_dim,
        num_heads, 
        num_layers, 
        ff_hidden_dim, 
        dropout, 
        mode
    ): 
        super().__init__()
        self.input_proj = EmbeddingLayer(w, m, d_dim)
        self.encoder = TransformerEncoder(
            d_dim, 
            num_heads, 
            num_layers, 
            ff_hidden_dim, 
            dropout
        )
        if mode == "supervised":
            self.head = Head(w, d_dim, out_dim)
        elif mode == "unsupervised": 
            self.head = ReconstructionHead(d_dim, out_dim)
        else: 
            raise ValueError(f"Mode should be either 'supervised' or 'unsupervised'. Received {mode}.")
        
    def forward(self, x): 
        """
        Forward pass of the entire model 
        """
        embedding = self.input_proj(x)
        x1 = self.encoder(embedding) 
        out = self.head(x1)
        return out
