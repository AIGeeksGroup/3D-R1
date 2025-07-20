import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from .helpers import get_clones

class PointTransformerV3Layer(nn.Module):
    """
    Point Transformer v3 layer with self-attention and cross-attention
    """
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross-attention (if needed)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        
        # Activation
        self.activation = F.relu if activation == "relu" else F.gelu
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None, 
                tgt=None, tgt_mask=None, tgt_key_padding_mask=None,
                memory=None, memory_mask=None, memory_key_padding_mask=None):
        
        # Self-attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Cross-attention (if memory is provided)
        if memory is not None:
            src2 = self.cross_attn(src, memory, memory, attn_mask=memory_mask,
                                 key_padding_mask=memory_key_padding_mask)[0]
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        
        # Feed-forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm3(src)
        
        return src

class PointTransformerV3Encoder(nn.Module):
    """
    Point Transformer v3 encoder
    """
    
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        
    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            
        if self.norm is not None:
            output = self.norm(output)
            
        return output

class PointTransformerV3Decoder(nn.Module):
    """
    Point Transformer v3 decoder
    """
    
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        
        intermediate = []
        
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            if self.return_intermediate:
                intermediate.append(self.norm(output) if self.norm is not None else output)
        
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        
        if self.return_intermediate:
            return torch.stack(intermediate)
        
        return output.unsqueeze(0)

class PointTransformerV3PreEncoder(nn.Module):
    """
    Point Transformer v3 pre-encoder for point cloud tokenization
    """
    
    def __init__(self, in_channels, hidden_dim, num_points=2048):
        super().__init__()
        self.num_points = num_points
        
        # Point feature extraction
        self.point_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Position embedding
        self.pos_embed = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )
        
    def forward(self, xyz, features=None):
        """
        Args:
            xyz: (B, N, 3) point coordinates
            features: (B, C, N) point features (optional)
        """
        batch_size, num_points, _ = xyz.shape
        
        # Extract point features
        if features is not None:
            features = features.transpose(1, 2)  # (B, N, C)
            point_features = self.point_mlp(features)
        else:
            point_features = torch.zeros(batch_size, num_points, self.point_mlp[-1].out_features, 
                                       device=xyz.device)
        
        # Position embedding
        pos_features = self.pos_embed(xyz)
        
        # Combine features
        combined_features = point_features + pos_features
        
        # Downsample if needed
        if num_points > self.num_points:
            # Simple random sampling for now
            indices = torch.randperm(num_points)[:self.num_points]
            xyz = xyz[:, indices, :]
            combined_features = combined_features[:, indices, :]
        
        return xyz, combined_features, None

def build_point_transformer_v3_encoder(cfg):
    """Build Point Transformer v3 encoder"""
    encoder_layer = PointTransformerV3Layer(
        d_model=cfg.enc_dim,
        nhead=cfg.enc_nhead,
        dim_feedforward=cfg.enc_ffn_dim,
        dropout=cfg.enc_dropout,
        activation=cfg.enc_activation
    )
    
    encoder = PointTransformerV3Encoder(
        encoder_layer=encoder_layer,
        num_layers=cfg.enc_nlayers,
        norm=nn.LayerNorm(cfg.enc_dim)
    )
    
    return encoder

def build_point_transformer_v3_decoder(cfg):
    """Build Point Transformer v3 decoder"""
    decoder_layer = PointTransformerV3Layer(
        d_model=cfg.dec_dim,
        nhead=cfg.dec_nhead,
        dim_feedforward=cfg.dec_ffn_dim,
        dropout=cfg.dec_dropout,
        activation="relu"
    )
    
    decoder = PointTransformerV3Decoder(
        decoder_layer=decoder_layer,
        num_layers=cfg.dec_nlayers,
        norm=nn.LayerNorm(cfg.dec_dim),
        return_intermediate=True
    )
    
    return decoder

def build_point_transformer_v3_preencoder(cfg):
    """Build Point Transformer v3 pre-encoder"""
    preencoder = PointTransformerV3PreEncoder(
        in_channels=cfg.in_channel,
        hidden_dim=cfg.enc_dim,
        num_points=cfg.preenc_npoints
    )
    
    return preencoder
