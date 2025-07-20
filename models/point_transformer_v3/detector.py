import torch
import torch.nn as nn
from .config import model_config
from .criterion import build_criterion
from .helpers import GenericMLP
from .position_embedding import PositionEmbeddingCoordsSine
from .transformer import (
    build_point_transformer_v3_encoder,
    build_point_transformer_v3_decoder,
    build_point_transformer_v3_preencoder
)

class BoxProcessor:
    def __init__(self, dataset_config):
        self.dataset_config = dataset_config

    def compute_predicted_center(self, center_offset, query_xyz, point_cloud_dims):
        center_unnormalized = query_xyz + center_offset
        return center_unnormalized, center_unnormalized

    def compute_predicted_size(self, size_normalized, point_cloud_dims):
        return size_normalized

    def compute_predicted_angle(self, angle_logits, angle_residual):
        return angle_logits.squeeze(-1)

    def compute_objectness_and_cls_prob(self, cls_logits):
        cls_prob = torch.nn.functional.softmax(cls_logits, dim=-1)
        objectness_prob = 1 - cls_prob[..., -1]
        return cls_prob[..., :-1], objectness_prob

class Model_PointTransformerV3_DETR(nn.Module):
    def __init__(self, tokenizer, encoder, decoder, dataset_config, 
                 encoder_dim=256, decoder_dim=256, num_queries=256, criterion=None):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.decoder = decoder
        
        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=[encoder_dim],
            output_dim=decoder_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
        )
        
        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=decoder_dim, pos_type="fourier", normalize=True
        )
        
        self.query_projection = GenericMLP(
            input_dim=decoder_dim,
            hidden_dims=[decoder_dim],
            output_dim=decoder_dim,
            use_conv=True,
            output_use_activation=True,
        )
        
        # MLP heads
        self.sem_cls_head = nn.Linear(decoder_dim, dataset_config.num_semcls + 1)
        self.center_head = nn.Linear(decoder_dim, 3)
        self.size_head = nn.Linear(decoder_dim, 3)
        self.angle_head = nn.Linear(decoder_dim, 1)
        
        self.box_processor = BoxProcessor(dataset_config)
        self.criterion = criterion

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def run_encoder(self, point_clouds):
        xyz, features = self._break_up_pc(point_clouds)
        
        # Point Transformer v3 encoding
        enc_xyz, enc_features, enc_inds = self.tokenizer(xyz, features)
        
        # Encoder expects batch x npoints x channel
        enc_features = self.encoder(enc_features)
        
        return enc_xyz, enc_features, enc_inds

    def forward(self, inputs, is_eval=False):
        point_clouds = inputs["point_clouds"]
        point_cloud_dims = [
            inputs["point_cloud_dims_min"],
            inputs["point_cloud_dims_max"],
        ]
        
        # Feature encoding
        enc_xyz, enc_features, enc_inds = self.run_encoder(point_clouds)
        
        # Project to decoder dimension
        enc_features = self.encoder_to_decoder_projection(enc_features.transpose(1, 2)).transpose(1, 2)
        
        # Generate queries (simplified)
        batch_size, num_points, feat_dim = enc_features.shape
        query_xyz = enc_xyz[:, :256, :]  # Use first 256 points as queries
        query_features = enc_features[:, :256, :]
        
        # Position embedding
        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        
        # Decoder
        box_features = self.decoder(
            query_features, enc_features, 
            query_pos=query_embed, pos=self.pos_embedding(enc_xyz, input_range=point_cloud_dims)
        )
        
        # MLP heads
        sem_cls_logits = self.sem_cls_head(box_features)
        center_offset = self.center_head(box_features).sigmoid() - 0.5
        size_normalized = self.size_head(box_features).sigmoid()
        angle_logits = self.angle_head(box_features)
        
        # Box predictions
        center_normalized, center_unnormalized = self.box_processor.compute_predicted_center(
            center_offset, query_xyz, point_cloud_dims
        )
        size_unnormalized = self.box_processor.compute_predicted_size(
            size_normalized, point_cloud_dims
        )
        angle_continuous = self.box_processor.compute_predicted_angle(
            angle_logits, angle_logits
        )
        
        # Objectness and class probabilities
        semcls_prob, objectness_prob = self.box_processor.compute_objectness_and_cls_prob(sem_cls_logits)
        
        outputs = {
            'sem_cls_logits': sem_cls_logits,
            'center_normalized': center_normalized,
            'center_unnormalized': center_unnormalized,
            'size_normalized': size_normalized,
            'size_unnormalized': size_unnormalized,
            'angle_logits': angle_logits,
            'angle_continuous': angle_continuous,
            'objectness_prob': objectness_prob,
            'sem_cls_prob': semcls_prob,
            'prop_features': box_features.unsqueeze(0),  # Add layer dimension
            'enc_features': enc_features,
            'enc_xyz': enc_xyz,
            'query_xyz': query_xyz,
        }
        
        return outputs

def detector_PointTransformerV3(args, dataset_config):
    """Build Point Transformer v3 detector"""
    cfg = model_config(args, dataset_config)
    
    tokenizer = build_point_transformer_v3_preencoder(cfg)
    encoder = build_point_transformer_v3_encoder(cfg)
    decoder = build_point_transformer_v3_decoder(cfg)
    
    criterion = build_criterion(cfg, dataset_config)
    
    model = Model_PointTransformerV3_DETR(
        tokenizer,
        encoder,
        decoder,
        cfg.dataset_config,
        encoder_dim=cfg.enc_dim,
        decoder_dim=cfg.dec_dim,
        num_queries=cfg.nqueries,
        criterion=criterion
    )
    return model
