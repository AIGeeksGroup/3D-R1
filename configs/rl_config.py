"""
RL training configuration for 3D-R1
"""

class RLConfig:
    """Configuration class for RL training"""
    
    def __init__(self):
        # GRPO hyperparameters
        self.beta = 0.1  # KL penalty coefficient
        self.gamma = 0.99  # discount factor
        self.gae_lambda = 0.95  # GAE lambda
        self.clip_ratio = 0.2  # PPO clip ratio
        self.target_kl = 0.01  # target KL divergence
        self.max_grad_norm = 1.0  # gradient clipping
        self.lr = 1e-5  # learning rate
        self.batch_size = 4
        self.num_epochs = 4
        self.group_size = 8  # group size for GRPO
        
        # Reward function weights
        self.format_weight = 0.3
        self.perception_weight = 0.4
        self.semantic_weight = 0.3
        
        # Training parameters
        self.max_epochs = 10
        self.save_every = 500
        self.eval_every = 1000
        self.log_every = 10
        
        # Model parameters
        self.freeze_detector = True
        self.freeze_llm = True
        self.max_des_len = 512
        self.max_prompt = 1
        
        # Dataset parameters
        self.dataset = 'scenecold_dataset'
        self.use_color = True
        self.use_normal = True
        self.batchsize_per_gpu = 2
        self.num_workers = 16
        
        # CLIP model for semantic similarity
        self.clip_model_name = "ViT-B/32"
        
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'beta': self.beta,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_ratio': self.clip_ratio,
            'target_kl': self.target_kl,
            'max_grad_norm': self.max_grad_norm,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'group_size': self.group_size,
            'format_weight': self.format_weight,
            'perception_weight': self.perception_weight,
            'semantic_weight': self.semantic_weight,
            'max_epochs': self.max_epochs,
            'save_every': self.save_every,
            'eval_every': self.eval_every,
            'log_every': self.log_every,
            'freeze_detector': self.freeze_detector,
            'freeze_llm': self.freeze_llm,
            'max_des_len': self.max_des_len,
            'max_prompt': self.max_prompt,
            'dataset': self.dataset,
            'use_color': self.use_color,
            'use_normal': self.use_normal,
            'batchsize_per_gpu': self.batchsize_per_gpu,
            'num_workers': self.num_workers,
            'clip_model_name': self.clip_model_name,
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

# Default configurations
DEFAULT_RL_CONFIG = RLConfig()

# High reward weight configuration
HIGH_REWARD_CONFIG = RLConfig()
HIGH_REWARD_CONFIG.format_weight = 0.4
HIGH_REWARD_CONFIG.perception_weight = 0.4
HIGH_REWARD_CONFIG.semantic_weight = 0.2

# Conservative training configuration
CONSERVATIVE_CONFIG = RLConfig()
CONSERVATIVE_CONFIG.beta = 0.2  # Higher KL penalty
CONSERVATIVE_CONFIG.lr = 5e-6  # Lower learning rate
CONSERVATIVE_CONFIG.max_grad_norm = 0.5  # Lower gradient norm

# Aggressive training configuration
AGGRESSIVE_CONFIG = RLConfig()
AGGRESSIVE_CONFIG.beta = 0.05  # Lower KL penalty
AGGRESSIVE_CONFIG.lr = 2e-5  # Higher learning rate
AGGRESSIVE_CONFIG.max_grad_norm = 2.0  # Higher gradient norm
