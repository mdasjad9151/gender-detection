"""
Configuration management for the gender detection system
"""
from dataclasses import dataclass, field
from typing import Dict
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for the model and system"""
    
    # Audio processing parameters
    sample_rate: int = 16000
    duration: float = 2.0
    n_mfcc: int = 40
    
    # Model parameters
    n_estimators: int = 200
    random_state: int = 42
    test_size: float = 0.2
    
    # Paths
    artifacts_dir: str = "artifacts"
    model_path: str = "artifacts/gender_rf.pkl"
    scaler_path: str = "artifacts/scaler.pkl"
    config_path: str = "artifacts/config.json"
    feedback_dir: str = "feedback_data"
    log_dir: str = "logs"
    
    # Labels
    label_map: Dict[int, str] = field(default_factory=lambda: {
        0: "Female",
        1: "Male"
    })
    
    # Retraining parameters
    feedback_threshold: int = 100  
    
    def __post_init__(self):
        """Create necessary directories"""
        Path(self.artifacts_dir).mkdir(exist_ok=True)
        Path(self.feedback_dir).mkdir(exist_ok=True)
        Path(self.log_dir).mkdir(exist_ok=True)