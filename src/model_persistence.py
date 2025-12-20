"""
Component 4: Model Persistence
Handles saving and loading trained models
"""
import joblib
import json
from pathlib import Path
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class ModelPersistence:
    """Handles model serialization and deserialization"""
    
    def __init__(self, config):
        self.config = config
        self.model_path = Path(config.model_path)
        self.scaler_path = Path(config.scaler_path)
        self.config_path = Path(config.config_path)
        
        # Ensure artifacts directory exists
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model, scaler, metrics: Dict = None):

        try:
            # Save model
            joblib.dump(model, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
            
            # Save scaler
            joblib.dump(scaler, self.scaler_path)
            logger.info(f"Scaler saved to {self.scaler_path}")
            
            # Save configuration
            config_dict = {
                "sample_rate": self.config.sample_rate,
                "duration": self.config.duration,
                "n_mfcc": self.config.n_mfcc,
                "n_estimators": self.config.n_estimators,
                "label_map": {str(k): v for k, v in self.config.label_map.items()},
                "metrics": metrics
            }
            
            with open(self.config_path, "w") as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self) -> Tuple:
   
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            if not self.scaler_path.exists():
                raise FileNotFoundError(f"Scaler not found at {self.scaler_path}")
            
            model = joblib.load(self.model_path)
            scaler = joblib.load(self.scaler_path)
            
            logger.info("Model and scaler loaded successfully")
            
            return model, scaler
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_config(self) -> Dict:
    
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config not found at {self.config_path}")
            
            with open(self.config_path, "r") as f:
                config = json.load(f)
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def model_exists(self) -> bool:
        """Check if trained model exists"""
        return self.model_path.exists() and self.scaler_path.exists()