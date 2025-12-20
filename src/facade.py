import logging
import sys
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
from config import ModelConfig
from feature_extractor import AudioFeatureExtractor
from dataset_loader import DatasetLoader
from model_trainer import ModelTrainer
from model_persistence import ModelPersistence
from feedback_manager import FeedbackManager


sys.path.insert(0, str(Path(__file__).parent))
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GenderDetectionFacade:
    """
    Facade Pattern: Simplified interface to the gender detection system
    
    This is the main entry point for all operations:
    - Training models
    - Making predictions
    - Collecting feedback
    - Retraining with feedback
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
       
        self.config = config or ModelConfig()
        
        # Initialize all subsystems
        self.feature_extractor = AudioFeatureExtractor(self.config)
        self.dataset_loader = DatasetLoader(self.feature_extractor)
        self.model_trainer = ModelTrainer(self.config)
        self.model_persistence = ModelPersistence(self.config)
        self.feedback_manager = FeedbackManager(self.config)
        

        self._model = None
        self._scaler = None
        
        logger.info("GenderDetectionFacade initialized")
    

    
    def train_initial_model(self, data_dir: str, 
                           classes: Optional[List[str]] = None) -> Dict:

        if classes is None:
            classes = ["female", "male"]
        
        logger.info("=" * 60)
        logger.info("Starting Initial Model Training")
        logger.info("=" * 60)
        
        # Load data
        logger.info(f"Loading data from {data_dir}")
        X, y = self.dataset_loader.load_from_directory(Path(data_dir), classes)
        
        if len(X) == 0:
            raise ValueError("No data loaded. Check your data directory.")
        
        logger.info(f"Loaded {len(X)} samples")
        
        # Train model
        model, scaler, metrics = self.model_trainer.train_model(X, y)
        
        # Save model
        self.model_persistence.save_model(model, scaler, metrics)
        
        # Update internal references
        self._model = model
        self._scaler = scaler
        
        logger.info("=" * 60)
        logger.info("Initial Training Complete")
        logger.info("=" * 60)
        
        return metrics
    
    def retrain_with_feedback(self, original_data_dir: Optional[str] = None) -> Dict:

        logger.info("=" * 60)
        logger.info("Starting Model Retraining with Feedback")
        logger.info("=" * 60)
        
        all_X = []
        all_y = []
        
        # Load original data if provided
        if original_data_dir:
            logger.info(f"Loading original data from {original_data_dir}")
            X_orig, y_orig = self.dataset_loader.load_from_directory(
                Path(original_data_dir),
                ["female", "male"]
            )
            if len(X_orig) > 0:
                all_X.append(X_orig)
                all_y.append(y_orig)
                logger.info(f"Loaded {len(X_orig)} original samples")
        
        # Load feedback data
        logger.info("Loading feedback data")
        X_feedback, y_feedback = self.dataset_loader.load_from_directory(
            self.feedback_manager.feedback_dir,
            ["female", "male"]
        )
        
        if len(X_feedback) == 0:
            logger.warning("No feedback data available for retraining")
            return None
        
        all_X.append(X_feedback)
        all_y.append(y_feedback)
        logger.info(f"Loaded {len(X_feedback)} feedback samples")
        
        # Combine all data
        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        
        logger.info(f"Total samples for retraining: {len(X)}")
        
        # Train and save
        model, scaler, metrics = self.model_trainer.train_model(X, y)
        self.model_persistence.save_model(model, scaler, metrics)
        
        # Update internal references
        self._model = model
        self._scaler = scaler
        
        logger.info("=" * 60)
        logger.info("Retraining Complete")
        logger.info("=" * 60)
        
        return metrics
    

    def predict(self, audio_path: str) -> Dict:

        if self._model is None or self._scaler is None:
            logger.info("Loading model for first prediction")
            self._model, self._scaler = self.model_persistence.load_model()
        

        features = self.feature_extractor.extract_features(audio_path)
        features_scaled = self._scaler.transform(features.reshape(1, -1))
        
        # Predict
        prediction = self._model.predict(features_scaled)[0]
        probabilities = self._model.predict_proba(features_scaled)[0]
        
        result = {
            "prediction": self.config.label_map[prediction],
            "label_id": int(prediction),
            "confidence": float(probabilities[prediction]),
            "probabilities": {
                self.config.label_map[i]: float(prob)
                for i, prob in enumerate(probabilities)
            },
            "audio_path": audio_path
        }
        
        logger.info(f"Prediction: {result['prediction']} "
                   f"(confidence: {result['confidence']:.2%})")
        
        return result
    
    def predict_batch(self, audio_paths: List[str]) -> List[Dict]:

        results = []
        for path in audio_paths:
            try:
                result = self.predict(path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting {path}: {e}")
                results.append({"error": str(e), "audio_path": path})
        
        return results
    

    
    def submit_feedback(self, audio_path: str, predicted_label: int,
                       correct_label: int, user_id: Optional[str] = None,
                       confidence: Optional[float] = None):

        self.feedback_manager.save_feedback(
            audio_path, predicted_label, correct_label, user_id, confidence
        )
        
        # Check if we should trigger retraining
        stats = self.feedback_manager.get_feedback_stats()
        
        if stats["total"] >= self.config.feedback_threshold:
            if stats["total"] % self.config.feedback_threshold == 0:
                logger.info(f"Feedback threshold reached ({stats['total']} samples). "
                          "Consider retraining the model.")
    
    def get_feedback_statistics(self) -> Dict:

        return self.feedback_manager.get_feedback_stats()
    

    
    def is_model_trained(self) -> bool:
        """Check if a trained model exists"""
        return self.model_persistence.model_exists()
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        if not self.is_model_trained():
            return {"trained": False}
        
        config = self.model_persistence.load_config()
        stats = self.get_feedback_statistics()
        
        return {
            "trained": True,
            "config": config,
            "feedback_stats": stats
        }


