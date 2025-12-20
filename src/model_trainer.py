"""
Component 3: Model Training
Handles model training, evaluation, and hyperparameter tuning
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, f1_score)
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and evaluates machine learning models"""
    
    def __init__(self, config):
        self.config = config
        self.n_estimators = config.n_estimators
        self.random_state = config.random_state
        self.test_size = config.test_size
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Tuple:

        logger.info("Starting model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state
        )
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        
        logger.info("Training Random Forest classifier...")
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        
        metrics = self._calculate_metrics(y_test, y_pred)
        
        logger.info(f"Model Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Model F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"\n{metrics['classification_report']}")
        
        return model, scaler, metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, 
                          y_pred: np.ndarray) -> Dict:

        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred)
        }
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict:

        from sklearn.model_selection import cross_val_score
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        
        logger.info(f"Cross-validation scores: {scores}")
        logger.info(f"Mean CV accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return {
            'cv_scores': scores.tolist(),
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std()
        }