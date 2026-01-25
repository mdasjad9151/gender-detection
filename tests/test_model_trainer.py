import pytest
import numpy as np
from model_trainer import ModelTrainer


class TestModelTrainer:
    """Test ModelTrainer class"""
    
    def test_initialization(self, test_config):
        """Test model trainer initialization"""
        trainer = ModelTrainer(test_config)
        
        assert trainer.n_estimators == test_config.n_estimators
        assert trainer.random_state == test_config.random_state
    
    def test_train_model(self, test_config, sample_training_data):
        """Test model training"""
        trainer = ModelTrainer(test_config)
        X, y = sample_training_data
        
        model, scaler, metrics = trainer.train_model(X, y)
        
        assert model is not None
        assert scaler is not None
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'confusion_matrix' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_train_model_insufficient_data(self, test_config):
        """Test training with insufficient data"""
        trainer = ModelTrainer(test_config)
        X = np.random.randn(5, 80)
        y = np.array([0, 0, 0, 1, 1])
        
        # Should still work but may have low accuracy
        model, scaler, metrics = trainer.train_model(X, y)
        assert model is not None
    
    def test_cross_validate(self, test_config, sample_training_data):
        """Test cross-validation"""
        trainer = ModelTrainer(test_config)
        X, y = sample_training_data
        
        cv_results = trainer.cross_validate(X, y, cv=3)
        
        assert 'cv_scores' in cv_results
        assert 'mean_accuracy' in cv_results
        assert 'std_accuracy' in cv_results
        assert len(cv_results['cv_scores']) == 3
