import pytest
from facade import GenderDetectionFacade


class TestGenderDetectionFacade:
    """Test GenderDetectionFacade class"""
    
    def test_initialization(self, test_config):
        """Test facade initialization"""
        facade = GenderDetectionFacade(test_config)
        
        assert facade.feature_extractor is not None
        assert facade.dataset_loader is not None
        assert facade.model_trainer is not None
        assert facade.model_persistence is not None
        assert facade.feedback_manager is not None
    
    def test_is_model_trained_false(self, test_config):
        """Test model trained check when no model exists"""
        facade = GenderDetectionFacade(test_config)
        
        assert not facade.is_model_trained()
    
    def test_train_initial_model(self, test_config, sample_dataset):
        """Test initial model training"""
        facade = GenderDetectionFacade(test_config)
        
        metrics = facade.train_initial_model(
            data_dir=str(sample_dataset),
            classes=["female", "male"]
        )
        
        assert metrics is not None
        assert 'accuracy' in metrics
        assert facade.is_model_trained()
    
    def test_predict(self, test_config, sample_dataset, sample_audio_file):
        """Test prediction"""
        facade = GenderDetectionFacade(test_config)
        
        # Train first
        facade.train_initial_model(str(sample_dataset))
        
        # Predict
        result = facade.predict(str(sample_audio_file))
        
        assert 'prediction' in result
        assert 'label_id' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert result['prediction'] in ['Female', 'Male']
    
    def test_predict_batch(self, test_config, sample_dataset):
        """Test batch prediction"""
        facade = GenderDetectionFacade(test_config)
        
        # Train first
        facade.train_initial_model(str(sample_dataset))
        
        # Get test files
        test_files = list((sample_dataset / "female").glob("*.wav"))[:3]
        paths = [str(f) for f in test_files]
        
        results = facade.predict_batch(paths)
        
        assert len(results) == 3
        assert all('prediction' in r for r in results)
    
    def test_submit_feedback(self, test_config, sample_dataset, sample_audio_file):
        """Test submitting feedback"""
        facade = GenderDetectionFacade(test_config)
        
        # Train and predict
        facade.train_initial_model(str(sample_dataset))
        result = facade.predict(str(sample_audio_file))
        
        # Submit feedback
        facade.submit_feedback(
            audio_path=str(sample_audio_file),
            predicted_label=result['label_id'],
            correct_label=0,
            user_id="test_user"
        )
        
        stats = facade.get_feedback_statistics()
        assert stats['total'] > 0
    
    def test_get_model_info(self, test_config, sample_dataset):
        """Test getting model info"""
        facade = GenderDetectionFacade(test_config)
        
        # Before training
        info = facade.get_model_info()
        assert info['trained'] is False
        
        # After training
        facade.train_initial_model(str(sample_dataset))
        info = facade.get_model_info()
        
        assert info['trained'] is True
        assert 'config' in info
        assert 'feedback_stats' in info
    
    def test_retrain_with_feedback(self, test_config, sample_dataset, sample_audio_file):
        """Test retraining with feedback"""
        facade = GenderDetectionFacade(test_config)
        
        # Initial training
        facade.train_initial_model(str(sample_dataset))
        
        # Add feedback
        for _ in range(3):
            facade.submit_feedback(
                audio_path=str(sample_audio_file),
                predicted_label=1,
                correct_label=0
            )
        
        # Retrain
        metrics = facade.retrain_with_feedback(str(sample_dataset))
        
        assert metrics is not None
        assert 'accuracy' in metrics

