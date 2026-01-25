import pytest


class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_complete_workflow(self, test_config, sample_dataset, sample_audio_file):
        """Test complete workflow from training to prediction to feedback"""
        facade = GenderDetectionFacade(test_config)
        
        # Step 1: Train model
        metrics = facade.train_initial_model(str(sample_dataset))
        assert metrics['accuracy'] > 0
        
        # Step 2: Make prediction
        result = facade.predict(str(sample_audio_file))
        assert result['prediction'] in ['Female', 'Male']
        
        # Step 3: Submit feedback
        facade.submit_feedback(
            str(sample_audio_file),
            result['label_id'],
            0,
            "test_user"
        )
        
        # Step 4: Check stats
        stats = facade.get_feedback_statistics()
        assert stats['total'] >= 1
        
        # Step 5: Retrain
        new_metrics = facade.retrain_with_feedback(str(sample_dataset))
        assert new_metrics is not None
    
    def test_multiple_predictions(self, test_config, sample_dataset):
        """Test multiple predictions in sequence"""
        facade = GenderDetectionFacade(test_config)
        
        facade.train_initial_model(str(sample_dataset))
        
        # Get all test files
        female_files = list((sample_dataset / "female").glob("*.wav"))
        male_files = list((sample_dataset / "male").glob("*.wav"))
        
        # Predict on all
        for audio_file in female_files + male_files:
            result = facade.predict(str(audio_file))
            assert result['prediction'] in ['Female', 'Male']
            assert 0 <= result['confidence'] <= 1