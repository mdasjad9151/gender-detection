import pytest
from feedback_manager import FeedbackManager


class TestFeedbackManager:
    """Test FeedbackManager class"""
    
    def test_initialization(self, test_config):
        """Test feedback manager initialization"""
        manager = FeedbackManager(test_config)
        
        assert manager.feedback_dir.exists()
        assert (manager.feedback_dir / "female").exists()
        assert (manager.feedback_dir / "male").exists()
    
    def test_save_feedback(self, test_config, sample_audio_file):
        """Test saving feedback"""
        manager = FeedbackManager(test_config)
        
        manager.save_feedback(
            audio_path=str(sample_audio_file),
            predicted_label=1,
            correct_label=0,
            user_id="test_user",
            confidence=0.85
        )
        
        # Check files were created
        female_dir = manager.feedback_dir / "female"
        audio_files = list(female_dir.glob("*.wav"))
        json_files = list(female_dir.glob("*.json"))
        
        assert len(audio_files) > 0
        assert len(json_files) > 0
    
    def test_save_feedback_correct_prediction(self, test_config, sample_audio_file):
        """Test saving feedback for correct prediction"""
        manager = FeedbackManager(test_config)
        
        manager.save_feedback(
            audio_path=str(sample_audio_file),
            predicted_label=0,
            correct_label=0
        )
        
        female_dir = manager.feedback_dir / "female"
        audio_files = list(female_dir.glob("*correct*.wav"))
        
        assert len(audio_files) > 0
    
    def test_get_feedback_stats_empty(self, test_config):
        """Test getting stats with no feedback"""
        manager = FeedbackManager(test_config)
        
        stats = manager.get_feedback_stats()
        
        assert stats['total'] == 0
        assert stats['correct_predictions'] == 0
        assert stats['corrected_predictions'] == 0
    
    def test_get_feedback_stats_with_data(self, test_config, sample_audio_file):
        """Test getting stats with feedback data"""
        manager = FeedbackManager(test_config)
        
        # Add some feedback
        manager.save_feedback(str(sample_audio_file), 0, 0)
        manager.save_feedback(str(sample_audio_file), 1, 0)
        
        stats = manager.get_feedback_stats()
        
        assert stats['total'] >= 2
        assert 'by_class' in stats
    
    def test_clear_feedback(self, test_config, sample_audio_file):
        """Test clearing feedback data"""
        manager = FeedbackManager(test_config)
        
        # Add feedback
        manager.save_feedback(str(sample_audio_file), 0, 0)
        
        # Clear specific class
        manager.clear_feedback("female")
        
        stats = manager.get_feedback_stats()
        assert stats['by_class']['Female']['total'] == 0
