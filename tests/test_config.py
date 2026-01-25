import pytest
from config import ModelConfig


class TestModelConfig:
    """Test ModelConfig class"""
    
    def test_default_initialization(self):
        """Test default config initialization"""
        config = ModelConfig()
        
        assert config.sample_rate == 16000
        assert config.duration == 2.0
        assert config.n_mfcc == 40
        assert config.n_estimators == 200
        assert config.random_state == 42
        assert config.label_map == {0: "Female", 1: "Male"}
    
    def test_custom_initialization(self):
        """Test custom config values"""
        config = ModelConfig(
            sample_rate=22050,
            n_mfcc=20,
            n_estimators=100
        )
        
        assert config.sample_rate == 22050
        assert config.n_mfcc == 20
        assert config.n_estimators == 100
    
    def test_directory_creation(self, temp_dir):
        """Test that directories are created on init"""
        config = ModelConfig()
        config.artifacts_dir = str(temp_dir / "artifacts")
        config.feedback_dir = str(temp_dir / "feedback")
        config.__post_init__()
        
        assert (temp_dir / "artifacts").exists()
        assert (temp_dir / "feedback").exists()