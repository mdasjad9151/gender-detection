import pytest
import numpy as np
from feature_extractor import AudioFeatureExtractor


class TestAudioFeatureExtractor:
    """Test AudioFeatureExtractor class"""
    
    def test_initialization(self, test_config):
        """Test feature extractor initialization"""
        extractor = AudioFeatureExtractor(test_config)
        
        assert extractor.sample_rate == test_config.sample_rate
        assert extractor.duration == test_config.duration
        assert extractor.n_mfcc == test_config.n_mfcc
    
    def test_load_audio_fixed_length_truncate(self, test_config, sample_audio_file):
        """Test audio loading with truncation"""
        extractor = AudioFeatureExtractor(test_config)
        
        audio, sr = extractor.load_audio_fixed_length(str(sample_audio_file))
        
        expected_length = int(test_config.sample_rate * test_config.duration)
        assert len(audio) == expected_length
        assert sr == test_config.sample_rate
    
    def test_load_audio_fixed_length_pad(self, test_config, temp_dir):
        """Test audio loading with padding"""
        import soundfile as sf
        
        # Create short audio file
        short_audio = np.random.randn(8000).astype(np.float32)
        audio_path = temp_dir / "short_audio.wav"
        sf.write(str(audio_path), short_audio, test_config.sample_rate)
        
        extractor = AudioFeatureExtractor(test_config)
        audio, sr = extractor.load_audio_fixed_length(str(audio_path))
        
        expected_length = int(test_config.sample_rate * test_config.duration)
        assert len(audio) == expected_length
        assert len(audio) > len(short_audio)  # Padded
    
    def test_extract_features(self, test_config, sample_audio_file):
        """Test feature extraction"""
        extractor = AudioFeatureExtractor(test_config)
        
        features = extractor.extract_features(str(sample_audio_file))
        
        # Should have n_mfcc * 2 features (mean + std)
        expected_shape = test_config.n_mfcc * 2
        assert features.shape == (expected_shape,)
        assert not np.isnan(features).any()
    
    def test_extract_features_invalid_file(self, test_config):
        """Test feature extraction with invalid file"""
        extractor = AudioFeatureExtractor(test_config)
        
        with pytest.raises(Exception):
            extractor.extract_features("nonexistent_file.wav")
    
    def test_extract_features_batch(self, test_config, sample_dataset):
        """Test batch feature extraction"""
        extractor = AudioFeatureExtractor(test_config)
        
        audio_files = list((sample_dataset / "female").glob("*.wav"))
        paths = [str(f) for f in audio_files[:3]]
        
        features = extractor.extract_features_batch(paths)
        
        assert features.shape[0] == 3
        assert features.shape[1] == test_config.n_mfcc * 2