"""
Pytest fixtures and configuration
"""
import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import ModelConfig
from feature_extractor import AudioFeatureExtractor
from dataset_loader import DatasetLoader
from model_trainer import ModelTrainer
from model_persistence import ModelPersistence
from feedback_manager import FeedbackManager
from facade import GenderDetectionFacade


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def test_config(temp_dir):
    """Create a test configuration"""
    config = ModelConfig()
    config.artifacts_dir = str(temp_dir / "artifacts")
    config.model_path = str(temp_dir / "artifacts" / "model.pkl")
    config.scaler_path = str(temp_dir / "artifacts" / "scaler.pkl")
    config.config_path = str(temp_dir / "artifacts" / "config.json")
    config.feedback_dir = str(temp_dir / "feedback")
    config.log_dir = str(temp_dir / "logs")
    return config


@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing"""
    # Generate 2 seconds of random audio at 16kHz
    sample_rate = 16000
    duration = 2.0
    n_samples = int(sample_rate * duration)
    audio = np.random.randn(n_samples).astype(np.float32)
    return audio, sample_rate


@pytest.fixture
def sample_audio_file(temp_dir, sample_audio_data):
    """Create a temporary audio file"""
    import soundfile as sf
    
    audio, sr = sample_audio_data
    audio_path = temp_dir / "test_audio.wav"
    sf.write(str(audio_path), audio, sr)
    return audio_path


@pytest.fixture
def sample_dataset(temp_dir):
    """Create a sample dataset directory structure"""
    import soundfile as sf
    
    # Create directory structure
    data_dir = temp_dir / "data"
    female_dir = data_dir / "female"
    male_dir = data_dir / "male"
    
    female_dir.mkdir(parents=True)
    male_dir.mkdir(parents=True)
    
    # Generate sample audio files
    sample_rate = 16000
    duration = 2.0
    n_samples = int(sample_rate * duration)
    
    # Create 5 female samples
    for i in range(5):
        audio = np.random.randn(n_samples).astype(np.float32)
        sf.write(str(female_dir / f"female_{i}.wav"), audio, sample_rate)
    
    # Create 5 male samples
    for i in range(5):
        audio = np.random.randn(n_samples).astype(np.float32)
        sf.write(str(male_dir / f"male_{i}.wav"), audio, sample_rate)
    
    return data_dir


@pytest.fixture
def sample_features():
    """Generate sample MFCC features"""
    n_mfcc = 40
    features = np.random.randn(n_mfcc * 2)  # mean + std
    return features


@pytest.fixture
def sample_training_data():
    """Generate sample training data"""
    n_samples = 100
    n_features = 80
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    return X, y