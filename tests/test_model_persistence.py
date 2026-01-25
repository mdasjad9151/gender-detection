import pytest
from model_persistence import ModelPersistence
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class TestModelPersistence:
    """Test ModelPersistence class"""
    
    def test_initialization(self, test_config):
        """Test model persistence initialization"""
        persistence = ModelPersistence(test_config)
        
        assert persistence.model_path.parent.exists()
    
    def test_save_and_load_model(self, test_config, sample_training_data):
        """Test saving and loading model"""
        X, y = sample_training_data
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        scaler = StandardScaler()
        
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)
        
        # Save
        persistence = ModelPersistence(test_config)
        metrics = {'accuracy': 0.95, 'f1_score': 0.94}
        persistence.save_model(model, scaler, metrics)
        
        # Check files exist
        assert persistence.model_path.exists()
        assert persistence.scaler_path.exists()
        assert persistence.config_path.exists()
        
        # Load
        loaded_model, loaded_scaler = persistence.load_model()
        
        assert loaded_model is not None
        assert loaded_scaler is not None
    
    def test_load_model_not_exists(self, test_config):
        """Test loading non-existent model"""
        persistence = ModelPersistence(test_config)
        
        with pytest.raises(FileNotFoundError):
            persistence.load_model()
    
    def test_model_exists(self, test_config):
        """Test model existence check"""
        persistence = ModelPersistence(test_config)
        
        assert not persistence.model_exists()
        
        # Create dummy files
        persistence.model_path.parent.mkdir(parents=True, exist_ok=True)
        persistence.model_path.touch()
        persistence.scaler_path.touch()
        
        assert persistence.model_exists()
    
    def test_load_config(self, test_config, sample_training_data):
        """Test loading saved configuration"""
        X, y = sample_training_data
        
        model = RandomForestClassifier(n_estimators=10)
        scaler = StandardScaler()
        
        persistence = ModelPersistence(test_config)
        persistence.save_model(model, scaler, {'accuracy': 0.95})
        
        config = persistence.load_config()
        
        assert config['sample_rate'] == test_config.sample_rate
        assert config['n_mfcc'] == test_config.n_mfcc
        assert 'metrics' in config
