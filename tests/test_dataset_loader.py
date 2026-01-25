import pytest
import numpy as np
from dataset_loader import DatasetLoader


class TestDatasetLoader:
    """Test DatasetLoader class"""
    
    def test_initialization(self, test_config):
        """Test dataset loader initialization"""
        extractor = AudioFeatureExtractor(test_config)
        loader = DatasetLoader(extractor)
        
        assert loader.feature_extractor is not None
    
    def test_load_from_directory(self, test_config, sample_dataset):
        """Test loading dataset from directory"""
        extractor = AudioFeatureExtractor(test_config)
        loader = DatasetLoader(extractor)
        
        X, y = loader.load_from_directory(sample_dataset, ["female", "male"])
        
        assert X.shape[0] == 10  # 5 female + 5 male
        assert y.shape[0] == 10
        assert X.shape[1] == test_config.n_mfcc * 2
        assert set(y) == {0, 1}  # Both classes present
    
    def test_load_from_directory_missing_class(self, test_config, temp_dir):
        """Test loading with missing class directory"""
        extractor = AudioFeatureExtractor(test_config)
        loader = DatasetLoader(extractor)
        
        data_dir = temp_dir / "data"
        data_dir.mkdir()
        (data_dir / "female").mkdir()
        # Missing male directory
        
        X, y = loader.load_from_directory(data_dir, ["female", "male"])
        
        assert len(X) == 0  # No data loaded
        assert len(y) == 0
    
    def test_load_from_directory_nonexistent(self, test_config):
        """Test loading from non-existent directory"""
        extractor = AudioFeatureExtractor(test_config)
        loader = DatasetLoader(extractor)
        
        with pytest.raises(FileNotFoundError):
            loader.load_from_directory(Path("nonexistent"), ["female", "male"])
    
    def test_load_from_file_list(self, test_config, sample_dataset):
        """Test loading from file list"""
        extractor = AudioFeatureExtractor(test_config)
        loader = DatasetLoader(extractor)
        
        # Get sample files
        female_files = list((sample_dataset / "female").glob("*.wav"))
        paths = [str(f) for f in female_files[:3]]
        labels = [0, 0, 0]
        
        X, y = loader.load_from_file_list(paths, labels)
        
        assert X.shape[0] == 3
        assert np.all(y == 0)

