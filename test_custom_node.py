import unittest
import torch
import numpy as np
from nodes.autoregressive_transformer import AutoregressiveTransformer
from nodes.rolling_diffusion_decoder import RollingDiffusionDecoder
from nodes.flux_dev_integration import FluxDevModelIntegration

class MockModel:
    """Mock model for testing purposes."""
    def __init__(self):
        self.device = "cpu"
    
    def __call__(self, *args, **kwargs):
        # Return a mock result
        return {"images": torch.zeros((1, 3, 512, 512))}

class TestAutoregressiveTransformer(unittest.TestCase):
    """Test cases for the AutoregressiveTransformer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.transformer = AutoregressiveTransformer(device="cpu")
        self.mock_model = MockModel()
        self.latent = torch.randn((1, 4, 64, 64))
        self.conditioning = {"positive": {"text": "test"}, "negative": {"text": "test"}, "cfg": 7.0}
    
    def test_initialization(self):
        """Test initialization of the transformer."""
        self.assertEqual(self.transformer.order, "top-to-bottom")
        self.assertEqual(self.transformer.attention_persistence, 0.5)
        self.assertEqual(self.transformer.device, "cpu")
        self.assertIsNone(self.transformer.attention_cache)
    
    def test_get_generation_order(self):
        """Test generation order calculation."""
        # Test top-to-bottom order
        order = self.transformer._get_generation_order(2, 2)
        expected_order = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.assertEqual(order, expected_order)
        
        # Test left-to-right order
        self.transformer.order = "left-to-right"
        order = self.transformer._get_generation_order(2, 2)
        expected_order = [(0, 0), (1, 0), (0, 1), (1, 1)]
        self.assertEqual(order, expected_order)
        
        # Test spiral-out order
        self.transformer.order = "spiral-out"
        order = self.transformer._get_generation_order(3, 3)
        # The center should be the first element
        self.assertEqual(order[0], (1, 1))
    
    def test_generate_sequence(self):
        """Test sequence generation."""
        result = self.transformer.generate_sequence(
            self.mock_model, self.latent, self.conditioning, num_steps=2
        )
        
        # Check that the result has the same shape as the input
        self.assertEqual(result.shape, self.latent.shape)

class TestRollingDiffusionDecoder(unittest.TestCase):
    """Test cases for the RollingDiffusionDecoder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.decoder = RollingDiffusionDecoder(window_size=32, overlap=8, device="cpu")
        self.mock_model = MockModel()
        self.latent = torch.randn((1, 4, 64, 64))
        self.conditioning = {"positive": {"text": "test"}, "negative": {"text": "test"}, "cfg": 7.0}
    
    def test_initialization(self):
        """Test initialization of the decoder."""
        self.assertEqual(self.decoder.window_size, 32)
        self.assertEqual(self.decoder.overlap, 8)
        self.assertEqual(self.decoder.device, "cpu")
    
    def test_create_blend_weights(self):
        """Test creation of blending weights."""
        weights = self.decoder._create_blend_weights(4)
        
        # Check that weights start at 0 and end at 1
        self.assertAlmostEqual(weights[0].item(), 0.0)
        self.assertAlmostEqual(weights[-1].item(), 1.0)
        
        # Check that weights are monotonically increasing
        for i in range(1, len(weights)):
            self.assertGreater(weights[i].item(), weights[i-1].item())
    
    def test_process_latent(self):
        """Test latent processing."""
        result = self.decoder.process_latent(
            self.mock_model, self.latent, self.conditioning, steps=2
        )
        
        # Check that the result has the same shape as the input
        self.assertEqual(result.shape, self.latent.shape)

class TestFluxDevIntegration(unittest.TestCase):
    """Test cases for the FluxDevModelIntegration class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Skip actual model loading for tests
        self.integration = FluxDevModelIntegration(device="cpu")
        self.integration.model = MockModel()
    
    def test_process_prompt(self):
        """Test prompt processing."""
        result = self.integration.process_prompt("test prompt")
        
        # Check that the result contains the prompt
        self.assertEqual(result["text"], "test prompt")
    
    def test_latent_to_image(self):
        """Test latent to image conversion."""
        latent = torch.randn((1, 4, 64, 64))
        result = self.integration.latent_to_image(latent)
        
        # Check that the result has the expected shape (upscaled by 8x)
        self.assertEqual(result.shape, (1, 3, 64*8, 64*8))

if __name__ == "__main__":
    unittest.main()
