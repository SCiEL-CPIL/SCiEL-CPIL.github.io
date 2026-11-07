"""Fourier basis symbolic regression candidate."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Fourier basis: constant + 5 sine/cosine pairs at frequencies 1..5
    # This gives 1 + 5*2 = 11 terms, but we have 10 params.
    # Use: 1, sin(x), cos(x), sin(2x), cos(2x), sin(3x), cos(3x), sin(4x), cos(4x), sin(5x)
    # That's 10 features.
    
    freqs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Build Fourier features
    # Feature 0: 1 (constant)
    # Features 1-2: sin(x), cos(x)
    # Features 3-4: sin(2x), cos(2x)
    # Features 5-6: sin(3x), cos(3x)
    # Features 7-8: sin(4x), cos(4x)
    # Feature 9: sin(5x)
    
    ones = np.ones_like(x)
    sin1 = np.sin(x)
    cos1 = np.cos(x)
    sin2 = np.sin(2 * x)
    cos2 = np.cos(2 * x)
    sin3 = np.sin(3 * x)
    cos3 = np.cos(3 * x)
    sin4 = np.sin(4 * x)
    cos4 = np.cos(4 * x)
    sin5 = np.sin(5 * x)
    
    features = np.column_stack([
        ones,
        sin1,
        cos1,
        sin2,
        cos2,
        sin3,
        cos3,
        sin4,
        cos4,
        sin5
    ])  # (N, 10)
    
    return features @ params
# EVOLVE-BLOCK-END
