"""Symbolic regression with learnable-center Gaussians and rich trig basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has exactly 10 elements
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Structural change: learnable-center Gaussian bumps
    # Instead of fixed centers, encode center via parameter:
    #   center_i = 1.5 * tanh(p[i])  -> maps to [-1.5, 1.5], well-conditioned
    #   Fixed width = 0.7 to avoid ill-conditioning
    #
    # Basis:
    # p[0]: constant
    # p[1]: x
    # p[2]: x^2
    # p[3]: x^3
    # p[4]: sin(x)
    # p[5]: cos(x)
    # p[6]: sin(2x)
    # p[7]: cos(2x)
    # p[8]: amplitude for learnable-center Gaussian (center = 1.5*tanh(p[8]))
    # p[9]: amplitude for second learnable-center Gaussian (center = 1.5*tanh(p[9]))
    
    # Compute learnable centers (smooth, bounded, differentiable)
    c1 = 1.5 * np.tanh(p[8])
    c2 = 1.5 * np.tanh(p[9])
    
    # Fixed width to keep gradients well-behaved for BFGS
    width = 0.7
    w2 = width * width
    
    term0 = np.ones_like(x)
    term1 = x
    term2 = x ** 2
    term3 = x ** 3
    term4 = np.sin(x)
    term5 = np.cos(x)
    term6 = np.sin(2.0 * x)
    term7 = np.cos(2.0 * x)
    # Learnable-center Gaussians: amplitude * exp(-((x - center)^2) / w^2)
    term8 = np.exp(-((x - c1) ** 2) / w2)
    term9 = np.exp(-((x - c2) ** 2) / w2)
    
    return (p[0] * term0 + p[1] * term1 + p[2] * term2 + p[3] * term3 +
            p[4] * term4 + p[5] * term5 + p[6] * term6 + p[7] * term7 +
            p[8] * term8 + p[9] * term9)
# EVOLVE-BLOCK-END
