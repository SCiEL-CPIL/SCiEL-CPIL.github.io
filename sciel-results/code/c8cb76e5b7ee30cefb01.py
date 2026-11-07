"""Symbolic regression: recover an unknown function from 1 input with 10 params."""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Basis function expansion with 10 parameters.
    
    We use a combination of:
    - Constant term
    - Linear term
    - Quadratic term
    - Cubic term (scaled)
    - Sine terms (2 frequencies)
    - Cosine terms (2 frequencies)
    - Gaussian bump (centered, width param)
    
    The basis functions are chosen to be well-conditioned for BFGS starting from ones.
    x is in [-3, 3], so we scale appropriately.
    """
    p = params  # length 10
    
    # Scale x to [-1, 1] for better conditioning
    xs = x / 3.0
    
    # Polynomial basis (scaled to avoid overflow)
    term_const = np.ones_like(xs)
    term_lin = xs
    term_quad = xs * xs
    term_cubic = xs * xs * xs
    
    # Trigonometric basis: frequencies chosen so periods fit in [-1,1]
    # k=1: period 2, k=2: period 1
    term_sin1 = np.sin(np.pi * xs)
    term_cos1 = np.cos(np.pi * xs)
    term_sin2 = np.sin(2 * np.pi * xs)
    term_cos2 = np.cos(2 * np.pi * xs)
    
    # Gaussian bump: exp(-(xs - center)^2 / width^2)
    # Use a fixed center at 0 and a fixed width to keep it simple
    # Actually, let's use a parametric width for flexibility
    # But we only have 10 params. Let's use:
    # p[0]: const
    # p[1]: lin
    # p[2]: quad
    # p[3]: cubic
    # p[4]: sin(pi*xs)
    # p[5]: cos(pi*xs)
    # p[6]: sin(2*pi*xs)
    # p[7]: cos(2*pi*xs)
    # p[8]: gaussian amplitude
    # p[9]: gaussian width (log-scale for positivity)
    
    # Gaussian with fixed center at 0
    width = np.exp(p[9])  # ensures positive width
    term_gauss = np.exp(-xs * xs / (width * width + 1e-8))
    
    result = (p[0] * term_const +
              p[1] * term_lin +
              p[2] * term_quad +
              p[3] * term_cubic +
              p[4] * term_sin1 +
              p[5] * term_cos1 +
              p[6] * term_sin2 +
              p[7] * term_cos2 +
              p[8] * term_gauss)
    
    return result
