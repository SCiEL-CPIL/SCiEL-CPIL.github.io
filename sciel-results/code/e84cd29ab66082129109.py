"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Adaptive basis: some basis functions depend on params themselves.
    Structure:
      params[0:3]  -> linear coefficients for [1, x, x^2]
      params[3]    -> amplitude of sin(freq*x)
      params[4]    -> amplitude of cos(freq*x)
      params[5]    -> frequency (clipped to [0.1, 10])
      params[6]    -> center of Gaussian bump
      params[7]    -> width of Gaussian bump (clipped to [0.1, 5])
      params[8]    -> amplitude of tanh((x - c2)/w2)
      params[9]    -> center of tanh (width fixed at 1.0)
    """
    x = np.asarray(x, dtype=float)
    
    # Unpack params
    c0 = params[0]
    c1 = params[1]
    c2 = params[2]
    a_sin = params[3]
    a_cos = params[4]
    freq = np.clip(params[5], 0.1, 10.0)
    g_center = params[6]
    g_width = np.clip(params[7], 0.1, 5.0)
    a_tanh = params[8]
    t_center = params[9]
    
    # Polynomial part
    poly = c0 + c1 * x + c2 * x**2
    
    # Periodic part with parametric frequency
    periodic = a_sin * np.sin(freq * x) + a_cos * np.cos(freq * x)
    
    # Gaussian bump with parametric center and width
    gaussian = np.exp(-((x - g_center) / g_width) ** 2)
    
    # Hyperbolic tangent step with parametric center
    tanh_step = a_tanh * np.tanh((x - t_center) / 1.0)
    
    return poly + periodic + gaussian + tanh_step
# EVOLVE-BLOCK-END
