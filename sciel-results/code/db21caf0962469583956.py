import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # x is expected to be in [0.1, 6]
    # Map to [0, 2*pi] for Fourier series
    t = (x - 0.1) / 5.9 * 2.0 * np.pi
    
    y = np.zeros_like(x)
    
    # params[0]: constant term
    y += params[0] * 1.0
    
    if len(params) > 1:
        # params[1]: sin(t)
        y += params[1] * np.sin(t)
    
    if len(params) > 2:
        # params[2]: cos(t)
        y += params[2] * np.cos(t)
    
    if len(params) > 3:
        # params[3]: sin(2*t)
        y += params[3] * np.sin(2.0 * t)
    
    if len(params) > 4:
        # params[4]: cos(2*t)
        y += params[4] * np.cos(2.0 * t)
    
    if len(params) > 5:
        # params[5]: sin(3*t)
        y += params[5] * np.sin(3.0 * t)
    
    if len(params) > 6:
        # params[6]: cos(3*t)
        y += params[6] * np.cos(3.0 * t)
    
    if len(params) > 7:
        # params[7]: sin(4*t)
        y += params[7] * np.sin(4.0 * t)
    
    if len(params) > 8:
        # params[8]: cos(4*t)
        y += params[8] * np.cos(4.0 * t)
    
    # params[9]: linear term to capture non-periodic trends
    # This helps if the target function isn't purely periodic
    y += params[9] * (x - 3.05)
    
    return y
