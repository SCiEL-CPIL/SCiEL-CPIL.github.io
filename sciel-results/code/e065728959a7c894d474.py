# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x: array of input points (uniformly from [-2, 2])
    params: array of 10 coefficients
    Returns: predicted values, same shape as x
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    m = len(x)
    
    # Fourier basis on [-2, 2]
    # Frequencies: 0.5, 1.0, 1.5, 2.0, 2.5 (multiplied by pi)
    # We use 10 params:
    # 0: Constant
    # 1: cos(0.5*pi*x)
    # 2: sin(0.5*pi*x)
    # 3: cos(1.0*pi*x)
    # 4: sin(1.0*pi*x)
    # 5: cos(1.5*pi*x)
    # 6: sin(1.5*pi*x)
    # 7: cos(2.0*pi*x)
    # 8: sin(2.0*pi*x)
    # 9: cos(2.5*pi*x)
    
    basis = np.zeros((m, 10))
    
    basis[:, 0] = 1.0
    
    # k=0.5
    basis[:, 1] = np.cos(0.5 * np.pi * x)
    basis[:, 2] = np.sin(0.5 * np.pi * x)
    
    # k=1.0
    basis[:, 3] = np.cos(1.0 * np.pi * x)
    basis[:, 4] = np.sin(1.0 * np.pi * x)
    
    # k=1.5
    basis[:, 5] = np.cos(1.5 * np.pi * x)
    basis[:, 6] = np.sin(1.5 * np.pi * x)
    
    # k=2.0
    basis[:, 7] = np.cos(2.0 * np.pi * x)
    basis[:, 8] = np.sin(2.0 * np.pi * x)
    
    # k=2.5
    basis[:, 9] = np.cos(2.5 * np.pi * x)
    
    pred = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return pred
# EVOLVE-BLOCK-END
