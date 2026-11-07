import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Fourier Series approach
    # Domain x is in [-2, 2]
    # We use 10 parameters:
    # p[0]: DC (constant)
    # p[1]: sin(x)
    # p[2]: cos(x)
    # p[3]: sin(2x)
    # p[4]: cos(2x)
    # p[5]: sin(3x)
    # p[6]: cos(3x)
    # p[7]: sin(4x)
    # p[8]: cos(4x)
    # p[9]: sin(5x)
    
    n = len(x)
    basis = np.zeros((n, 10))
    
    # DC term
    basis[:, 0] = 1.0
    
    # Harmonic 1
    basis[:, 1] = np.sin(x)
    basis[:, 2] = np.cos(x)
    
    # Harmonic 2
    basis[:, 3] = np.sin(2 * x)
    basis[:, 4] = np.cos(2 * x)
    
    # Harmonic 3
    basis[:, 5] = np.sin(3 * x)
    basis[:, 6] = np.cos(3 * x)
    
    # Harmonic 4
    basis[:, 7] = np.sin(4 * x)
    basis[:, 8] = np.cos(4 * x)
    
    # Harmonic 5
    basis[:, 9] = np.sin(5 * x)
    
    return basis @ params
