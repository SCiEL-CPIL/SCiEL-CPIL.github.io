# EVOLVE-BLOCK-START
import numpy as np

# Pre-computed RMS scales for basis functions on uniform [-3, 3]
# Index: 0:1, 1:x, 2:x^2, 3:x^3, 4:sin(x), 5:cos(x), 6:sin(2x), 7:cos(2x), 8:1/(1+x^2), 9:exp(-x^2/4)
_SCALES = np.array([
    1.0,                 # 1
    np.sqrt(3.0),        # x
    np.sqrt(81.0 / 5.0), # x^2
    np.sqrt(729.0 / 7.0),# x^3
    np.sqrt((6.0 - np.sin(6.0)) / 12.0), # sin(x)
    np.sqrt((6.0 + np.sin(6.0)) / 12.0), # cos(x)
    np.sqrt((12.0 - np.sin(12.0)) / 24.0), # sin(2x)
    np.sqrt((12.0 + np.sin(12.0)) / 24.0), # cos(2x)
    0.508,               # 1/(1+x^2) approx
    0.645                # exp(-x^2/4) approx
])


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the hidden function using a diverse basis of 10 functions,
    normalized to have unit RMS on [-3, 3] to improve optimizer conditioning.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    
    # Build the 10 basis functions
    c0 = np.ones(n, dtype=np.float64)
    c1 = x
    c2 = x * x
    c3 = c2 * x
    s1 = np.sin(x)
    s2 = np.cos(x)
    s3 = np.sin(2.0 * x)
    s4 = np.cos(2.0 * x)
    s5 = 1.0 / (1.0 + c2)
    s6 = np.exp(-c2 / 4.0)
    
    # Stack into (n, 10) matrix
    V = np.empty((n, 10), dtype=np.float64)
    V[:, 0] = c0
    V[:, 1] = c1
    V[:, 2] = c2
    V[:, 3] = c3
    V[:, 4] = s1
    V[:, 5] = s2
    V[:, 6] = s3
    V[:, 7] = s4
    V[:, 8] = s5
    V[:, 9] = s6
    
    # Normalize columns by their pre-computed RMS scales
    V /= _SCALES
    
    return V @ params
# EVOLVE-BLOCK-END
