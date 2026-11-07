# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier spectral basis with non-integer (irrational) frequencies.
    
    Basis functions (10 total):
    0: 1 (DC / bias)
    1: sin(2.17 * x)
    2: sin(3.41 * y)
    3: sin(2.83 * z)
    4: cos(1.73 * x + 2.91 * y)
    5: cos(2.47 * y + 3.13 * z)
    6: cos(1.93 * z + 2.29 * x)
    7: sin(3.71 * x * y)
    8: sin(2.63 * y * z)
    9: sin(4.19 * z * x)
    
    Rationale:
    - Non-integer frequencies avoid aliasing and ensure linear independence
    - Cross-frequency terms capture multiplicative interactions in frequency domain
    - Fourier basis can approximate any L2 function on bounded domain (Weierstrass)
    - Different structural family from polynomial/reciprocal bases
    """
    p = params
    
    # Precompute frequency constants (irrational approximations for independence)
    f1 = 2.17
    f2 = 3.41
    f3 = 2.83
    f4 = 1.73
    f5 = 2.91
    f6 = 2.47
    f7 = 3.13
    f8 = 1.93
    f9 = 2.29
    f10 = 3.71
    f11 = 2.63
    f12 = 4.19
    
    b0 = np.ones_like(x)
    b1 = np.sin(f1 * x)
    b2 = np.sin(f2 * y)
    b3 = np.sin(f3 * z)
    b4 = np.cos(f4 * x + f5 * y)
    b5 = np.cos(f6 * y + f7 * z)
    b6 = np.cos(f8 * z + f9 * x)
    b7 = np.sin(f10 * x * y)
    b8 = np.sin(f11 * y * z)
    b9 = np.sin(f12 * z * x)
    
    return (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
            p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
            p[8] * b8 + p[9] * b9)
# EVOLVE-BLOCK-END
