"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure x is a 1D array for safe operations, though it usually is
    # x shape: (N,)
    
    # Compute basis functions
    # We use 10 basis functions to match the 10 parameters
    # Using orthogonalized polynomials for better conditioning
    
    # 1. Constant
    f0 = np.ones_like(x)
    # 2. Linear
    f1 = x
    # 3. Quadratic (shifted to be orthogonal to constant on [-3,3])
    f2 = x**2 - 3.0
    # 4. Cubic (shifted to be orthogonal to constant and linear)
    f3 = x**3 - 2.7 * x
    # 5. Sine
    f4 = np.sin(x)
    # 6. Cosine
    f5 = np.cos(x)
    # 7. Sine 2x
    f6 = np.sin(2 * x)
    # 8. Cosine 2x
    f7 = np.cos(2 * x)
    # 9. Tanh
    f8 = np.tanh(x)
    # 10. Sine 3x (higher frequency term)
    f9 = np.sin(3 * x)
    
    # Stack features into a matrix of shape (N, 10)
    # Columns correspond to params[0] ... params[9]
    features = np.column_stack((f0, f1, f2, f3, f4, f5, f6, f7, f8, f9))
    
    # Linear combination
    # params shape: (10,)
    # features shape: (N, 10)
    # Result shape: (N,)
    return features.dot(params)

# EVOLVE-BLOCK-END
