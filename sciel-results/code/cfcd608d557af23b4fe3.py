import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        x = x.flatten()
        
    # Define frequencies
    k = 1
    
    # Normalization constants for [-3, 3]
    # Integral of 1^2 is 6
    norm_const = np.sqrt(6.0)
    
    # We will build the basis matrix
    # Columns:
    # 0: 1 / norm_const
    # 1: sin(x) / norm_s1
    # 2: cos(x) / norm_c1
    # ...
    
    # Pre-compute norms for k=1..5
    # norm_s_k = sqrt(3 - sin(6k)/(2k))
    # norm_c_k = sqrt(3 + sin(6k)/(2k))
    
    # We need 10 terms.
    # Let's use:
    # 0: Constant
    # 1: sin(x)
    # 2: cos(x)
    # 3: sin(2x)
    # 4: cos(2x)
    # 5: sin(3x)
    # 6: cos(3x)
    # 7: sin(4x)
    # 8: cos(4x)
    # 9: sin(5x)
    
    # This matches the current term selection.
    
    # Calculate norms
    # k=1
    norm_s1 = np.sqrt(3.0 - np.sin(6.0) / 2.0)
    norm_c1 = np.sqrt(3.0 + np.sin(6.0) / 2.0)
    # k=2
    norm_s2 = np.sqrt(3.0 - np.sin(12.0) / 4.0)
    norm_c2 = np.sqrt(3.0 + np.sin(12.0) / 4.0)
    # k=3
    norm_s3 = np.sqrt(3.0 - np.sin(18.0) / 6.0)
    norm_c3 = np.sqrt(3.0 + np.sin(18.0) / 6.0)
    # k=4
    norm_s4 = np.sqrt(3.0 - np.sin(24.0) / 8.0)
    norm_c4 = np.sqrt(3.0 + np.sin(24.0) / 8.0)
    # k=5
    norm_s5 = np.sqrt(3.0 - np.sin(30.0) / 10.0)
    # norm_c5 not used
    
    # Build basis
    # Using matrix multiplication for efficiency and clarity
    
    # Create a matrix of shape (N, 10)
    N = len(x)
    B = np.empty((N, 10), dtype=float)
    
    B[:, 0] = 1.0 / norm_const
    
    B[:, 1] = np.sin(x) / norm_s1
    B[:, 2] = np.cos(x) / norm_c1
    
    B[:, 3] = np.sin(2 * x) / norm_s2
    B[:, 4] = np.cos(2 * x) / norm_c2
    
    B[:, 5] = np.sin(3 * x) / norm_s3
    B[:, 6] = np.cos(3 * x) / norm_c3
    
    B[:, 7] = np.sin(4 * x) / norm_s4
    B[:, 8] = np.cos(4 * x) / norm_c4
    
    B[:, 9] = np.sin(5 * x) / norm_s5
    
    # Prediction is B @ params
    result = B.dot(params)
    
    return result
