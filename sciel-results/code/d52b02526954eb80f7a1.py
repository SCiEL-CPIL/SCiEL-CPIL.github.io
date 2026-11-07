import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Chebyshev Polynomial approach
    # Domain x is in [-2, 2]
    # We use 10 parameters for Chebyshev polynomials T_0 to T_9
    # T_k(z) where z = x / 2.0
    
    n = len(x)
    z = x / 2.0  # Scale to [-1, 1]
    
    basis = np.zeros((n, 10))
    
    # T_0(z) = 1
    basis[:, 0] = 1.0
    
    if n > 0:
        # T_1(z) = z
        basis[:, 1] = z
        
        # Recurrence: T_k(z) = 2*z*T_{k-1}(z) - T_{k-2}(z)
        # T_2(z)
        if n > 0:
            basis[:, 2] = 2 * z * basis[:, 1] - basis[:, 0]
            
        # T_3(z)
        if n > 0:
            basis[:, 3] = 2 * z * basis[:, 2] - basis[:, 1]
            
        # T_4(z)
        if n > 0:
            basis[:, 4] = 2 * z * basis[:, 3] - basis[:, 2]
            
        # T_5(z)
        if n > 0:
            basis[:, 5] = 2 * z * basis[:, 4] - basis[:, 3]
            
        # T_6(z)
        if n > 0:
            basis[:, 6] = 2 * z * basis[:, 5] - basis[:, 4]
            
        # T_7(z)
        if n > 0:
            basis[:, 7] = 2 * z * basis[:, 6] - basis[:, 5]
            
        # T_8(z)
        if n > 0:
            basis[:, 8] = 2 * z * basis[:, 7] - basis[:, 6]
            
        # T_9(z)
        if n > 0:
            basis[:, 9] = 2 * z * basis[:, 8] - basis[:, 7]

    return basis @ params
