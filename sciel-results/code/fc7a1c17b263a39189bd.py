import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Chebyshev polynomials of the first kind
    # T0 = 1
    # T1 = x
    # T2 = 2x^2 - 1
    # T3 = 4x^3 - 3x
    # T4 = 8x^4 - 8x^2 + 1
    
    p = params.astype(float)
    
    # Precompute Chebyshev terms for x and y
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2 * x**2 - 1
    T3_x = 4 * x**3 - 3 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2 * y**2 - 1
    T3_y = 4 * y**3 - 3 * y
    
    # Basis terms (10 parameters)
    # We use the tensor product up to degree 2 (9 terms)
    # and add T3_x to utilize the 10th parameter.
    # To maintain symmetry, maybe we should add a term that averages or picks one?
    # If the function is symmetric, T3_x and T3_y might both be needed.
    # However, we only have 1 slot.
    # Let's try adding T3_x * T0_y.
    
    term1 = T0_x * T0_y # 1
    term2 = T1_x * T0_y # x
    term3 = T0_x * T1_y # y
    term4 = T2_x * T0_y # 2x^2 - 1
    term5 = T1_x * T1_y # xy
    term6 = T0_x * T2_y # 2y^2 - 1
    term7 = T2_x * T1_y # (2x^2 - 1)y
    term8 = T1_x * T2_y # x(2y^2 - 1)
    term9 = T2_x * T2_y # (2x^2 - 1)(2y^2 - 1)
    term10 = T3_x * T0_y # 4x^3 - 3x
    
    # Linear combination
    return (p[0] * term1 + 
            p[1] * term2 + 
            p[2] * term3 + 
            p[3] * term4 + 
            p[4] * term5 + 
            p[5] * term6 + 
            p[6] * term7 + 
            p[7] * term8 + 
            p[8] * term9 + 
            p[9] * term10)
