import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Fourier-like basis functions
    # 1. Constant
    term0 = 1.0
    
    # 2. x terms
    term1 = np.sin(x)
    term2 = np.cos(x)
    
    # 3. y terms
    term3 = np.sin(y)
    term4 = np.cos(y)
    
    # 4. Interactions
    term5 = np.sin(x) * np.cos(y)
    term6 = np.cos(x) * np.sin(y)
    
    # 5. Sum/Difference frequencies
    term7 = np.sin(x + y)
    term8 = np.cos(x + y)
    
    # 6. Higher frequency interaction
    term9 = np.sin(2 * x) * np.sin(2 * y)
    
    return (p0 * term0 + p1 * term1 + p2 * term2 + p3 * term3 + p4 * term4 + 
            p5 * term5 + p6 * term6 + p7 * term7 + p8 * term8 + p9 * term9)
