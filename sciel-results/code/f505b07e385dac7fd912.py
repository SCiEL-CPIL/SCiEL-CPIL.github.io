import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # x is in [-3, 3]
    # Basis functions: 1, cos(k*pi*x/3), sin(k*pi*x/3)
    
    # Precompute arguments
    # k = 1..4 for full pairs, k=5 for cosine only
    # Params:
    # p0: 1
    # p1: cos(pi*x/3)
    # p2: sin(pi*x/3)
    # p3: cos(2*pi*x/3)
    # p4: sin(2*pi*x/3)
    # p5: cos(3*pi*x/3)
    # p6: sin(3*pi*x/3)
    # p7: cos(4*pi*x/3)
    # p8: sin(4*pi*x/3)
    # p9: cos(5*pi*x/3)
    
    arg = np.pi * x / 3.0
    
    y = params[0] * np.ones_like(x)
    
    y += params[1] * np.cos(arg)
    y += params[2] * np.sin(arg)
    
    y += params[3] * np.cos(2 * arg)
    y += params[4] * np.sin(2 * arg)
    
    y += params[5] * np.cos(3 * arg)
    y += params[6] * np.sin(3 * arg)
    
    y += params[7] * np.cos(4 * arg)
    y += params[8] * np.sin(4 * arg)
    
    y += params[9] * np.cos(5 * arg)
    
    return y
