import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale inputs to O(1) to improve conditioning for BFGS
    # x range ~ [-0.57, 0.53], center 0, scale 0.5
    # v range ~ [-0.18, 0.28], center 0, scale 0.25
    X = x / 0.5
    V = v / 0.25
    
    # Precompute powers
    X2 = X * X
    X3 = X2 * X
    V2 = V * V
    V3 = V2 * V
    XV = X * V
    X2V = X2 * V
    XV2 = X * V2
    
    # Linear combination of 10 basis functions
    # p0: 1 (bias)
    # p1: X
    # p2: X^2
    # p3: X^3
    # p4: V
    # p5: V^2
    # p6: V^3
    # p7: XV
    # p8: X^2V
    # p9: XV^2
    
    return (params[0] + 
            params[1] * X + 
            params[2] * X2 + 
            params[3] * X3 + 
            params[4] * V + 
            params[5] * V2 + 
            params[6] * V3 + 
            params[7] * XV + 
            params[8] * X2V + 
            params[9] * XV2)
