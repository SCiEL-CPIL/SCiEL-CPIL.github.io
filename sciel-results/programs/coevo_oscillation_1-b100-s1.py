import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-inspired basis for a nonlinear oscillator.
    Uses scaled inputs to improve BFGS conditioning.
    Includes polynomial terms up to 5th order in x and v, plus cross terms and trig functions.
    """
    # Deadline guard: if we're close to the limit, return a simple fallback
    # But since equation is called many times during fitting, we just compute fast.
    
    # Scale inputs to be roughly in [-1, 1]
    X = x / 0.6
    V = v / 0.3
    
    # Polynomial terms
    X2 = X * X
    X3 = X2 * X
    X4 = X2 * X2
    X5 = X3 * X2
    
    V2 = V * V
    V3 = V2 * V
    V4 = V2 * V2
    V5 = V3 * V2
    
    # Cross terms
    XV = X * V
    X2V = X2 * V
    XV2 = X * V2
    
    # Trig terms
    sinX = np.sin(X)
    sinV = np.sin(V)
    
    # Basis:
    # 0: 1
    # 1: X
    # 2: V
    # 3: X^2
    # 4: X^3
    # 5: X^4
    # 6: X^5
    # 7: V^2
    # 8: V^3
    # 9: X*V
    
    # We have 10 params. Let's choose the most physically relevant ones.
    # For a Duffing oscillator: y = -k*x - c*x^3 - d*v - e*v^3 + ...
    # But we also need to capture the data well.
    
    # Let's use:
    # 0: 1
    # 1: X
    # 2: X^3
    # 3: X^5
    # 4: V
    # 5: V^3
    # 6: X*V
    # 7: X^2*V
    # 8: sin(X)
    # 9: sin(V)
    
    t0 = np.ones_like(x)
    t1 = X
    t2 = X3
    t3 = X5
    t4 = V
    t5 = V3
    t6 = XV
    t7 = X2V
    t8 = sinX
    t9 = sinV
    
    T = np.column_stack([t0, t1, t2, t3, t4, t5, t6, t7, t8, t9])
    
    return T @ params
