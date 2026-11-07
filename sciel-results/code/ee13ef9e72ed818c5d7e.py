import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-3, 3] to [-pi, pi] for natural Fourier period
    arg = np.pi * x / 3.0
    
    # 10 Fourier basis terms:
    # p0: 1 (DC)
    # p1: cos(arg), p2: sin(arg)       -> k=1
    # p3: cos(2*arg), p4: sin(2*arg)   -> k=2
    # p5: cos(3*arg), p6: sin(3*arg)   -> k=3
    # p7: cos(4*arg), p8: sin(4*arg)   -> k=4
    # p9: cos(5*arg)                   -> k=5 cosine only
    
    y = params[0] * np.ones_like(x)
    y += params[1] * np.cos(arg)
    y += params[2] * np.sin(arg)
    y += params[3] * np.cos(2.0 * arg)
    y += params[4] * np.sin(2.0 * arg)
    y += params[5] * np.cos(3.0 * arg)
    y += params[6] * np.sin(3.0 * arg)
    y += params[7] * np.cos(4.0 * arg)
    y += params[8] * np.sin(4.0 * arg)
    y += params[9] * np.cos(5.0 * arg)
    
    return y
