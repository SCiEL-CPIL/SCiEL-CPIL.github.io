"""Rewritten program for symbolic regression with 2 inputs."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0  # 10s safety margin

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a safe prediction (e.g., mean of y if available, or 0)
        # Since we don't have y here, return 0 or params[0]
        return np.zeros_like(x)

    # Fourier Basis
    # p[0]: 1
    # p[1]: cos(x)
    # p[2]: sin(x)
    # p[3]: cos(y)
    # p[4]: sin(y)
    # p[5]: cos(x+y)
    # p[6]: sin(x+y)
    # p[7]: cos(x-y)
    # p[8]: sin(x-y)
    # p[9]: cos(2x)
    
    p = params
    
    term_0 = 1.0
    term_1 = np.cos(x)
    term_2 = np.sin(x)
    term_3 = np.cos(y)
    term_4 = np.sin(y)
    term_5 = np.cos(x + y)
    term_6 = np.sin(x + y)
    term_7 = np.cos(x - y)
    term_8 = np.sin(x - y)
    term_9 = np.cos(2 * x)
    
    return (p[0] * term_0 + 
            p[1] * term_1 + 
            p[2] * term_2 + 
            p[3] * term_3 + 
            p[4] * term_4 + 
            p[5] * term_5 + 
            p[6] * term_6 + 
            p[7] * term_7 + 
            p[8] * term_8 + 
            p[9] * term_9)
# EVOLVE-BLOCK-END
