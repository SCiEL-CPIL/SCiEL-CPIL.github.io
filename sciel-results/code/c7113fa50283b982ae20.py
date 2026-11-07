# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fixed-Center Radial Basis Function (RBF) approximation.
    
    Uses 5 RBFs centered at specific points in the domain [0.5, 2]^3.
    Parameters control the amplitude and width of each RBF.
    
    Centers:
    0: (0.5, 0.5, 0.5)
    1: (2.0, 0.5, 0.5)
    2: (0.5, 2.0, 0.5)
    3: (0.5, 0.5, 2.0)
    4: (1.25, 1.25, 1.25)
    
    Params:
    p0, p1: Amplitude, log-Width for Center 0
    p2, p3: Amplitude, log-Width for Center 1
    p4, p5: Amplitude, log-Width for Center 2
    p6, p7: Amplitude, log-Width for Center 3
    p8, p9: Amplitude, log-Width for Center 4
    """
    # Extract parameters
    a0, lw0 = params[0], params[1]
    a1, lw1 = params[2], params[3]
    a2, lw2 = params[4], params[5]
    a3, lw3 = params[6], params[7]
    a4, lw4 = params[8], params[9]
    
    # Convert log-widths to widths to ensure positivity
    w0 = np.exp(lw0)
    w1 = np.exp(lw1)
    w2 = np.exp(lw2)
    w3 = np.exp(lw3)
    w4 = np.exp(lw4)
    
    # Define centers
    cx0, cy0, cz0 = 0.5, 0.5, 0.5
    cx1, cy1, cz1 = 2.0, 0.5, 0.5
    cx2, cy2, cz2 = 0.5, 2.0, 0.5
    cx3, cy3, cz3 = 0.5, 0.5, 2.0
    cx4, cy4, cz4 = 1.25, 1.25, 1.25
    
    # Calculate squared distances
    d0 = (x - cx0)**2 + (y - cy0)**2 + (z - cz0)**2
    d1 = (x - cx1)**2 + (y - cy1)**2 + (z - cz1)**2
    d2 = (x - cx2)**2 + (y - cy2)**2 + (z - cz2)**2
    d3 = (x - cx3)**2 + (y - cy3)**2 + (z - cz3)**2
    d4 = (x - cx4)**2 + (y - cy4)**2 + (z - cz4)**2
    
    # Compute RBF contributions
    # Using exp(-w * d)
    # Clip arguments to prevent overflow/underflow issues in exp, though exp(-large) is safe (0)
    # exp(large) is dangerous, but here argument is -w*d, which is negative or zero.
    # So exp(-w*d) is in (0, 1]. Safe.
    
    rbf0 = a0 * np.exp(-w0 * d0)
    rbf1 = a1 * np.exp(-w1 * d1)
    rbf2 = a2 * np.exp(-w2 * d2)
    rbf3 = a3 * np.exp(-w3 * d3)
    rbf4 = a4 * np.exp(-w4 * d4)
    
    pred = rbf0 + rbf1 + rbf2 + rbf3 + rbf4
    
    return pred
# EVOLVE-BLOCK-END
