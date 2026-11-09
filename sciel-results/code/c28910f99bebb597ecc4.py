"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # 5 Polynomial terms (Quadratic without xy)
    # 1, x, y, x^2, y^2
    # 5 Gaussian terms (Centers at 0,0 and corners +/-0.5)
    
    # Polynomial part
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    
    # Gaussian part
    # Widths fixed at 5 (sigma^2 = 0.2, sigma ~ 0.45, covers local regions)
    # Center 0,0
    g0 = np.exp(-5.0 * (x**2 + y**2))
    # Center 0.5, 0.5
    g1 = np.exp(-5.0 * ((x - 0.5)**2 + (y - 0.5)**2))
    # Center -0.5, -0.5
    g2 = np.exp(-5.0 * ((x + 0.5)**2 + (y + 0.5)**2))
    # Center 0.5, -0.5
    g3 = np.exp(-5.0 * ((x - 0.5)**2 + (y + 0.5)**2))
    # Center -0.5, 0.5
    g4 = np.exp(-5.0 * ((x + 0.5)**2 + (y - 0.5)**2))
    
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    return (p0 + p1 * x + p2 * y + p3 * (x**2) + p4 * (y**2) + 
            p5 * g0 + p6 * g1 + p7 * g2 + p8 * g3 + p9 * g4)
# EVOLVE-BLOCK-END
