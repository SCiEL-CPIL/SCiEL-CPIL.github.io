import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Combined RBF + Legendre polynomial model
    # 5 Gaussian RBF centers + 5 Legendre polynomial terms
    # Centers: (-1,-1), (1,-1), (-1,1), (1,1), (0,0)
    
    cx = np.array([-1.0, 1.0, -1.0, 1.0, 0.0])
    cy = np.array([-1.0, -1.0, 1.0, 1.0, 0.0])
    
    # Gaussian width parameter (fixed for stability)
    sigma = 0.8
    
    # Compute Gaussian kernels for each center
    b_gauss = np.zeros_like(x)
    for i in range(5):
        dx = x - cx[i]
        dy = y - cy[i]
        dist_sq = dx**2 + dy**2
        b_gauss += params[i] * np.exp(-dist_sq / (2.0 * sigma**2))
    
    # Legendre polynomial terms (5 terms selected for orthogonality and coverage)
    # P_0(t) = 1
    # P_1(t) = t
    # P_2(t) = 0.5 * (3*t**2 - 1)
    # P_3(t) = 0.5 * (5*t**3 - 3*t)
    
    px0 = np.ones_like(x)
    px1 = x
    px2 = 0.5 * (3.0 * x**2 - 1.0)
    px3 = 0.5 * (5.0 * x**3 - 3.0 * x)
    
    py0 = np.ones_like(y)
    py1 = y
    py2 = 0.5 * (3.0 * y**2 - 1.0)
    py3 = 0.5 * (5.0 * y**3 - 3.0 * y)
    
    # 5 Legendre-based terms: (0,0), (1,0), (0,1), (1,1), (2,2)
    # This gives constant, linear x, linear y, interaction, and quadratic interaction
    b_legendre = (params[5] * px0 * py0 +
                  params[6] * px1 * py0 +
                  params[7] * px0 * py1 +
                  params[8] * px1 * py1 +
                  params[9] * px2 * py2)
    
    pred = b_gauss + b_legendre
    
    return pred
