"""
Hybrid Basis Approximation: Chebyshev Polynomials + Fixed Gaussians.

This program uses a linear combination of Chebyshev polynomials and fixed-location
Gaussian functions to approximate the target function. 
1. Chebyshev polynomials provide a well-conditioned basis for smooth trends on [-1, 1].
2. Fixed Gaussians provide local flexibility to capture bumps or sharp features.
3. The model is linear in the parameters, ensuring that scipy.optimize.minimize (BFGS)
   will converge to the global minimum of the MSE quickly and reliably, avoiding
   the local minima and divergence issues common with rational functions.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates y using a hybrid basis:
    y(x) = sum_{k=0}^5 p_k * T_k(x/2) + sum_{j=1}^4 g_j * exp(-(x - c_j)^2 / (2*sigma^2))
    
    Parameters:
    - x: Input array
    - params: Array of 10 parameters.
      - params[0:6] are coefficients for Chebyshev polynomials T_0 to T_5.
      - params[6:10] are amplitudes for 4 fixed Gaussians.
    """
    x = np.asarray(x, dtype=float)
    
    # 1. Chebyshev Part
    # Map x from [-2, 2] to t in [-1, 1]
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_5
    # T_0 = 1
    # T_1 = t
    # T_2 = 2t^2 - 1
    # T_3 = 4t^3 - 3t
    # T_4 = 8t^4 - 8t^2 + 1
    # T_5 = 16t^5 - 20t^3 + 5t
    
    t2 = t * t
    t3 = t2 * t
    t4 = t2 * t2
    t5 = t4 * t
    
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t2 - 1.0
    T3 = 4.0 * t3 - 3.0 * t
    T4 = 8.0 * t4 - 8.0 * t2 + 1.0
    T5 = 16.0 * t5 - 20.0 * t3 + 5.0
    
    # Extract coefficients
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    
    cheb_part = (p0 * T0 + p1 * T1 + p2 * T2 + p3 * T3 + p4 * T4 + p5 * T5)
    
    # 2. Gaussian Part
    # Fixed centers and width to ensure locality and stability
    centers = np.array([-1.5, -0.5, 0.5, 1.5])
    sigma = 0.5
    sigma2 = 2.0 * sigma * sigma
    
    # Compute Gaussians: exp(-(x - c)^2 / sigma2)
    # We can do this efficiently by broadcasting
    # x is (N,), centers is (4,)
    # diff is (N, 4)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    gaussians = np.exp(-(diff ** 2) / sigma2)
    
    # Extract amplitudes
    g1 = params[6]
    g2 = params[7]
    g3 = params[8]
    g4 = params[9]
    
    # Combine Gaussian parts
    gauss_part = (g1 * gaussians[:, 0] + 
                  g2 * gaussians[:, 1] + 
                  g3 * gaussians[:, 2] + 
                  g4 * gaussians[:, 3])
    
    # Total prediction
    result = cheb_part + gauss_part
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
