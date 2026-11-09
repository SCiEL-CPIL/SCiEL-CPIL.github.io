"""Combined symbolic regression program: polynomial + Gaussian RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    A flexible model combining polynomial terms (degree 0-4) with Gaussian
    radial basis functions at fixed centers. This provides both global trend
    capture (polynomial) and local feature adaptation (Gaussians), which
    together can approximate a wide range of smooth functions well-conditioned
    for BFGS optimization.
    
    Basis: 1, x, x^2, x^3, x^4, G(x; -1.5), G(x; -0.5), G(x; 0.5), G(x; 1.5),
           sin(x)
    where G(x; c) = exp(-(x-c)^2 / (2*sigma^2)) with sigma=0.6
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has exactly 10 elements
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = p
    
    # Scaled Chebyshev polynomials for better conditioning on [-2, 2]
    u = 0.5 * x
    u2 = u * u
    u3 = u2 * u
    u4 = u2 * u2
    
    T0 = np.ones_like(x)
    T1 = u
    T2 = 2.0 * u2 - 1.0
    T3 = 4.0 * u3 - 3.0 * u
    T4 = 8.0 * u4 - 8.0 * u2 + 1.0
    
    # Gaussian RBF terms at strategic centers with moderate width
    sigma = 0.6
    sigma2 = 2.0 * sigma * sigma
    
    c1 = -1.5
    c2 = -0.5
    c3 = 0.5
    c4 = 1.5
    
    g1 = np.exp(-((x - c1) ** 2) / sigma2)
    g2 = np.exp(-((x - c2) ** 2) / sigma2)
    g3 = np.exp(-((x - c3) ** 2) / sigma2)
    g4 = np.exp(-((x - c4) ** 2) / sigma2)
    
    # Sinusoidal term for periodic features
    s = np.sin(x)
    
    result = (
        p0 * T0 +
        p1 * T1 +
        p2 * T2 +
        p3 * T3 +
        p4 * T4 +
        p5 * g1 +
        p6 * g2 +
        p7 * g3 +
        p8 * g4 +
        p9 * s
    )
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
