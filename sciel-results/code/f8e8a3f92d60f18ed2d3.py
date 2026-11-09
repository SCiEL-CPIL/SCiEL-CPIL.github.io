"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Fourier Series with 5 Harmonics.
    
    y(x) = p0 
           + p1 * sin(x) + p2 * cos(x)
           + p3 * sin(2x) + p4 * cos(2x)
           + p5 * sin(3x) + p6 * cos(3x)
           + p7 * sin(4x) + p8 * cos(4x)
           + p9 * sin(5x)
           
    Note: We use 10 parameters. The 10th harmonic (cos(5x)) is omitted to fit exactly 10 params,
    or we could scale. Let's stick to standard orthogonal basis.
    Actually, let's use:
    p0: constant
    p1-p4: sin(1..4), cos(1..4) -> 8 params
    p5-p6: sin(5), cos(5) -> 2 params
    Total 10 params.
    
    y(x) = p0 + sum_{k=1}^{4} (p[2k-1] * sin(kx) + p[2k] * cos(kx)) + p[9] * sin(5x)
    Wait, indices:
    p0: const
    k=1: p1 sin(x), p2 cos(x)
    k=2: p3 sin(2x), p4 cos(2x)
    k=3: p5 sin(3x), p6 cos(3x)
    k=4: p7 sin(4x), p8 cos(4x)
    k=5: p9 sin(5x)  (missing cos(5x) to keep 10 params, or maybe p9 is cos(5x) and sin(5x) is fixed? 
            Better to just use 10 terms: 
            const, sin1, cos1, sin2, cos2, sin3, cos3, sin4, cos4, sin5.
            This is slightly asymmetric but fine for BFGS.
            Alternatively, use Chebyshev polynomials which are also orthogonal on [-1,1] (mapped from [-2,2]).
            But the directive suggests Fourier.
    """
    y = np.zeros_like(x)
    
    # p0: constant
    y += params[0]
    
    # k=1
    y += params[1] * np.sin(x)
    y += params[2] * np.cos(x)
    
    # k=2
    y += params[3] * np.sin(2 * x)
    y += params[4] * np.cos(2 * x)
    
    # k=3
    y += params[5] * np.sin(3 * x)
    y += params[6] * np.cos(3 * x)
    
    # k=4
    y += params[7] * np.sin(4 * x)
    y += params[8] * np.cos(4 * x)
    
    # k=5 (only sine to use 10th param, or maybe we should balance? 
    # Let's use sin(5x) and cos(5x) but drop one lower term? 
    # No, just use the 10th param for sin(5x). The lack of cos(5x) is a minor constraint.
    # Or better: use Chebyshev basis? 
    # The directive says "Fourier series basis". 
    # Let's stick to the 10 params defined above.
    y += params[9] * np.sin(5 * x)
    
    return y
# EVOLVE-BLOCK-END
