# EVOLVE-BLOCK-START
import numpy as np


def rudin_problem10() -> np.ndarray:
    """
    Returns the coefficients for the 'flattest' degree-10 spherical harmonic.
    
    Strategy:
    The L4/L2 ratio is minimized by spherical harmonics that are as spatially
    uniform as possible. The zonal harmonic (m=0) is highly peaked at the poles,
    resulting in a high L4/L2 ratio. The maximum-order harmonics (m = +/- L)
    are concentrated in a ring around the equator and are significantly flatter
    (more uniform in the azimuthal direction and less peaked in the polar direction
    compared to m=0). 
    
    For L=10, the harmonic with the highest order (m=10 or m=-10) provides a
    substantial improvement in the L4/L2 ratio over the all-ones vector (which
    includes the peaked m=0 component). 
    
    We return coefficients corresponding to a linear combination of the 
    maximum-order real harmonics (cos(10*phi) and sin(10*phi) components),
    effectively creating a ring-like distribution on the sphere.
    
    Coefficients are ordered m = -10, -9, ..., 0, ..., 9, 10.
    c[i] corresponds to m = i - 10.
    
    We set c[0] (m=-10) and c[20] (m=10) to 1.0, and all others to 0.0.
    """
    k = 10
    coeffs = np.zeros(2 * k + 1, dtype=np.float64)
    
    # Index for m = -10 is 0
    # Index for m = 10 is 20
    coeffs[0] = 1.0
    coeffs[20] = 1.0
    
    return coeffs


# EVOLVE-BLOCK-END
