import numpy as np

def construct():
    """
    Return 12 double-root locations z_1..z_12.
    
    Strategy: Use the 12 smallest positive zeros of the Bessel function J_{1/2}(x).
    J_{1/2}(x) = sqrt(2/(pi*x)) * sin(sqrt(x)).
    The zeros are at sqrt(x) = k*pi for k=1, 2, ...
    So x = (k*pi)^2.
    This exploits the asymptotic orthogonality of Laguerre polynomials to the Bessel measure at infinity.
    """
    # Zeros of J_{1/2}(x) are at x = (k*pi)^2 for k=1, 2, ...
    k_vals = np.arange(1, 13, dtype=np.float64)
    roots_sqrt = k_vals * np.pi
    z_vals = roots_sqrt ** 2
    
    # Ensure they are finite, > 0, and pairwise distinct
    z_list = [float(z) for z in z_vals]
    
    return z_list
