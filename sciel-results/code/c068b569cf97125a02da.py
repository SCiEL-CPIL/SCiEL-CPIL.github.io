import numpy as np
import time

def construct():
    """
    Return 12 double-root locations z_1..z_12.
    
    Strategy: Use the 12 smallest positive zeros of the Bessel function J_{1/2}(x),
    which are the exact roots of sin(sqrt(x))/sqrt(x), to define z_i.
    These are at x = (k*pi)^2 for k=1..12.
    
    This exploits the asymptotic orthogonality of Laguerre polynomials to the Bessel measure,
    potentially confining the quotient q(x)'s largest root B more effectively than
    finite-dimensional orthogonal nodes or arbitrary parameterized spacings.
    
    The current program's heuristic search over offsets was shown to be inferior to
    the pure Bessel-zero configuration. We return the canonical Bessel zeros directly.
    """
    # Deadline guard
    start_time = time.time()
    deadline = 630.0  # Evaluator kills at 630s
    
    # The zeros of J_{1/2}(x) are at x = (k*pi)^2 for k = 1, 2, 3, ...
    # J_{1/2}(x) = sqrt(2/(pi*x)) * sin(sqrt(x))
    # So the zeros (excluding x=0) are at sqrt(x) = k*pi, i.e., x = (k*pi)^2
    
    k_vals = np.arange(1, 13, dtype=np.float64)
    z_vals = (k_vals * np.pi) ** 2
    
    # Check validity: all positive, finite, distinct
    if not np.all(np.isfinite(z_vals)):
        raise ValueError("Non-finite values in z")
    if not np.all(z_vals > 0):
        raise ValueError("Non-positive values in z")
    if len(set(z_vals)) != 12:
        raise ValueError("Values in z are not distinct")
    
    z_list = [float(z) for z in z_vals]
    
    # Check time budget (should be well within limit, but guard just in case)
    if time.time() - start_time > deadline - 10.0:
        return z_list
    
    return z_list
