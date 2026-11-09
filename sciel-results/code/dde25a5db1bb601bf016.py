"""Bootstrap portfolio: model-and-optimize for stress-strain of aluminium rod."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Record the start time to implement a deadline guard.
_START_TIME = time.time()
_TIMEOUT_LIMIT = 140.0

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.
    
    Structural change: Use a purely piecewise-linear basis with 4 kinks
    to capture yield points and work-hardening segments. This removes the
    rational term which may be interfering with the kink fitting, and
    allocates parameters to more breakpoints for better resolution.
    
    Model:
    stress = C(T) + m1*strain 
           + A1 * max(0, strain - b1) 
           + A2 * max(0, strain - b2) 
           + A3 * max(0, strain - b3)
           + A4 * max(0, strain - b4)
    
    Where:
    C(T) = p0 + p1*temp  (2 params for intercept)
    m1   = p2            (1 param for initial slope/elastic modulus)
    b1, b2, b3, b4 = p3, p4, p5, p6  (4 params for breakpoints)
    A1, A2, A3, A4 = p7, p8, p9, p10? No, only 10 params total.
    
    Let's count: p0..p9 is 10 params.
    C(T) = p0 + p1*temp (2)
    m1   = p2 (1)
    b1, b2, b3 = p3, p4, p5 (3)
    A1, A2, A3 = p6, p7, p8 (3)
    Total = 2 + 1 + 3 + 3 = 9. We have 1 param left.
    
    Let's use 4 kinks and 1 param for something else?
    Or 3 kinks and 2 params for C(T)?
    
    Let's try:
    C(T) = p0 + p1*temp (2)
    m1   = p2 (1)
    b1, b2, b3, b4 = p3, p4, p5, p6 (4)
    A1, A2, A3 = p7, p8, p9 (3)
    Total = 2 + 1 + 4 + 3 = 10.
    This gives 4 breakpoints but only 3 slope changes. The 4th breakpoint is unused?
    No, I need A4 for b4.
    
    Alternative:
    C(T) = p0 (1) -- constant offset
    m1(T) = p1 + p2*temp (2) -- elastic modulus with temp dependence
    b1, b2, b3, b4 = p3, p4, p5, p6 (4)
    A1, A2, A3, A4 = p7, p8, p9 (3) -- still short one.
    
    Let's go back to 3 kinks but better utilization.
    C(T) = p0 + p1*temp (2)
    m1   = p2 (1)
    b1, b2, b3 = p3, p4, p5 (3)
    A1, A2, A3 = p6, p7, p8 (3)
    p9 is left over. Use it for a second temp term in C(T)?
    C(T) = p0 + p1*temp + p9*temp**2 (3)
    Total = 3 + 1 + 3 + 3 = 10.
    
    This model:
    stress = (p0 + p1*T + p9*T^2) + p2*strain 
           + p6 * max(0, strain - p3)
           + p7 * max(0, strain - p4)
           + p8 * max(0, strain - p5)
    
    This is a pure piecewise linear function of strain, with temperature-dependent
    intercept. It should be very easy for BFGS to fit since it's linear in params
    if breakpoints are fixed, but breakpoints are params too. The non-linearity
    is only in the breakpoint positions.
    
    Let's implement this.
    """
    if time.time() - _START_TIME > _TIMEOUT_LIMIT:
        return np.zeros_like(strain, dtype=np.float64)

    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent intercept
    C_T = p0 + p1 * temp + p9 * temp**2
    
    # Linear elastic term
    term_linear = p2 * strain
    
    # Breakpoints
    b1 = p3
    b2 = p4
    b3 = p5
    
    # Slope changes
    A1 = p6
    A2 = p7
    A3 = p8
    
    # Piecewise-linear terms: A * max(0, strain - b)
    # max(0, x) = 0.5 * (x + abs(x))
    term1 = A1 * 0.5 * ((strain - b1) + np.abs(strain - b1))
    term2 = A2 * 0.5 * ((strain - b2) + np.abs(strain - b2))
    term3 = A3 * 0.5 * ((strain - b3) + np.abs(strain - b3))
    
    stress = C_T + term_linear + term1 + term2 + term3
    
    # Ensure output is finite
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
# EVOLVE-BLOCK-END
