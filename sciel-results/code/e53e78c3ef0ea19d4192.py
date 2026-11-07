# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Structural change: Non-parametric local basis expansion with learnable centers.
    
    The previous program used a global linear basis (Chebyshev + fixed trig/Gaussian).
    This approach uses a set of localized RBF-like bumps (Gaussians) with centers
    spread across the domain. The parameters control the amplitude and width of these
    bumps. This allows the model to fit complex, non-smooth, or highly oscillatory
    functions by effectively creating a local adaptive basis.
    
    To handle the 10-parameter constraint effectively:
    - We define 5 Gaussian bumps.
    - Each bump has an amplitude and a width parameter.
    - The centers are fixed at evenly spaced locations in [-3, 3] to ensure global coverage
      and avoid the optimizer getting stuck in local minima regarding center placement,
      which is often difficult with BFGS from a single start point.
    - The widths are parameterized via log-scale to ensure positivity and better conditioning.
    
    Parameters:
    p0: Amplitude of bump 0
    p1: Width of bump 0 (log-scaled)
    p2: Amplitude of bump 1
    p3: Width of bump 1 (log-scaled)
    p4: Amplitude of bump 2
    p5: Width of bump 2 (log-scaled)
    p6: Amplitude of bump 3
    p7: Width of bump 3 (log-scaled)
    p8: Amplitude of bump 4
    p9: Width of bump 4 (log-scaled)
    
    Additionally, we can include a global offset or linear trend if needed, but with only 10 params,
    5 bumps (10 params) is a tight fit. Let's stick to 5 bumps.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Fixed centers evenly spaced in [-3, 3]
    # 5 points: -3, -1.5, 0, 1.5, 3
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    # Initialize output
    y = np.zeros_like(x)
    
    # Define widths from params (log-scale to ensure positive width)
    # width = exp(p) / 2.0 ? or just exp(p)?
    # Let's use width = np.exp(params[1::2])
    # This ensures width > 0.
    
    amps = params[0::2]  # p0, p2, p4, p6, p8
    widths = np.exp(params[1::2])  # p1, p3, p5, p7, p9
    
    # Compute Gaussian bumps
    # To avoid numerical issues with very small widths (exp underflow) or large widths,
    # we compute the exponent carefully.
    # Gaussian: amp * exp(-0.5 * ((x - center) / width)^2)
    
    for i in range(5):
        c = centers[i]
        w = widths[i]
        a = amps[i]
        
        # Calculate squared distance
        diff = (x - c)
        # Normalize by width
        # If w is very small, diff/w can be huge. exp(-huge) -> 0. This is fine.
        # If w is very large, diff/w is small. exp(-small) -> ~1. This is fine.
        
        # Use a stable computation if possible, but standard exp is usually fine for float64
        # unless values are extreme.
        z = diff / w
        y += a * np.exp(-0.5 * z * z)
        
    return y
# EVOLVE-BLOCK-END
