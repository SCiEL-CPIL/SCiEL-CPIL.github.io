import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Fourier series basis to model periodic
    phase-space dynamics of the oscillator.
    
    Basis terms:
    1. 1 (bias)
    2. sin(x)
    3. cos(x)
    4. sin(v)
    5. cos(v)
    6. sin(x+v)
    7. cos(x+v)
    8. sin(x-v)
    9. cos(x-v)
    10. sin(2x)
    
    This basis explicitly models harmonic and inter-harmonic couplings
    that are natural for bounded oscillator dynamics in phase space.
    """
    # Time guard
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s limit
    
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute Fourier basis terms directly on original inputs
    # The inputs are already bounded, so trig functions will be well-behaved
    # x range: [-0.5728, 0.5275], v range: [-0.1754, 0.2821]
    # These are small angles, so sin/cos are smooth and well-conditioned
    
    b0 = np.ones_like(x)
    b1 = np.sin(x)
    b2 = np.cos(x)
    b3 = np.sin(v)
    b4 = np.cos(v)
    b5 = np.sin(x + v)
    b6 = np.cos(x + v)
    b7 = np.sin(x - v)
    b8 = np.cos(x - v)
    b9 = np.sin(2 * x)
    
    # Combine linearly
    result = (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4 +
              p5 * b5 + p6 * b6 + p7 * b7 + p8 * b8 + p9 * b9)
    
    # Time check - for safety
    if time.time() - start_time > timeout:
        pass
    
    return result
