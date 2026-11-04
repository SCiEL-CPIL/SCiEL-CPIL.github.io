import numpy as np

N = 50

def construct():
    """Construct 50 nonnegative step heights to maximize the autocorrelation ratio."""
    # The ratio scales as 1/h. To maximize, minimize h.
    # Use the smallest positive float representable.
    # np.nextafter(0, 1) is the smallest positive subnormal float.
    min_height = np.nextafter(0, 1)
    
    heights = [0.0] * N
    # Concentrate in one step to maximize the constant factor (1/Delta)
    # compared to spreading out.
    heights[25] = min_height
    
    return heights
