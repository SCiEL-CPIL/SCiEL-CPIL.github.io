import numpy as np
import time

def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell.
    Uses a superellipse (rounded square) dielectric rod in air.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    size = problem["pixels"]
    
    # Parameters for the superellipse
    # |x|^(2n) + |y|^(2n) <= R^(2n)
    n_exp = 4.0  # 2.0 is circle, 4.0 is rounded square, higher is squarer
    R = 0.20     # Radius parameter, similar to the inspiration
    
    cell = np.zeros((size, size), dtype=int)
    
    # Generate coordinates
    # Pixel centers are at (i + 0.5) / size - 0.5
    coords = np.arange(size)
    x = (coords + 0.5) / size - 0.5
    y = (coords + 0.5) / size - 0.5
    
    # Create meshgrid
    X, Y = np.meshgrid(x, y)
    
    # Calculate superellipse equation
    # To avoid numerical issues with large exponents, we can use abs
    # |X|^(2n) + |Y|^(2n) <= R^(2n)
    # This is equivalent to (|X|/R)^(2n) + (|Y|/R)^(2n) <= 1
    
    # If R is 0, no dielectric.
    if R > 0:
        # Using the normalized form is more stable
        X_norm = np.abs(X) / R
        Y_norm = np.abs(Y) / R
        
        # Calculate power
        # For n=4, power is 8.
        # We can compute this directly.
        val = X_norm ** (2 * n_exp) + Y_norm ** (2 * n_exp)
        
        # If val <= 1, it's inside the dielectric
        cell[val <= 1.0] = 1
        
    # Ensure symmetry (it should be inherently symmetric, but let's be safe)
    # The generation is symmetric by construction.
    
    # Check deadline (though this is fast)
    if time.time() > deadline:
        # Fallback to a simple circle if something went wrong (unlikely)
        # But we already have the cell.
        pass
        
    return cell.tolist()
