import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Non-linear physics-inspired model (Van der Pol / Duffing hybrid).
    
    Instead of a linear combination of basis functions (y = X @ p), we use a 
    closed-form expression where parameters enter non-linearly. This changes the 
    optimization landscape from convex (linear least squares) to non-convex, 
    which may allow BFGS to find a better fit if the true dynamics are closer 
    to this functional form.
    
    Model:
    y = p0 * x * exp(-p1 * v**2) + p2 * v * (1 + p3 * x**2) + p4 * x**3 + p5 * v**3
      + p6 * x * v * (1 + p7 * x**2) + p8 * x * v**2 + p9 * v * x**2
      
    We scale inputs to O(1) to keep parameters near 1.0 (initial guess).
    x in [-0.57, 0.53] -> x_s = x / 0.6
    v in [-0.18, 0.28] -> v_s = v / 0.3
    
    The target y is in [-0.1, 0.1].
    If we use scaled inputs, the coefficients should be O(0.1-1.0).
    """
    # Time guard
    start_time = time.time()
    
    # Scale inputs to roughly [-1, 1] to improve conditioning
    x_s = x / 0.6
    v_s = v / 0.3
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Non-linear terms
    # Damping term: x * exp(-p1 * v^2)
    # This captures velocity-dependent damping of the restoring force.
    term1 = p0 * x_s * np.exp(-p1 * v_s**2)
    
    # Velocity term with position-dependent gain: v * (1 + p3 * x^2)
    term2 = p2 * v_s * (1.0 + p3 * x_s**2)
    
    # Non-linear restoring force (Duffing): x^3
    term3 = p4 * x_s**3
    
    # Non-linear damping: v^3
    term4 = p5 * v_s**3
    
    # Cross term with position modulation: x * v * (1 + p7 * x^2)
    term5 = p6 * x_s * v_s * (1.0 + p7 * x_s**2)
    
    # Higher order cross terms
    term6 = p8 * x_s * v_s**2
    term7 = p9 * v_s * x_s**2
    
    # Sum
    y_pred = term1 + term2 + term3 + term4 + term5 + term6 + term7
    
    # Ensure finite
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Time check (should be very fast, but just in case)
    if time.time() - start_time > 140:
         return np.zeros_like(x)
         
    return y_pred
