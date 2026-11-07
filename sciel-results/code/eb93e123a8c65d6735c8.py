import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a closed-form expression with 10 parameters to model elastic, 
    plastic hardening, and thermal effects.
    """
    start_time = time.time()
    
    # Safety guard: If inputs are malformed or too large, return zeros
    if not np.all(np.isfinite(strain)) or not np.all(np.isfinite(temp)):
        return np.zeros_like(strain)
        
    e = strain
    t = temp
    
    # Compute basis functions
    # e^2 and e^3 can be large, but strain is in [-0.002, 0.96]
    e2 = e * e
    e3 = e2 * e
    
    # Denominator for rational functions
    denom = 1.0 + e2
    
    # r1: e / (1 + e^2) -> ~e for small e, ~1/e for large e (bounded derivative)
    r1 = e / denom
    # r2: e^3 / (1 + e^2) -> ~e^3 for small e, ~e for large e
    r2 = e3 / denom
    
    # Additional basis functions for better plastic region modeling
    # tanh-based saturation for plastic flow
    # We use a scaled tanh to capture the yield plateau and hardening
    # scale factor ~ 10 to make the transition sharp around e~0.01-0.05
    sat1 = np.tanh(10.0 * e)
    # Higher order saturation
    sat2 = np.tanh(5.0 * e)
    
    # Temperature dependent effects
    # t is in [0.066, 1]
    # Thermal softening often follows an exponential or polynomial decay
    # Let's use a simple polynomial in t for flexibility
    
    stress = (params[0] * 1.0 +
              params[1] * e +
              params[2] * r1 +
              params[3] * r2 +
              params[4] * sat1 +
              params[5] * sat2 +
              params[6] * (t * e) +
              params[7] * (t * r1) +
              params[8] * (t * sat1) +
              params[9] * (t * t))
    
    # Check for non-finite results
    if not np.all(np.isfinite(stress)):
        stress = np.nan_to_num(stress, nan=0.0, posinf=0.0, neginf=0.0)
        
    # Deadline guard check
    elapsed = time.time() - start_time
    if elapsed > 140.0:
        # In a search context, this would break. Here, it's just a warning log conceptually.
        # We return the best we have.
        pass
        
    return stress
