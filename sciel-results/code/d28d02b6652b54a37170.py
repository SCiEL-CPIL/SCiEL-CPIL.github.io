import numpy as np
import time

def equation(strain, temp, params):
    """
    Strategy: Linear-in-Parameters Basis Expansion.
    
    Instead of a nonlinear rational function where parameters appear in the 
    denominator (creating a non-convex landscape), we use a linear combination 
    of physically motivated basis functions. The parameters are the weights 
    of these basis functions, which are linear in temperature.
    
    This transforms the optimization problem into a convex least-squares 
    problem, which BFGS solves robustly and quickly from any starting point.
    
    Basis functions:
    1. Constant (offset)
    2. Linear strain (elastic)
    3. Quadratic strain (hardening)
    4. Cubic strain (hardening)
    5. Tanh yield transition (smooth kink)
    
    Parameters p[0..9] define the coefficients for these 5 bases, each 
    linear in temperature: c_i(t) = p[2i] + p[2i+1]*t.
    """
    s = strain
    t = temp
    p = params
    
    # Check for non-finite inputs/params to avoid crashes
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(s)) or not np.all(np.isfinite(t)):
        return np.zeros_like(s, dtype=float)
        
    # Construct basis functions
    # 1. Offset
    phi1 = np.ones_like(s, dtype=float)
    
    # 2. Linear
    phi2 = s
    
    # 3. Quadratic
    phi3 = s * s
    
    # 4. Cubic
    phi4 = s * s * s
    
    # 5. Yield transition (smooth step)
    # Using a steep tanh to approximate a yield point around s=0.01
    # The center 0.01 is a guess; the linear combination can shift the effective effect
    # if we had more params, but with 5 bases, we rely on the weight.
    # To make it more flexible, we could use s directly in the tanh argument scaled by a param?
    # No, that would make it nonlinear in params.
    # We stick to fixed basis to maintain linearity.
    # Let's use a slightly wider tanh to capture the transition region better.
    phi5 = np.tanh(50.0 * (s - 0.01))
    
    # Calculate weights (linear in temp)
    w1 = p[0] + p[1] * t
    w2 = p[2] + p[3] * t
    w3 = p[4] + p[5] * t
    w4 = p[6] + p[7] * t
    w5 = p[8] + p[9] * t
    
    # Combine
    stress = w1 * phi1 + w2 * phi2 + w3 * phi3 + w4 * phi4 + w5 * phi5
    
    return stress
