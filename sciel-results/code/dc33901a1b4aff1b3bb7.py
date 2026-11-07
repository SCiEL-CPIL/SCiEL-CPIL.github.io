import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model with 10 parameters.
    
    Structure:
    V_max * Monod(s) * TempResponse * pHResponse * Inhibition(b) + Bias
    
    TempResponse = 1 + a1*sin(t) + a2*cos(t) + a3*t
    pHResponse = 1 + b1*sin(p) + b2*cos(p) + b3*p
    
    Parameters:
    p0: log(K_s)
    p1: log(K_b)
    p2: log(V_max)
    p3: bias
    p4: Temp Sin coef
    p5: Temp Cos coef
    p6: Temp Linear coef
    p7: pH Sin coef
    p8: pH Cos coef
    p9: pH Linear coef
    """
    eps = 1e-6
    
    # Extract parameters
    # Use clipping to prevent overflow in exp if params drift, though BFGS usually stays bounded
    # but starting from 1.0, exp(1) is safe.
    K_s = np.exp(np.clip(params[0], -20, 20)) + eps
    K_b = np.exp(np.clip(params[1], -20, 20)) + eps
    V_max = np.exp(np.clip(params[2], -20, 20)) + eps
    bias = params[3]
    
    # Monod
    monod = s / (K_s + s)
    
    # Density Inhibition
    inhib_b = K_b / (K_b + b)
    
    # Temperature Response
    # Scale temp to have period ~10-20 C? 
    # Let's use a standard frequency. 
    # t_scaled = (temp - 37.0)
    # We want sin/cos to capture the bell shape.
    t_scaled = (temp - 37.0)
    t_sin = np.sin(t_scaled)
    t_cos = np.cos(t_scaled)
    # Linear term allows shifting the center if the pure periodic part isn't enough
    temp_factor = 1.0 + params[4] * t_sin + params[5] * t_cos + params[6] * t_scaled
    
    # pH Response
    p_scaled = (pH - 7.0)
    p_sin = np.sin(p_scaled)
    p_cos = np.cos(p_scaled)
    ph_factor = 1.0 + params[7] * p_sin + params[8] * p_cos + params[9] * p_scaled
    
    # Core
    core = V_max * monod * temp_factor * ph_factor * inhib_b
    
    result = core + bias
    
    # Ensure finite
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
