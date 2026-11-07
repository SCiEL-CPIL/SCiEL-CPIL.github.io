import numpy as np

def equation(strain, temp, params):
    """
    Symbolic regression for stress-strain of an aluminium rod.
    
    Strategy: A linear combination of 10 carefully chosen basis functions.
    This makes the model linear in the parameters `p`, which allows
    scipy.optimize.minimize (BFGS) to converge extremely quickly and
    reliably to the global minimum, avoiding the local minima traps
    inherent in non-linear parameterizations like tanh switches.
    
    The basis functions are designed to capture:
    1. Elastic behavior (linear in strain, modulated by temp).
    2. Yield transition (smooth step at strain ~0.001).
    3. Plastic hardening (linear and quadratic in excess strain).
    4. Temperature effects (linear and interaction terms).
    """
    s = strain
    t = temp
    p = params
    
    # --- Basis Functions ---
    
    # 1. Constant offset (residual stress or baseline)
    phi0 = np.ones_like(s)
    
    # 2. Temperature linear term (thermal expansion/stress effect)
    phi1 = t
    
    # 3. Strain linear term (elastic modulus, constant part)
    phi2 = s
    
    # 4. Strain-temperature interaction (modulus decrease with temp)
    phi3 = s * t
    
    # 5. Yield step function (smooth tanh approximation of Heaviside at yield strain)
    #    Yield strain for Al is typically around 0.001-0.01. We use a fixed center
    #    but let the weight adjust. The sharpness is fixed to avoid local minima
    #    in the "width" parameter.
    yield_strain = 0.0015
    width = 500.0
    phi4 = np.tanh(width * (s - yield_strain))
    
    # 6. Plastic linear hardening (active only after yield)
    #    Using max(s - yield_strain, 0) ensures it's zero in elastic region.
    #    BFGS handles kinks well in practice for this type of objective.
    excess_strain = np.maximum(s - yield_strain, 0.0)
    phi5 = excess_strain
    
    # 7. Plastic quadratic hardening
    phi6 = excess_strain * excess_strain
    
    # 8. Temperature-modulated yield step
    #    Allows the yield point to shift with temperature
    yield_shift = -0.0005 * t  # Yield drops with temp
    phi7 = np.tanh(width * (s - (yield_strain + yield_shift)))
    
    # 9. Temperature-modulated plastic hardening
    #    Hardening often decreases with temperature
    phi8 = excess_strain * t
    
    # 10. Elastic quadratic correction (small non-linearity in elastic regime)
    phi9 = s * s * np.maximum(1.0 - width*(s-yield_strain), 0.0) # Roughly elastic only
    
    # Combine linearly
    # sigma = sum(p_i * phi_i)
    
    # To ensure stability and avoid huge values if p explodes,
    # we can clip the inputs to reasonable ranges if necessary, 
    # but the problem states inputs are in range.
    
    pred = (p[0] * phi0 +
            p[1] * phi1 +
            p[2] * phi2 +
            p[3] * phi3 +
            p[4] * phi4 +
            p[5] * phi5 +
            p[6] * phi6 +
            p[7] * phi7 +
            p[8] * phi8 +
            p[9] * phi9)
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
