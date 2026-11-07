import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod using a piecewise smooth hinge model.
    
    Structure:
    - Elastic regime: linear in strain with slope E(T)
    - Plastic regime: smooth transition via s*(1-exp(-k*s)) to capture yield plateau
    - Temperature dependence: linear modulation of key parameters
    
    10 params:
      p0: Base elastic modulus E0
      p1: Temp coeff for E
      p2: Yield stress base Y0
      p3: Temp coeff for Y
      p4: Yield strain threshold eps_y
      p5: Hardening modulus H
      p6: Temp coeff for H
      p7: Hinge sharpness k (controls smoothness of transition)
      p8: Offset (intercept)
      p9: Temp coeff for offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-modulated parameters
    E = p0 + p1 * temp          # Elastic modulus
    Y = p2 + p3 * temp          # Yield stress
    eps_y = p4                  # Yield strain threshold
    H = p5 + p6 * temp          # Hardening modulus
    k = np.abs(p7) + 1.0        # Hinge sharpness (ensure positive)
    offset = p8 + p9 * temp     # Intercept
    
    # Smooth hinge function: s * (1 - exp(-k*s))
    # For s >> 0, this approaches s; for s << 0, this approaches 0
    # This creates a smooth transition from elastic to plastic
    
    # Plastic strain component (smooth hinge)
    plastic_strain = strain * (1.0 - np.exp(-k * strain))
    
    # Elastic contribution
    elastic_stress = E * strain
    
    # Plastic contribution: yield stress + hardening
    # The hinge function naturally limits the elastic-like behavior
    # and adds hardening beyond yield
    plastic_stress = Y * np.tanh(strain / (eps_y + 1e-8)) + H * plastic_strain
    
    # Combine: elastic dominates at low strain, plastic at high strain
    # Use a smooth blending based on strain magnitude
    blend = 1.0 / (1.0 + np.exp(-k * (strain - eps_y)))
    
    stress = offset + blend * plastic_stress + (1.0 - blend) * elastic_stress
    
    return stress
