"""Smoothstep-based piecewise polynomial model for aluminium rod stress-strain."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress = E_eff * strain * (1 - S1) + F1_eff * S1 * (1 - S2) + F2_eff * S2
    
    This model uses smoothstep basis functions (S(x) = 3x^2 - 2x^3) to create a 
    C1-continuous transition between three distinct regimes:
    1. Elastic: Linear response E*strain.
    2. Yield/Plateau: Constant stress F1 (approximating the yield plateau).
    3. Hardening/Saturation: Constant stress F2 (approximating ultimate strength).
    
    The smoothstep function allows for a sharp but differentiable transition, 
    which is better suited for BFGS optimization than hard kinks or rational functions.
    
    Parameters:
    p0, p1: E_eff = p0 + p1*temp (Elastic modulus, temperature-dependent)
    p2, p3: F1_eff = p2 + p3*temp (Yield stress, temperature-dependent)
    p4, p5: F2_eff = p4 + p5*temp (Ultimate stress, temperature-dependent)
    p6, p7: y1 = 0.002 * exp(p6) + 0.001 * p7 * temp (Yield strain transition center)
    p8, p9: y2 = 0.05 * exp(p8) + 0.01 * p9 * temp (Hardening saturation strain center)
    
    For params=1 (BFGS start):
    y1 = 0.002*2.718 + 0.001*1 ≈ 0.0064
    y2 = 0.05*2.718 + 0.01*1 ≈ 0.1459
    These are reasonable starting points for aluminium yield/hardening strains.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent coefficients
    # Elastic Modulus
    E = p0 + p1 * temp
    # Yield Stress (Plateau level)
    F1 = p2 + p3 * temp
    # Ultimate Stress (Saturation level)
    F2 = p4 + p5 * temp
    
    # Transition Centers (Strain values)
    # Using exp to ensure positive scale factors if needed, but here we just use them as offsets/scales
    # To keep values in a reasonable range for BFGS starting at 1:
    # y1 center
    y1 = 0.002 * np.exp(p6) + 0.001 * p7 * temp
    # y2 center
    y2 = 0.05 * np.exp(p8) + 0.01 * p9 * temp
    
    # Smoothstep Transition Widths
    # Fixed widths to avoid adding more parameters, scaled relative to the strain range
    # Strain range is ~1.0. 
    # Width for yield transition should be small (sharp)
    w1 = 0.005
    # Width for hardening transition should be larger (gradual)
    w2 = 0.05
    
    # Compute normalized arguments for smoothstep
    # t1 = (strain - (y1 - w1)) / (2*w1)
    # We want S1 to be 0 when strain < y1-w1, 1 when strain > y1+w1
    # Standard smoothstep: s(t) = 3t^2 - 2t^3 for t in [0,1]
    
    t1 = (strain - (y1 - w1)) / (2.0 * w1)
    t2 = (strain - (y2 - w2)) / (2.0 * w2)
    
    # Clip to [0, 1] to ensure step behavior outside the transition zone
    t1 = np.clip(t1, 0.0, 1.0)
    t2 = np.clip(t2, 0.0, 1.0)
    
    # Smoothstep functions
    S1 = 3.0 * t1**2 - 2.0 * t1**3
    S2 = 3.0 * t2**2 - 2.0 * t2**3
    
    # Construct Stress
    # Regime 1: Elastic (weight 1 - S1)
    # Regime 2: Yield Plateau (weight S1 * (1 - S2))
    # Regime 3: Saturation (weight S2)
    
    # Note: The elastic term is E*strain. 
    # The yield term is F1.
    # The saturation term is F2.
    
    stress = E * strain * (1.0 - S1) + F1 * S1 * (1.0 - S2) + F2 * S2
    
    return stress
# EVOLVE-BLOCK-END
