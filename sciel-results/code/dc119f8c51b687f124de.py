import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod using a dual-tanh formulation.
    
    The model decouples the sharp yield transition from gradual plastic hardening.
    - Elastic regime: linear in strain.
    - Yield transition: smooth step (tanh) from elastic to plastic.
    - Plastic hardening: gradual increase with strain, modulated by temperature.
    
    Parameters:
        p0: Elastic modulus scale
        p1: Yield strain location (offset)
        p2: Yield transition sharpness
        p3: Yield stress level
        p4: Hardening modulus scale
        p5: Hardening saturation strain
        p6: Temperature effect on yield stress
        p7: Temperature effect on hardening
        p8: Baseline stress offset
        p9: Temperature interaction term
    """
    eps = strain
    T = temp
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure positive scales for numerical stability
    # Use softplus-like: log(1+exp(x)) but simpler: x^2 + 0.1
    E_scale = np.abs(p0) + 0.1
    yield_strain = p1 + 0.001  # small offset to avoid singularity
    sharpness = np.abs(p2) + 1.0
    yield_stress = p3
    hard_mod = np.abs(p4) + 0.1
    sat_strain = np.abs(p5) + 0.01
    temp_yield = p6
    temp_hard = p7
    offset = p8
    temp_interact = p9
    
    # Normalize temperature to [0, 1] range approximately
    # Training temp range: [0.06667, 1]
    T_norm = T - 0.06667
    T_norm = T_norm / (1.0 - 0.06667 + 1e-9)
    
    # Temperature-adjusted yield stress
    # Higher temperature reduces yield stress
    yield_stress_T = yield_stress * (1.0 + temp_yield * T_norm)
    
    # Temperature-adjusted hardening
    hard_mod_T = hard_mod * (1.0 + temp_hard * T_norm)
    
    # Elastic stress
    elastic_stress = E_scale * eps
    
    # Yield transition: tanh-based smooth step
    # When eps < yield_strain, transition ~ 0 (elastic)
    # When eps > yield_strain, transition ~ 1 (plastic)
    transition_arg = sharpness * (eps - yield_strain)
    # Clip to avoid overflow in tanh
    transition_arg = np.clip(transition_arg, -50, 50)
    yield_transition = np.tanh(transition_arg)
    # Map from [-1, 1] to [0, 1]
    yield_fraction = 0.5 * (yield_transition + 1.0)
    
    # Plastic strain (beyond yield)
    plastic_strain = np.maximum(eps - yield_strain, 0.0)
    
    # Hardening: saturating function
    # hardening = hard_mod_T * (1 - exp(-plastic_strain / sat_strain))
    # Or simpler: hard_mod_T * plastic_strain / (plastic_strain + sat_strain)
    hardening = hard_mod_T * plastic_strain / (plastic_strain + sat_strain)
    
    # Total stress:
    # Elastic part is reduced as we enter plastic regime
    # Plastic part adds yield stress + hardening
    # stress = elastic_stress * (1 - yield_fraction) + (yield_stress_T + hardening) * yield_fraction
    # But this can be discontinuous in derivative. Let's use a smoother blend.
    
    # Alternative: stress = E_scale * eps * (1 - yield_fraction) + (yield_stress_T + hardening) * yield_fraction
    # At yield: eps = yield_strain, yield_fraction = 0.5
    # stress = E_scale * yield_strain * 0.5 + (yield_stress_T) * 0.5
    
    # Better formulation:
    # stress = E_scale * eps + yield_fraction * (yield_stress_T - E_scale * yield_strain + hardening)
    # This ensures continuity at yield point.
    
    stress = elastic_stress + yield_fraction * (yield_stress_T - E_scale * yield_strain + hardening)
    
    # Add offset and temperature interaction
    stress = stress + offset + temp_interact * T_norm * eps
    
    return stress
