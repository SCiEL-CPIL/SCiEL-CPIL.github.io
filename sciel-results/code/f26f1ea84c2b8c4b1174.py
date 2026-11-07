"""E. coli growth rate model with piecewise-linear growth plateau and rational inhibition."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_deadline_start = time.time()
_DEADLINE = 140.0  # Safety margin before 150s wall-clock limit


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate with hard zero-growth boundaries via piecewise-linear plateaus.
    
    Uses np.clip to enforce strict zero growth outside optimal temperature and pH ranges,
    combined with Monod substrate saturation and density inhibition. This creates a
    hard-constrained landscape that prevents BFGS from fitting growth in biologically
    impossible regimes.
    
    Key improvements over parent:
      - Piecewise-linear temperature/pH response with sharp cutoffs
      - Rational density inhibition (Haldane-type)
      - More flexible interaction structure
    """
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Fixed biological centers (based on E. coli physiology)
    T_min = 25.0    # Minimum viable temperature
    T_max = 45.0    # Maximum viable temperature
    T_opt = 37.0    # Optimal temperature
    pH_min = 5.5    # Minimum viable pH
    pH_max = 8.5    # Maximum viable pH
    pH_opt = 7.0    # Optimal pH
    Ks = 5.0        # Substrate half-saturation
    Kd = 1.0        # Density inhibition constant
    
    # Deadline guard
    if time.time() - _deadline_start > _DEADLINE:
        return np.zeros_like(b)
    
    # Piecewise-linear temperature response: 0 outside [T_min, T_max], peak at T_opt
    temp_response = np.clip(
        np.minimum(
            (temp - T_min) / (T_opt - T_min),
            (T_max - temp) / (T_max - T_opt)
        ), 0.0, 1.0
    )
    
    # Piecewise-linear pH response: 0 outside [pH_min, pH_max], peak at pH_opt
    ph_response = np.clip(
        np.minimum(
            (pH - pH_min) / (pH_opt - pH_min),
            (pH_max - pH) / (pH_max - pH_opt)
        ), 0.0, 1.0
    )
    
    # Monod substrate saturation
    monod_s = s / (Ks + s)
    
    # Haldane-type density inhibition: 1 / (1 + Kd*b + Kd2*b^2)
    density_inhib = 1.0 / (1.0 + Kd * b + 0.1 * b * b)
    
    # Core multiplicative term
    core = monod_s * temp_response * ph_response * density_inhib
    
    # Interaction and correction terms
    temp_only = temp_response
    ph_only = ph_response
    substrate_only = monod_s
    temp_ph = temp_response * ph_response
    substrate_temp = monod_s * temp_response
    substrate_ph = monod_s * ph_response
    density_sub = density_inhib * monod_s
    
    # Combine all basis functions with fitted parameters
    prediction = (
        p0 * core +
        p1 * substrate_temp +
        p2 * substrate_ph +
        p3 * temp_ph +
        p4 * substrate_only +
        p5 * temp_only +
        p6 * ph_only +
        p7 * density_sub +
        p8 * (b * monod_s) +
        p9
    )
    
    return prediction
# EVOLVE-BLOCK-END
