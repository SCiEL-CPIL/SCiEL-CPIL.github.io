"""E. coli growth rate model with piecewise-linear growth plateau."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # 5s safety margin before 150s kill


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with a piecewise-linear growth plateau.
    
    Uses hard-constrained boundaries for temperature and pH to enforce
    zero growth outside biologically viable ranges, combined with
    rational inhibition terms for substrate saturation and density.
    
    The 10 parameters are:
      p0: overall scale
      p1: substrate half-saturation (Ks)
      p2: temperature lower bound
      p3: temperature upper bound
      p4: pH lower bound
      p5: pH upper bound
      p6: density inhibition constant
      p7: temperature center (shift)
      p8: pH center (shift)
      p9: quadratic temperature correction
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        return np.zeros_like(b, dtype=np.float64)
    
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure positive constants where needed
    Ks = max(abs(p1), 0.1)
    T_lower = min(p2, p3) - 0.1
    T_upper = max(p2, p3) + 0.1
    pH_lower = min(p4, p5) - 0.1
    pH_upper = max(p4, p5) + 0.1
    Kd = max(abs(p6), 0.01)
    T_center = p7
    pH_center = p8
    
    # Piecewise-linear temperature response: zero outside [T_lower, T_upper],
    # linear rise/fall to peak at T_center
    T_range = T_upper - T_lower
    if T_range < 0.01:
        T_range = 0.01
    
    # Temperature shape: triangular peak
    temp_dist_left = np.clip(T_center - temp, 0, T_center - T_lower)
    temp_dist_right = np.clip(temp - T_center, 0, T_upper - T_center)
    
    # Normalize distances to create triangular response in [0, 1]
    left_span = T_center - T_lower
    right_span = T_upper - T_center
    if left_span < 0.01:
        left_span = 0.01
    if right_span < 0.01:
        right_span = 0.01
    
    temp_response_left = np.clip(1.0 - temp_dist_left / left_span, 0, 1)
    temp_response_right = np.clip(1.0 - temp_dist_right / right_span, 0, 1)
    
    # Combine: use min of left and right responses for triangular shape
    temp_response = np.where(temp <= T_center, temp_response_left, temp_response_right)
    
    # Clip to ensure zero outside bounds
    temp_response = np.clip(temp_response, 0, 1)
    
    # Piecewise-linear pH response: similar triangular shape
    pH_dist_left = np.clip(pH_center - pH, 0, pH_center - pH_lower)
    pH_dist_right = np.clip(pH - pH_center, 0, pH_upper - pH_center)
    
    pH_left_span = pH_center - pH_lower
    pH_right_span = pH_upper - pH_lower
    if pH_left_span < 0.01:
        pH_left_span = 0.01
    if pH_right_span < 0.01:
        pH_right_span = 0.01
    
    ph_response_left = np.clip(1.0 - pH_dist_left / pH_left_span, 0, 1)
    ph_response_right = np.clip(1.0 - pH_dist_right / pH_right_span, 0, 1)
    
    ph_response = np.where(pH <= pH_center, ph_response_left, ph_response_right)
    ph_response = np.clip(ph_response, 0, 1)
    
    # Substrate saturation: rational (Monod-like)
    monod_s = s / (Ks + s)
    monod_s = np.clip(monod_s, 0, 1)
    
    # Density inhibition: rational
    density_inhib = 1.0 / (1.0 + Kd * b)
    density_inhib = np.clip(density_inhib, 0, 1)
    
    # Quadratic temperature correction for flexibility
    temp_norm = np.clip((temp - T_center) / max(T_range / 2, 0.1), -1, 1)
    temp_quad_correction = 1.0 + p9 * temp_norm ** 2
    temp_quad_correction = np.clip(temp_quad_correction, 0, 2)
    
    # Combine all terms
    prediction = p0 * monod_s * temp_response * ph_response * density_inhib * temp_quad_correction
    
    # Ensure finite output
    prediction = np.where(np.isfinite(prediction), prediction, 0.0)
    
    return prediction
# EVOLVE-BLOCK-END
