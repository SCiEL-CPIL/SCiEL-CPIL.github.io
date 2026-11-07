"""Rewritten program: piecewise-linear growth plateau with rational inhibition."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model.
    
    Model components:
    1. Temperature factor: smooth piecewise-linear ramp using sigmoid transitions.
    2. pH factor: similar smooth piecewise-linear ramp for pH.
    3. Substrate factor: Monod-like saturation s / (Ks + s).
    4. Density inhibition: 1 / (1 + b/Ki).
    
    params (length 10):
    params[0]: max growth rate multiplier
    params[1]: T_min (temperature lower bound for growth)
    params[2]: T_opt (temperature optimum)
    params[3]: T_max (temperature upper bound for growth)
    params[4]: pH_min (pH lower bound)
    params[5]: pH_opt (pH optimum)
    params[6]: pH_max (pH upper bound)
    params[7]: Ks (substrate half-saturation constant)
    params[8]: Ki (density inhibition constant)
    params[9]: offset (baseline growth rate)
    """
    p = params
    
    # Ensure parameters are positive where needed
    Ks = np.abs(p[7]) + 1e-6
    Ki = np.abs(p[8]) + 1e-6
    
    # Temperature factor: smooth piecewise linear
    T_min = p[1]
    T_opt = p[2]
    T_max = p[3]
    
    # Ensure ordering with soft enforcement
    T_min_eff = T_min
    T_max_eff = T_max
    
    # Smooth ramp using sigmoid-like transitions
    # Width of transition regions
    w = 0.1
    
    # Rising part: 0 below T_min, 1 at T_opt
    # Use a smooth approximation of the linear ramp
    rise_width = np.maximum(T_opt - T_min_eff, w)
    # Smooth step from T_min to T_opt
    t_rise = (temp - T_min_eff) / rise_width
    temp_rise = np.clip(t_rise, 0, 1)
    # Apply smooth sigmoid to make it differentiable
    temp_rise_smooth = 1.0 / (1.0 + np.exp(-10.0 * (temp_rise - 0.5)))
    
    # Falling part: 1 at T_opt, 0 above T_max
    fall_width = np.maximum(T_max_eff - T_opt, w)
    t_fall = (temp - T_opt) / fall_width
    temp_fall = np.clip(1.0 - t_fall, 0, 1)
    temp_fall_smooth = 1.0 / (1.0 + np.exp(-10.0 * (temp_fall - 0.5)))
    
    # Combine: min of rise and fall to create the plateau
    temp_factor = np.minimum(temp_rise_smooth, temp_fall_smooth)
    
    # pH factor: smooth piecewise linear
    pH_min = p[4]
    pH_opt = p[5]
    pH_max = p[6]
    
    pH_min_eff = pH_min
    pH_max_eff = pH_max
    
    pH_rise_width = np.maximum(pH_opt - pH_min_eff, w)
    pH_rise = (pH - pH_min_eff) / pH_rise_width
    pH_rise = np.clip(pH_rise, 0, 1)
    pH_rise_smooth = 1.0 / (1.0 + np.exp(-10.0 * (pH_rise - 0.5)))
    
    pH_fall_width = np.maximum(pH_max_eff - pH_opt, w)
    pH_fall = (pH - pH_opt) / pH_fall_width
    pH_fall = np.clip(1.0 - pH_fall, 0, 1)
    pH_fall_smooth = 1.0 / (1.0 + np.exp(-10.0 * (pH_fall - 0.5)))
    
    pH_factor = np.minimum(pH_rise_smooth, pH_fall_smooth)
    
    # Substrate: Monod saturation
    substrate_factor = s / (Ks + s)
    
    # Density inhibition
    density_factor = 1.0 / (1.0 + b / Ki)
    
    # Combine
    mu = p[0] * temp_factor * pH_factor * substrate_factor * density_factor + p[9]
    
    return mu
# EVOLVE-BLOCK-END
