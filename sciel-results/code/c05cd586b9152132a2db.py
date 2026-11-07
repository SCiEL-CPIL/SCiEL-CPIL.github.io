import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Center and scale inputs for numerical conditioning
    # Training ranges: b in [1, 4.998], s in [1, 4.999], temp in [1.002, 39.98], pH in [2, 12]
    # Centers: b_c=3.0, s_c=3.0, T_c=20.5, pH_c=7.0
    # Scales: b_s=1.5, s_s=1.5, T_s=19.0, pH_s=5.0
    
    b_c = (b - 3.0) / 1.5
    s_c = (s - 3.0) / 1.5
    T_c = (temp - 20.5) / 19.0
    pH_c = (pH - 7.0) / 5.0
    
    # Monod-like substrate saturation in original space
    # params[0] is K_s (half-saturation constant)
    K_s = params[0] + 0.01  # small offset to avoid division by zero issues
    u_s = s / (K_s + s)
    
    # Temperature effect: cubic polynomial in normalized space
    # This allows for more flexible temperature response including potential asymmetry
    t_poly = params[1] + params[2] * T_c + params[3] * T_c**2 + params[4] * T_c**3
    
    # pH effect: quadratic polynomial in normalized space
    pH_poly = params[5] + params[6] * pH_c + params[7] * pH_c**2
    
    # Cell density effect: linear in normalized space
    y_b = params[8] * b_c
    
    # Main model: substrate-gated temperature effect + pH effect + density
    # The substrate gates the temperature response, reflecting biological reality
    y = u_s * t_poly + pH_poly + y_b
    
    # Add explicit interaction terms to capture non-separable effects
    # These help BFGS escape local minima by providing explicit gradients for interactions
    # Substrate-temperature interaction (beyond the gating)
    y_s_T = 0.0  # Already captured by u_s * t_poly
    # Substrate-pH interaction
    y_s_pH = 0.0  # Not included to keep model parsimonious
    
    return y
