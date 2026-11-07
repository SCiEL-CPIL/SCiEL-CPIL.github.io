import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # params[0]: bias
    # params[1]: substrate half-saturation constant (Monod-like)
    # params[2-4]: temperature polynomial coefficients (linear, quadratic, cubic)
    # params[5-6]: pH polynomial coefficients (linear, quadratic)
    # params[7]: density coefficient
    # params[8]: substrate-temp interaction
    # params[9]: substrate-pH interaction

    # Monod-like substrate saturation with fitted half-saturation
    u_s = s / (1.0 + params[1] * s)

    # Temperature polynomial (cubic to capture non-monotonic growth vs temp)
    t_poly = params[2] + params[3] * temp + params[4] * temp**2

    # pH polynomial (quadratic for optimal pH)
    pH_poly = params[5] + params[6] * pH

    # Density effect (linear inhibition or promotion)
    y_b = params[7] * b

    # Interaction terms
    y_s_T = params[8] * u_s * temp
    y_s_pH = params[9] * u_s * pH

    # Combine: substrate-gated temperature effect + pH effect + density + interactions + bias
    return params[0] + u_s * t_poly + pH_poly + y_b + y_s_T + y_s_pH
