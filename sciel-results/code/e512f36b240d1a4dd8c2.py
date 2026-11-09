import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using sum-of-sines (Fourier) basis.
    
    y = bias + A_s * sin(w_s * s_norm + phi_s) + A_t * sin(w_t * temp_norm + phi_t)
      + A_p * sin(w_p * ph_norm + phi_p) + A_b * sin(w_b * b_norm + phi_b)
      + cross terms
    
    Parameters (10):
    0: bias
    1: A_t (temperature amplitude)
    2: w_t (temperature frequency, clipped to [0.5, 10])
    3: phi_t (temperature phase, clipped to [-pi, pi])
    4: A_p (pH amplitude)
    5: w_p (pH frequency, clipped to [0.5, 10])
    6: phi_p (pH phase, clipped to [-pi, pi])
    7: A_s (substrate amplitude)
    8: A_b (density amplitude)
    9: cross_coeff (interaction coefficient)
    """
    # Normalize inputs to [0, 1]
    b_norm = np.clip((b - 1.0) / 4.0, 0.0, 1.0)
    s_norm = np.clip((s - 1.0) / 39.0, 0.0, 1.0)
    temp_norm = np.clip((temp - 1.0) / 39.0, 0.0, 1.0)
    ph_norm = np.clip((pH - 2.0) / 10.0, 0.0, 1.0)
    
    # Unpack parameters with safe clipping
    bias = params[0]
    A_t = params[1]
    w_t = np.clip(params[2], 0.5, 10.0)
    phi_t = np.clip(params[3], -np.pi, np.pi)
    A_p = params[4]
    w_p = np.clip(params[5], 0.5, 10.0)
    phi_p = np.clip(params[6], -np.pi, np.pi)
    A_s = params[7]
    A_b = params[8]
    cross_coeff = params[9]
    
    # Fourier basis terms for each input
    # Temperature: captures optimal window (bell-shaped response)
    temp_term = A_t * np.sin(w_t * temp_norm + phi_t)
    
    # pH: captures optimal window (bell-shaped response)
    ph_term = A_p * np.sin(w_p * ph_norm + phi_p)
    
    # Substrate: monotonic saturation-like behavior
    # Use a combination that can approximate Hill kinetics
    s_term = A_s * np.sin(w_t * 0.5 * s_norm + phi_t * 0.5)  # Reuse some params for substrate
    
    # Density: linear-ish modulation
    b_term = A_b * np.sin(w_p * 0.5 * b_norm + phi_p * 0.5)
    
    # Cross term: interaction between temperature and pH
    cross_term = cross_coeff * temp_term * ph_term
    
    # Combine
    result = bias + temp_term + ph_term + s_term + b_term + cross_term
    
    # Ensure finite
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
