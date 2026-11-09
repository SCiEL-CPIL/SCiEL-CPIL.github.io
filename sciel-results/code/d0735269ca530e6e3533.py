# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a Fourier basis with an interaction term.
    
    Model: y = p0 + A_t*sin(F_t*T + P_t) + A_p*sin(F_p*pH + P_p) + A_s*sin(F_s*s) + A_b*sin(F_b*b) + A_int*sin(F_t*T + P_t)*sin(F_p*pH + P_p)
    
    We fix F_s and F_b to save parameters for the interaction term A_int.
    
    params layout (10 total):
      params[0]: base offset (p0)
      params[1]: amplitude for temp (A_t)
      params[2]: frequency for temp (F_t)
      params[3]: phase for temp (P_t)
      params[4]: amplitude for pH (A_p)
      params[5]: frequency for pH (F_p)
      params[6]: phase for pH (P_p)
      params[7]: amplitude for s (A_s) [F_s fixed to 0.5]
      params[8]: amplitude for b (A_b) [F_b fixed to 0.1]
      params[9]: interaction amplitude (A_int)
    """
    p0 = params[0]
    
    A_t = params[1]
    F_t = params[2]
    P_t = params[3]
    
    A_p = params[4]
    F_p = params[5]
    P_p = params[6]
    
    A_s = params[7]
    A_b = params[8]
    A_int = params[9]
    
    # Fixed frequencies for s and b to approximate linear/saturating effects
    F_s = 0.5
    F_b = 0.1
    
    # Compute base sine terms
    sin_t = np.sin(F_t * temp + P_t)
    sin_p = np.sin(F_p * pH + P_p)
    sin_s = np.sin(F_s * s)
    sin_b = np.sin(F_b * b)
    
    # Additive terms
    term_t = A_t * sin_t
    term_p = A_p * sin_p
    term_s = A_s * sin_s
    term_b = A_b * sin_b
    
    # Interaction term between Temp and pH
    term_int = A_int * sin_t * sin_p
    
    return p0 + term_t + term_p + term_s + term_b + term_int
# EVOLVE-BLOCK-END
