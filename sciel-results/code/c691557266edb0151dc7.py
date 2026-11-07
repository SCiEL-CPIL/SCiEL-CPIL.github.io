import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [0, 1] based on training ranges
    # b: [1, 5] -> (b-1)/4
    # s: [1, 5] -> (s-1)/4
    # temp: [1, 40] -> (temp-1)/39
    # pH: [2, 12] -> (pH-2)/10
    
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    p = params
    
    # Structural change: Use sigmoid-like basis for temperature and pH to capture
    # sharp drop-off outside optimal ranges, plus more interaction terms.
    # This is a different modeling family: additive with non-polynomial basis functions.
    
    # Sigmoid-like functions for temperature and pH (centered at 0.5 in normalized space)
    # Using tanh for smooth bounded non-linearity
    t_sig = np.tanh(p[4] * (t_norm - 0.5))
    ph_sig = np.tanh(p[5] * (ph_norm - 0.5))
    
    # Linear terms
    term_b_lin = p[0] * b_norm
    term_s_lin = p[1] * s_norm
    term_t_lin = p[2] * t_norm
    term_ph_lin = p[3] * ph_norm
    
    # Non-linear basis terms (tanh captures asymmetric optima)
    term_t_nonlin = t_sig
    term_ph_nonlin = ph_sig
    
    # Interaction terms: biologically plausible pairs
    term_b_s = p[6] * b_norm * s_norm
    term_t_ph = p[7] * t_norm * ph_norm
    term_b_t = p[8] * b_norm * t_norm
    
    # Offset
    offset = p[9]
    
    y = offset + term_b_lin + term_s_lin + term_t_lin + term_ph_lin + \
         term_t_nonlin + term_ph_nonlin + \
         term_b_s + term_t_ph + term_b_t
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        return np.zeros_like(y)
        
    return y
