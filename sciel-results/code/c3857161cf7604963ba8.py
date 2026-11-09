"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a sum-of-sines (Fourier) basis.
    
    Strategy:
    1. Normalize inputs to [0, 1] based on known training ranges.
    2. Construct a basis of sinusoidal terms with learnable frequencies and phases.
    3. The output is a linear combination of these basis functions plus a bias.
    
    This approach captures periodic or oscillatory constraints (like pH and temp optima) 
    more efficiently than polynomials or local RBFs with limited parameters.
    
    Parameters (10 total):
    p[0]: Bias (intercept)
    p[1]: Weight for sin(freq_b * b + phase_b)
    p[2]: Weight for sin(freq_s * s + phase_s)
    p[3]: Weight for sin(freq_t * temp + phase_t)
    p[4]: Weight for sin(freq_p * pH + phase_p)
    p[5]: Frequency for b
    p[6]: Frequency for s
    p[7]: Frequency for temp
    p[8]: Frequency for pH
    p[9]: Global phase shift (applied to all terms)
    """
    # Deadline guard
    start_time = time.time()
    timeout = 110.0  # Safety margin before 120s limit
    
    # Normalization constants based on task description ranges:
    # b: [1, 4.998] -> span ~4
    # s: [1, 4.999] -> span ~4
    # temp: [1.002, 39.98] -> span ~39
    # pH: [2, 12] -> span ~10
    
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    temp_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    # Extract parameters
    bias = params[0]
    w_b = params[1]
    w_s = params[2]
    w_t = params[3]
    w_p = params[4]
    
    freq_b = params[5]
    freq_s = params[6]
    freq_t = params[7]
    freq_p = params[8]
    phase_global = params[9]
    
    # To ensure stable optimization, we can constrain frequencies to be positive
    # by using exp() or just letting BFGS handle it. Since sin is periodic, 
    # negative frequency is equivalent to positive frequency with phase shift.
    # However, to avoid oscillation issues during optimization, we might want 
    # to keep them in a reasonable range. But for a "different strategy", 
    # let's just use the raw params and see if BFGS handles it.
    # Actually, using exp for frequencies ensures they are positive, which is 
    # often better conditioned.
    freq_b_eff = np.exp(freq_b)
    freq_s_eff = np.exp(freq_s)
    freq_t_eff = np.exp(freq_t)
    freq_p_eff = np.exp(freq_p)
    
    # Compute basis functions
    # We use a simple sum of sines. We could add cosines, but that would use 
    # more parameters. With 10 params, 1 bias + 4 weights + 4 freqs + 1 global phase 
    # is a tight but feasible configuration.
    # Alternatively, we could have individual phases, but that requires more params.
    # Let's try: bias + w_b*sin(f_b*b + p_g) + w_s*sin(f_s*s + p_g) + ...
    
    term_b = np.sin(freq_b_eff * b_norm + phase_global)
    term_s = np.sin(freq_s_eff * s_norm + phase_global)
    term_t = np.sin(freq_t_eff * temp_norm + phase_global)
    term_p = np.sin(freq_p_eff * ph_norm + phase_global)
    
    prediction = bias + w_b * term_b + w_s * term_s + w_t * term_t + w_p * term_p
    
    # Check for non-finite values
    if not np.all(np.isfinite(prediction)):
        return np.zeros_like(prediction)
        
    return prediction
# EVOLVE-BLOCK-END
