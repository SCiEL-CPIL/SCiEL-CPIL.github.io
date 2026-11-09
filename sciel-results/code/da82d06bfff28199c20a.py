# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate from population density, substrate, temperature and pH.
    
    Uses a sum-of-sines (Fourier) basis with learnable frequencies and phases
    for each input to capture periodic/oscillatory constraints (pH, temperature optima).
    
    params layout (10 total):
      params[0]: base offset
      params[1]: amplitude for b (population density)
      params[2]: frequency for b
      params[3]: amplitude for s (substrate)
      params[4]: frequency for s
      params[5]: amplitude for temp
      params[6]: frequency for temp
      params[7]: amplitude for pH
      params[8]: frequency for pH
      params[9]: phase shift (applied to all sine terms for global tuning)
    """
    # Extract parameters
    offset = params[0]
    amp_b = params[1]
    freq_b = params[2]
    amp_s = params[3]
    freq_s = params[4]
    amp_t = params[5]
    freq_t = params[6]
    amp_ph = params[7]
    freq_ph = params[8]
    phase = params[9]
    
    # Compute sine-based features for each input
    # Using sin(freq * x + phase) to allow periodic behavior
    term_b = amp_b * np.sin(freq_b * b + phase)
    term_s = amp_s * np.sin(freq_s * s + phase)
    term_t = amp_t * np.sin(freq_t * temp + phase)
    term_ph = amp_ph * np.sin(freq_ph * pH + phase)
    
    # Sum of all terms plus offset
    return offset + term_b + term_s + term_t + term_ph
# EVOLVE-BLOCK-END
