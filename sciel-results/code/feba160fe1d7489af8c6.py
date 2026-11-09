"""E. coli growth rate model using Fourier basis for flexible response surfaces."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a sum-of-sines (Fourier) basis.
    
    This structural change moves away from mechanistic priors (Monod/Hill/Gaussian)
    to a more general function approximator. The Fourier basis allows the model
    to capture complex, potentially periodic or oscillatory dependencies
    (e.g., sharp optima in temperature/pH) that might be better represented
    by trigonometric functions than by fixed-shape curves.
    
    Inputs are normalized to [0, 1] based on training ranges:
    b: [1, 5] -> [0, 1]
    s: [1, 50] -> [0, 1] (approx, using 1-50 for wider coverage if needed, but task says 1-4.998/39.98, let's use 1-50 for safety or 1-40?)
    Task says: [1, 4.998], [1, 4.999], [1.002, 39.98], [2, 12].
    So s goes up to ~40. Let's normalize s to [0,1] with range [1, 40].
    temp: [1, 40] -> [0, 1]
    pH: [2, 12] -> [0, 1]
    
    Parameters (10):
    p0: Amplitude (A) - overall scale
    p1: Bias (B)
    p2: Frequency f_b
    p3: Phase phi_b
    p4: Frequency f_s
    p5: Phase phi_s
    p6: Frequency f_t
    p7: Phase phi_t
    p8: Frequency f_p
    p9: Phase phi_p
    
    Model:
    y = A * (sin(f_b*b + phi_b) * sin(f_s*s + phi_s) * sin(f_t*t + phi_t) * sin(f_p*p + phi_p)) + B
    
    Note: To allow for positive and negative contributions and zero-crossings,
    we use the product of sine waves. The amplitude A and bias B allow scaling and shifting.
    However, a pure product of sines might be too restrictive (always zero if any input is at a zero crossing).
    
    Alternative: Sum of sines for each input, multiplied together?
    y = A * (c_b + sin(f_b*b + phi_b)) * (c_s + sin(f_s*s + phi_s)) * ...
    But we only have 10 params.
    
    Let's try:
    y = A * (sin(f_b*b + phi_b) + 1) * (sin(f_s*s + phi_s) + 1) * (sin(f_t*t + phi_t) + 1) * (sin(f_p*p + phi_p) + 1) + B
    
    This ensures the terms are in [0, 2] if sine is in [-1, 1].
    This allows for a range of values from 0 to 16*A + B.
    
    Wait, the frequencies and phases are the parameters.
    p0: A
    p1: B
    p2: f_b, p3: phi_b
    p4: f_s, p5: phi_s
    p6: f_t, p7: phi_t
    p8: f_p, p9: phi_p
    
    This uses all 10 parameters.
    """
    
    # 1. Normalize inputs to [0, 1]
    b_norm = np.clip((b - 1.0) / 4.0, 0.0, 1.0)
    s_norm = np.clip((s - 1.0) / 39.0, 0.0, 1.0) # Max s is ~40
    temp_norm = np.clip((temp - 1.0) / 39.0, 0.0, 1.0)
    ph_norm = np.clip((pH - 2.0) / 10.0, 0.0, 1.0)

    # 2. Unpack parameters
    A = params[0]
    B = params[1]
    
    f_b = params[2]
    phi_b = params[3]
    f_s = params[4]
    phi_s = params[5]
    f_t = params[6]
    phi_t = params[7]
    f_p = params[8]
    phi_p = params[9]
    
    # 3. Calculate Fourier terms
    # Using (sin + 1) to keep terms positive and bounded [0, 2]
    # This creates a multiplicative interaction that is zero when any sine is -1.
    # If the optimum is not at a zero-crossing, the phase shift will handle it.
    
    term_b = np.sin(f_b * b_norm + phi_b) + 1.0
    term_s = np.sin(f_s * s_norm + phi_s) + 1.0
    term_t = np.sin(f_t * temp_norm + phi_t) + 1.0
    term_p = np.sin(f_p * ph_norm + phi_p) + 1.0
    
    # 4. Combine
    # The product of these terms ranges from 0 to 16.
    result = A * term_b * term_s * term_t * term_p + B
    
    # 5. Ensure finite
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
