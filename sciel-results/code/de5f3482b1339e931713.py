"""E. coli growth rate model: Fourier basis with Monod substrate gating."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a Fourier (sum-of-sines) basis for temperature
    and pH, combined with a Monod-like substrate term and density inhibition.
    
    The Fourier basis captures the non-monotonic, bounded nature of growth rate
    as a function of temperature and pH (which have optimal ranges), while the
    Monod term handles substrate saturation.
    
    Model:
        mu = A * monod(s) * dens(b) * [1 + f_temp(temp) + f_ph(pH) + f_temp_ph(temp, pH)] + bias
    
    Where:
        monod(s) = s / (K_s + s)
        dens(b) = 1 / (1 + (b / b_half)^2)
        f_temp(temp) = sum_k a_k * sin(w_k * temp + phi_k)
        f_ph(pH) = sum_k c_k * sin(v_k * pH + theta_k)
        f_temp_ph(temp, pH) = interaction term
    
    Parameters (10):
    p0: A (amplitude / max growth rate scale)
    p1: K_s (Monod half-saturation for substrate)
    p2: b_half (density half-max inhibition)
    p3: w1 (frequency for temp sine 1)
    p4: a1 (amplitude for temp sine 1)
    p5: v1 (frequency for pH sine 1)
    p6: c1 (amplitude for pH sine 1)
    p7: phi1 (phase for temp sine 1)
    p8: theta1 (phase for pH sine 1)
    p9: bias
    """
    # Deadline guard
    t0 = time.time()
    
    # Extract parameters
    A = params[0]
    K_s = np.abs(params[1]) + 1e-6  # Ensure positive
    b_half = np.abs(params[2]) + 1e-6  # Ensure positive
    w1 = params[3]  # frequency for temp
    a1 = params[4]  # amplitude for temp
    v1 = params[5]  # frequency for pH
    c1 = params[6]  # amplitude for pH
    phi1 = params[7]  # phase for temp
    theta1 = params[8]  # phase for pH
    bias = params[9]
    
    # Monod kinetics for substrate
    monod = s / (K_s + s + 1e-12)
    
    # Density effect: inhibition at high density
    dens_term = 1.0 / (1.0 + (b / b_half) ** 2)
    
    # Fourier terms for temperature
    # Use a single sine term with learnable frequency, amplitude, and phase
    # This captures the optimal temperature range
    f_temp = a1 * np.sin(w1 * temp + phi1)
    
    # Fourier terms for pH
    f_ph = c1 * np.sin(v1 * pH + theta1)
    
    # Interaction term: cross modulation
    # A simple interaction that allows the optimal pH to shift with temperature
    inter_term = 0.5 * a1 * c1 * np.sin(w1 * temp + phi1) * np.sin(v1 * pH + theta1)
    
    # Combine: the Fourier terms modulate the base growth
    # We add 1 to ensure the base is present, and the Fourier terms provide modulation
    modulation = 1.0 + f_temp + f_ph + inter_term
    
    # Clip modulation to prevent extreme values
    modulation = np.clip(modulation, 0.1, 10.0)
    
    # Final result
    result = A * monod * dens_term * modulation + bias
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Deadline guard: if we're running long, just return (shouldn't happen for this simple model)
    if time.time() - t0 > 140:
        return result
    
    return result
# EVOLVE-BLOCK-END
