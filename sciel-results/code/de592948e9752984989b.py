# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a hybrid structure:
    1. Monod kinetics for substrate saturation (biologically motivated).
    2. A flexible, non-linear response surface for temperature and pH using a 
       fixed-basis RBF (Gaussian) expansion. This captures complex optimal ranges 
       and interactions without assuming simple periodicity (Fourier) or 
       simple quadratic shapes.
    
    The model structure is:
    mu = p0 * Monod(s) * (1 + RBF_correction(temp, pH)) + p1
    
    Where:
    - p0 scales the maximum growth rate determined by substrate.
    - p1 is a global bias.
    - RBF_correction is a linear combination of Gaussian kernels centered at 
      fixed points in the normalized (temp, pH) space. The coefficients of these 
      kernels are the remaining parameters (p2..p9).
    
    This approach allows the model to "bump" the growth rate up or down in 
    specific regions of the temperature-pH plane, capturing peaks (optimal 
    conditions) and valleys (inhibitory conditions) more flexibly than 
    global polynomials or simple Fourier series.
    
    Inputs:
    b: Population density (range ~[1, 5])
    s: Substrate concentration (range ~[1, 40])
    temp: Temperature (range ~[1, 40])
    pH: pH level (range ~[1, 14])
    
    Parameters (10):
    p0: Scaling factor for the Monod term (max growth rate scale)
    p1: Bias term
    p2..p9: Coefficients for 8 RBF centers in the (temp, pH) space
    """
    p = params
    
    # Time guard: if this function is called repeatedly in a loop or context 
    # where time is critical, we ensure it's fast. This function is O(N) 
    # with small constants, so it should be very fast. 
    # No complex optimization loop here, just evaluation.
    
    # Normalize inputs to [0, 1] for the RBF centers
    # b: [1, 5] -> not used in RBF, but available if needed. 
    #     In this hybrid, b is ignored in the RBF part to focus params on 
    #     temp/pH which are usually the primary regulators of max rate.
    #     If b matters, it could be included in a separate term, but we 
    #     only have 10 params. Let's stick to temp/pH RBF.
    # s: [1, 40]
    s_norm = (s - 1.0) / 39.0
    # temp: [1, 40]
    temp_norm = (temp - 1.0) / 39.0
    # pH: [1, 14]
    ph_norm = (pH - 1.0) / 13.0
    
    # Monod term for substrate
    # Use exp(p[0])? No, p0 is the scale. Let's use a fixed Ks or a simple 
    # scaling. The current program used exp(p[0]) for Ks and p[1] for scale.
    # Let's simplify: Use a fixed reasonable Ks (e.g., 5) to save a param, 
    # or keep it simple. 
    # Actually, let's keep the structure flexible but simple.
    # Monod = s / (Ks + s). Let Ks be fixed at 5.0 (mid-range).
    # Or, to be safer and let the optimizer adjust the "strength" of substrate 
    # limitation, we can just use s_norm directly if we assume linear 
    # saturation? No, Monod is better.
    # Let's use a fixed Ks = 5.0.
    Ks = 5.0
    monod = s / (Ks + s)
    
    # RBF Centers in normalized (temp, pH) space.
    # We choose 8 centers to cover the space.
    # Grid points: 2x4 or 4x2? Let's do a 3x3 grid minus one, or 4x2.
    # Let's use a 2x4 grid: 2 temp levels, 4 pH levels.
    # Temp centers: 0.25, 0.75
    # pH centers: 0.125, 0.375, 0.625, 0.875
    
    # Pre-compute centers
    temp_centers = np.array([0.25, 0.75])
    ph_centers = np.array([0.125, 0.375, 0.625, 0.875])
    
    # Create meshgrid for centers
    # We need 8 coefficients: p[2] to p[9]
    
    # Calculate RBF values for each data point against each center
    # temp_norm is shape (N,), ph_norm is shape (N,)
    
    # To be efficient, we can vectorize.
    # But N is 7500, and 8 centers. This is small.
    
    # Bandwidth (sigma) for RBF. Fixed to ensure stability.
    # If sigma is too small, the model becomes overfit/noisy.
    # If sigma is too large, it becomes a global polynomial.
    # Let's use sigma = 0.3 (in normalized units).
    sigma = 0.3
    
    rbf_sum = np.zeros_like(temp_norm, dtype=np.float64)
    
    idx = 0
    for t_c in temp_centers:
        for p_c in ph_centers:
            # Distance squared
            dt = temp_norm - t_c
            dp = ph_norm - p_c
            dist_sq = dt**2 + dp**2
            # Gaussian kernel
            kernel = np.exp(-dist_sq / (2.0 * sigma**2))
            # Add weighted contribution
            rbf_sum += p[idx + 2] * kernel
            idx += 1
    
    # Combine
    # mu = p0 * monod * (1 + rbf_sum) + p1
    # The (1 + rbf_sum) term modulates the growth rate.
    # If rbf_sum is negative enough, it can inhibit growth.
    # To ensure stability and avoid negative growth rates if not desired, 
    # we could clip, but negative growth (death) is biologically valid.
    
    result = p[0] * monod * (1.0 + rbf_sum) + p[1]
    
    # Ensure finite
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
