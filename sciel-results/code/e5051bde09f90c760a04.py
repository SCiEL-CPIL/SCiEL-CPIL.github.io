"""
Strategy: Liebig's Law of the Minimum via Smooth Maximum Aggregation.

Rationale:
The current program uses a linear combination of basis functions. While this creates a convex
loss landscape for BFGS, it may lack the capacity to model the sharp, multiplicative constraints
imposed by limiting factors (Liebig's Law). In biological growth, the rate is often determined
by the *most* limiting factor (the minimum of several potentials), not the sum or product.

This new approach models growth as a "soft minimum" of several factor-specific potential functions.
We use the smooth maximum function:
smooth_max(a, b, c, ...) ≈ -1/k * log(sum(exp(-k * a), exp(-k * b), ...))
This function approximates min(a, b, c, ...) as k -> infinity, but is differentiable and smooth
for finite k.

The factors are:
1. Substrate limitation (Monod-like)
2. Temperature limitation (Gaussian peak)
3. pH limitation (Gaussian peak)
4. Density limitation (Logistic-like or quadratic)

By modeling the growth rate as the soft minimum of these factors, the model can capture
sharp transitions where one factor becomes severely limiting. The parameters control the
scale and offset of each factor, allowing BFGS to fit the "bottleneck" structure.

To keep the problem well-conditioned and within the 120s budget, we use a fixed, moderate
temperature parameter 'k' for the smooth max, and only 10 parameters to scale and shift
the individual factor potentials.
"""

import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using Smooth Maximum (Soft Min) aggregation.
    
    Inputs:
    b: Population density (range ~[1, 5])
    s: Substrate (range ~[1, 5])
    temp: Temperature (range ~[1, 40])
    pH: pH level (range ~[2, 12])
    
    Params (length 10):
    p[0]: Scale for Substrate factor
    p[1]: Offset for Substrate factor
    p[2]: Scale for Temperature factor
    p[3]: Offset for Temperature factor
    p[4]: Scale for pH factor
    p[5]: Offset for pH factor
    p[6]: Scale for Density factor
    p[7]: Offset for Density factor
    p[8]: Global Scale (multiplies the final soft-min result)
    p[9]: Global Offset (adds to the final soft-min result)
    """
    # Normalize inputs to [0, 1] for numerical stability
    b_n = (b - 1.0) / 4.0
    s_n = (s - 1.0) / 4.0
    t_n = (temp - 1.0) / 39.0
    ph_n = (pH - 2.0) / 10.0

    # Define individual factor potentials.
    # We want these to be high when the factor is abundant/favorable, and low when limiting.
    # The soft-min will then pick the lowest one.
    
    # 1. Substrate: Monod-like saturation. Higher s_n -> higher potential.
    # Use a simple linear or quadratic form scaled by p[0] and shifted by p[1].
    # Let's use s_n / (1 + s_n) for saturation, but linear is safer for BFGS.
    # Let's try: potential_s = p[0] * s_n + p[1]
    pot_s = params[0] * s_n + params[1]

    # 2. Temperature: Gaussian peak at ~37C (t_n ~ 0.92).
    # Higher potential near optimum, lower away.
    # potential_t = p[2] * exp(-0.5 * ((t_n - 0.92) / 0.15)**2) + p[3]
    t_gauss = np.exp(-0.5 * ((t_n - 0.92) / 0.15)**2)
    pot_t = params[2] * t_gauss + params[3]

    # 3. pH: Gaussian peak at ~7.0 (ph_n ~ 0.5).
    # potential_ph = p[4] * exp(-0.5 * ((ph_n - 0.5) / 0.2)**2) + p[5]
    ph_gauss = np.exp(-0.5 * ((ph_n - 0.5) / 0.2)**2)
    pot_ph = params[4] * ph_gauss + params[5]

    # 4. Density: Growth often decreases at high density (crowding).
    # Potential should be high at low density, low at high density.
    # Let's use (1 - b_n) or similar.
    # potential_b = p[6] * (1 - b_n) + p[7]
    pot_b = params[6] * (1 - b_n) + params[7]

    # Smooth Maximum (Soft Min) of the four potentials.
    # soft_min(pot_s, pot_t, pot_ph, pot_b) = -1/k * log(exp(-k*pot_s) + exp(-k*pot_t) + exp(-k*pot_ph) + exp(-k*pot_b))
    # To avoid overflow, we can use the log-sum-exp trick.
    # Let k be a fixed sharpness parameter. A value of 10-50 is typical.
    # If k is too high, the function becomes non-smooth (kinks). If too low, it becomes an average.
    # Let's use k = 20 as a reasonable default. This is fixed to avoid adding another param.
    k = 20.0
    
    # Compute exponents
    exp_s = np.exp(-k * pot_s)
    exp_t = np.exp(-k * pot_t)
    exp_ph = np.exp(-k * pot_ph)
    exp_b = np.exp(-k * pot_b)
    
    # Sum of exponents
    sum_exp = exp_s + exp_t + exp_ph + exp_b
    
    # Avoid log(0)
    sum_exp = np.maximum(sum_exp, 1e-300)
    
    # Soft min
    soft_min_val = -(1.0 / k) * np.log(sum_exp)
    
    # Apply global scale and offset
    result = params[8] * soft_min_val + params[9]
    
    return result

# EVOLVE-BLOCK-END
