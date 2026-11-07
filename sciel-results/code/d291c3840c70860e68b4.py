"""E. coli growth rate model with smooth, linear-in-parameters basis functions."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with a linear-in-parameters combination of
    smooth biological basis functions.
    
    This formulation ensures the optimization problem is convex (least squares
    in params), allowing BFGS to reliably find the global minimum for the
    coefficients. The basis functions are chosen to capture known biological
    trends: substrate saturation (Monod), bell-shaped temperature/pH optima
    (Gaussian), and density inhibition.
    
    Inputs:
        b: biomass density
        s: substrate concentration
        temp: temperature
        pH: pH value
        params: array of 10 fitted constants
    
    Returns:
        Array of predicted growth rates
    """
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs for numerical stability
    # Estimated ranges from task description:
    # b: [1, 5] -> center 3, scale 2
    b_n = (b - 3.0) / 2.0
    # s: [1, 5] -> center 3, scale 2
    s_n = (s - 3.0) / 2.0
    # temp: [1, 40] -> center 20.5, scale 19.5
    T_n = (temp - 20.5) / 19.5
    # pH: [2, 12] -> center 7, scale 5
    pH_n = (pH - 7.0) / 5.0
    
    # 1. Substrate Saturation (Monod-like)
    # Using a smooth step or rational function. s_n is centered, so we shift to positive.
    # Let s_pos = s_n + 2 (range [0, 4] approx if s in [1,5])
    # Actually, simpler: use the raw normalized s_n in a Monod form that works for negatives too?
    # Monod is typically for concentration > 0. Let's use a smooth rational function.
    # f(s) = s_n / (1 + |s_n|) is not differentiable at 0.
    # Let's stick to a polynomial or a shifted Monod.
    # Since s is in [1, 5], s_n is in [-1, 1].
    # Let's use a logistic-like or simple rational: s_n / (1 + s_n^2) ? No, that's odd.
    # Let's use: 1 / (1 + exp(-k*s_n)) ? 
    # Better: Just use polynomial terms for s if Monod is tricky with centered data.
    # But Monod is biologically strong.
    # Let's map s_n to [0, 1] range for Monod: s_map = (s_n + 1) / 2
    s_map = (s_n + 1.0) / 2.0
    monod_s = s_map / (0.5 + s_map)  # K_s = 0.5 in mapped units
    
    # 2. Temperature Response
    # Gaussian bell curve centered at T_n = 0 (20.5 C)
    # Width sigma_T
    sigma_T = 0.5
    temp_gauss = np.exp(-(T_n ** 2) / (2 * sigma_T ** 2))
    
    # 3. pH Response
    # Gaussian bell curve centered at pH_n = 0 (7.0 pH)
    sigma_pH = 0.5
    pH_gauss = np.exp(-(pH_n ** 2) / (2 * sigma_pH ** 2))
    
    # 4. Biomass Inhibition
    # Inverse quadratic or similar
    inhib_b = 1.0 / (1.0 + b_n ** 2)
    
    # Construct Basis Functions
    # We have 10 parameters. We need 10 basis functions.
    
    # Basis 0: Bias
    basis0 = np.ones_like(b_n)
    
    # Basis 1: Substrate effect (Monod)
    basis1 = monod_s
    
    # Basis 2: Temperature effect
    basis2 = temp_gauss
    
    # Basis 3: pH effect
    basis3 = pH_gauss
    
    # Basis 4: Substrate * Temperature
    basis4 = monod_s * temp_gauss
    
    # Basis 5: Substrate * pH
    basis5 = monod_s * pH_gauss
    
    # Basis 6: Temperature * pH
    basis6 = temp_gauss * pH_gauss
    
    # Basis 7: Main Interaction (Substrate * Temp * pH)
    basis7 = monod_s * temp_gauss * pH_gauss
    
    # Basis 8: Biomass inhibition on main term
    basis8 = basis7 * inhib_b
    
    # Basis 9: Linear biomass effect (to capture simple density dependence)
    basis9 = b_n * temp_gauss * pH_gauss
    
    # Combine linearly
    prediction = (
        p0 * basis0 +
        p1 * basis1 +
        p2 * basis2 +
        p3 * basis3 +
        p4 * basis4 +
        p5 * basis5 +
        p6 * basis6 +
        p7 * basis7 +
        p8 * basis8 +
        p9 * basis9
    )
    
    # Ensure prediction is finite
    prediction = np.where(np.isfinite(prediction), prediction, 0.0)
    
    return prediction
# EVOLVE-BLOCK-END
