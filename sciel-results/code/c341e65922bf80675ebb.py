import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 140.0

def equation(b, s, temp, pH, params):
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros_like(b, dtype=np.float64)
    
    p = params
    
    # Normalization to roughly [0, 1] or similar manageable ranges
    # b: [1, 5] -> [0, 1]
    # s: [1, 5] -> [0, 1]
    # temp: [1, 40] (approx) -> [0, 1]
    # pH: [1, 10] (approx) -> [0, 1]
    
    b_n = b / 5.0
    s_n = s / 5.0
    t_n = (temp - 1.0) / 39.0
    ph_n = (pH - 1.0) / 9.0
    
    # Structural Change: 
    # Instead of a simple product of Monod * Gaussian * Gaussian * Inhibition,
    # we use a more flexible "Generalized Growth Model" that allows for:
    # 1. A non-monotonic response to substrate (Haldane/Andrews) or just Monod.
    #    Let's stick to Monod but allow the Hill coefficient to vary.
    # 2. A flexible temperature response: Not just Gaussian, but a double-Gaussian or 
    #    a shifted Gaussian to account for asymmetry, or simply a more robust Gaussian.
    #    Actually, the previous score dropped from 35 to 4.6. Why?
    #    The parent (p19) had score 35. The child (p23) has score 4.6.
    #    Wait, the prompt says "CURRENT PROGRAM (score 4.6347)".
    #    And the parent was 35.18. So the current program is WORSE than its parent.
    #    The diff shows the parent had "Robust Monod: use log-space...".
    #    The child changed to "Direct form... s_n**hill".
    #    Maybe the direct form is less stable or the parameter mapping changed?
    #    Parent p19:
    #    ks = exp(p[1])
    #    hill = clip(p[9], 0.5, 5.0)
    #    s_safe = clip(s_n, 1e-12, None)
    #    ratio = ks / s_safe
    #    ... (log space calculation implied)
    #    
    #    Child p23:
    #    ks = exp(p[1])
    #    hill = abs(p[9]) + 0.1
    #    s_pow = s_n ** hill
    #    ks_pow = ks ** hill
    #    monod = s_pow / (ks_pow + s_pow)
    #
    #    The issue might be that `hill` in the parent was clipped to [0.5, 5], ensuring stability.
    #    In the child, `hill` is `abs(p[9]) + 0.1`, which can be large if p[9] is large.
    #    If p[9] is 10, hill=10.1. s_n is in [0.2, 1]. s_n^10 is small. ks^10 might be huge or tiny.
    #    This can lead to numerical issues or poor gradient flow.
    #
    #    Also, the score dropped significantly. Let's revert to a more stable formulation
    #    but make a STRUCTURAL change as requested.
    #
    #    Structural Change Proposal:
    #    Use a **Fourier Series** or **Polynomial** approximation for the environmental factors (temp, pH)
    #    instead of fixed Gaussian shapes. This allows the optimizer to find the exact shape of the
    #    growth curve if it's not perfectly Gaussian.
    #    However, 10 parameters is tight.
    #
    #    Alternative Structural Change:
    #    Use a **Mixture of Experts** or **Switching Mechanism**.
    #    e.g., Growth rate = max(0, (Monod_term * Env_term) - Inhibition)
    #    Or, allow the temperature response to be a **Lorentzian** (Cauchy) which has heavier tails,
    #    sometimes fitting experimental data better than Gaussians.
    #
    #    Let's try a **Haldane (Andrews) kinetics** for substrate, which accounts for substrate inhibition.
    #    mu = (mu_max * s) / (Ks + s + s^2/Ki)
    #    This is structurally different from Monod.
    #    And for temperature, use a **Lorentzian** bell curve.
    #
    #    Parameters:
    #    p[0]: log(mu_max)
    #    p[1]: log(Ks)
    #    p[2]: log(Ki)  (Inhibition constant)
    #    p[3]: t_opt (Temperature optimum)
    #    p[4]: log(t_width) (Lorentzian width)
    #    p[5]: ph_opt (pH optimum)
    #    p[6]: log(ph_width) (Lorentzian width)
    #    p[7]: log(Kb) (Biomass inhibition constant)
    #    p[8]: Hill coefficient for biomass inhibition (or just fixed 2)
    #    p[9]: Bias / Baseline growth rate
    
    # Normalize inputs
    # b in [1, 5] -> b_n in [0.2, 1]
    # s in [1, 5] -> s_n in [0.2, 1]
    # temp in [1, 40] -> t_n in [0, 1]
    # pH in [1, 10] -> ph_n in [0, 1]
    
    b_n = b / 5.0
    s_n = s / 5.0
    t_n = (temp - 1.0) / 39.0
    ph_n = (pH - 1.0) / 9.0
    
    # 1. Substrate Kinetics: Haldane/Andrews
    # mu_s = s_n / (Ks + s_n + s_n^2 / Ki)
    # Normalize s_n to be around 1? s_n is [0.2, 1].
    # Let's work in original scale or scaled?
    # If we use s_n, Ks and Ki should be in same scale.
    
    # p[1] is log(Ks_scaled)
    # p[2] is log(Ki_scaled)
    # To ensure positivity and avoid 0 division:
    ks = np.exp(p[1]) + 1e-8
    ki = np.exp(p[2]) + 1e-8
    
    # Haldane term: s / (Ks + s + s^2/Ki)
    # To avoid overflow if Ki is small, compute carefully.
    # Term = s * Ki / (Ks*Ki + s*Ki + s^2)
    # This is numerically more stable.
    numerator = s_n * ki
    denominator = ks * ki + s_n * ki + s_n**2
    haldane = numerator / (denominator + 1e-12)
    
    # 2. Temperature Response: Lorentzian
    # L(t) = 1 / (1 + ((t - t_opt) / width)^2)
    # p[3] is t_opt
    # p[4] is log(width)
    t_opt = p[3]
    t_width = np.exp(p[4]) + 1e-8
    
    # Clip t_opt to reasonable range to prevent optimizer from running away?
    # t_n is in [0, 1]. If t_opt is 100, the factor is ~0 everywhere.
    # It's better to let the optimizer decide, but maybe constrain?
    # For now, let it be free, but the exp(width) ensures width > 0.
    
    temp_factor = 1.0 / (1.0 + ((t_n - t_opt) / t_width)**2)
    
    # 3. pH Response: Lorentzian
    # p[5] is ph_opt
    # p[6] is log(width)
    ph_opt = p[5]
    ph_width = np.exp(p[6]) + 1e-8
    
    ph_factor = 1.0 / (1.0 + ((ph_n - ph_opt) / ph_width)**2)
    
    # 4. Biomass Inhibition
    # p[7] is log(Kb)
    # p[8] is Hill coefficient for biomass. Let's use exp to ensure positive?
    # Or just a fixed power of 2?
    # Let's use a flexible inhibition: 1 / (1 + (b_n / Kb)^h)
    # p[8] is h.
    kb = np.exp(p[7]) + 1e-8
    h_b = np.abs(p[8]) + 0.1
    
    # Calculate (b_n / kb)^h
    # b_n is in [0.2, 1]. kb is positive.
    ratio_b = b_n / kb
    # If h is large, ratio_b^h can be 0 or inf.
    # Use log space?
    # log(ratio_b^h) = h * log(ratio_b)
    # If ratio_b < 1, log is negative.
    # Let's stick to direct power for now, clipping ratio_b to avoid 0**0?
    # b_n is never 0.
    dens_term = 1.0 / (1.0 + ratio_b**h_b)
    
    # 5. Max Rate and Bias
    # p[0] is log(mu_max)
    # p[9] is bias
    mu_max = np.exp(p[0]) + 1e-8
    bias = p[9]
    
    result = mu_max * haldane * temp_factor * ph_factor * dens_term + bias
    
    return result
