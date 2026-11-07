import numpy as np
import time

_deadline_start = time.time()
_DEADLINE = 140.0

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate with scaled inputs, parameterized optima, and soft-min interaction.
    """
    if time.time() - _deadline_start > _DEADLINE:
        return np.zeros_like(b)

    # Unpack parameters
    # p0: T_opt (scaled), p1: T_width (scaled)
    # p2: pH_opt (scaled), p3: pH_width (scaled)
    # p4: Ks (scaled), p5: Kd (scaled)
    # p6: Vmax
    # p7: Linear T correction
    # p8: Linear pH correction
    # p9: Offset
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Scale inputs to O(1) range to help BFGS
    # temp: [1, 40] -> [0.2, 2.0] roughly. Center ~ 1.0 (30C)
    # Let's use (temp - 20) / 10. Range: -1.9 to 2.0.
    t_scaled = (temp - 20.0) / 10.0
    
    # pH: [2, 12] -> [0, 5]. Center ~ 1.0 (7pH is 0.5).
    # Let's use (pH - 6) / 2. Range: -2 to 3.
    ph_scaled = (pH - 6.0) / 2.0
    
    # s: [1, 5] -> [0.2, 1.0].
    s_scaled = s / 5.0
    
    # b: [1, 5] -> [0.2, 1.0].
    b_scaled = b / 5.0

    # Ensure positive widths
    t_width = np.maximum(p1, 0.1)
    ph_width = np.maximum(p3, 0.1)
    ks = np.maximum(p4, 0.1)
    kd = np.maximum(p5, 0.1)
    vmax = p6

    # Gaussian responses on scaled inputs
    t_gauss = np.exp(-((t_scaled - p0) ** 2) / (2.0 * t_width ** 2))
    ph_gauss = np.exp(-((ph_scaled - p2) ** 2) / (2.0 * ph_width ** 2))

    # Soft Min Interaction
    # M = a*b / sqrt(a^2 + b^2 + eps)
    denom = np.sqrt(t_gauss**2 + ph_gauss**2 + 1e-6)
    soft_min = (t_gauss * ph_gauss) / denom

    # Monod on scaled substrate
    monod = s_scaled / (ks + s_scaled)

    # Density Inhibition on scaled density
    inhib = 1.0 / (1.0 + kd * b_scaled)

    # Core Prediction
    core = vmax * monod * soft_min * inhib

    # Additive Corrections (on scaled inputs to keep magnitudes similar)
    pred = core + p7 * t_scaled + p8 * ph_scaled + p9

    return pred
