"""Symbolic regression: an equation for E. coli growth rate."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with a flexible polynomial basis in standardized
    inputs, combined with Monod substrate saturation and quadratic/cubic terms
    to capture non-Gaussian optima and interactions.
    
    The model is linear in 10 parameters, allowing BFGS to converge reliably.
    
    Basis terms:
    1. Bias
    2. Monod substrate saturation
    3. Standardized temperature (linear)
    4. Standardized pH (linear)
    5. Temperature^2 (quadratic)
    6. pH^2 (quadratic)
    7. Temperature × pH (cross)
    8. Temperature × Monod (interaction)
    9. pH × Monod (interaction)
    10. Monod × (Temp^2 + pH^2) (higher-order interaction)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # --- Standardize inputs for numerical stability ---
    # b in [1, 5], s in [1, 40], temp in [1, 50], pH in [1, 50] (training ranges)
    # Use fixed centers and scales based on biological knowledge and training ranges
    
    # Temperature: optimal ~37°C, viable 15-45°C
    temp_center = 30.0
    temp_scale = 10.0
    t_std = (temp - temp_center) / temp_scale
    
    # pH: optimal ~7, viable 5-9
    ph_center = 7.0
    ph_scale = 2.0
    ph_std = (pH - ph_center) / ph_scale
    
    # Substrate: Monod saturation with Ks ~ 5 (mid-range of [1, 40])
    Ks = 5.0
    monod_s = s / (Ks + s)
    
    # Density: standardized for conditioning
    b_center = 3.0
    b_scale = 2.0
    b_std = (b - b_center) / b_scale

    # --- Construct basis terms ---
    # 1. Bias
    term0 = np.ones_like(b, dtype=np.float64)
    
    # 2. Monod substrate
    term1 = monod_s
    
    # 3. Linear temperature
    term2 = t_std
    
    # 4. Linear pH
    term3 = ph_std
    
    # 5. Quadratic temperature
    term4 = t_std * t_std
    
    # 6. Quadratic pH
    term5 = ph_std * ph_std
    
    # 7. Temperature × pH cross
    term6 = t_std * ph_std
    
    # 8. Temperature × Monod
    term7 = t_std * monod_s
    
    # 9. pH × Monod
    term8 = ph_std * monod_s
    
    # 10. Monod × (Temp^2 + pH^2)
    term9 = monod_s * (t_std * t_std + ph_std * ph_std)

    # --- Linear combination ---
    pred = (p0 * term0 + p1 * term1 + p2 * term2 + p3 * term3 +
            p4 * term4 + p5 * term5 + p6 * term6 + p7 * term7 +
            p8 * term8 + p9 * term9)

    return pred
# EVOLVE-BLOCK-END
