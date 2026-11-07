"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate from population density, substrate, temperature and pH."""
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Fixed biological centers
    T_opt = 37.0
    pH_opt = 7.0
    Ks = 1.0  # Monod half-saturation constant (fixed)
    
    # Basis functions - all linear in params
    # 1. Constant
    f0 = np.ones_like(b, dtype=float)
    
    # 2. Substrate saturation (Monod-like)
    f1 = s / (Ks + s)
    
    # 3. Temperature Gaussian response around optimum
    f2 = np.exp(-((temp - T_opt) ** 2) / (2.0 * 10.0))
    
    # 4. pH Gaussian response around optimum
    f3 = np.exp(-((pH - pH_opt) ** 2) / (2.0 * 1.0))
    
    # 5. Substrate-temperature interaction
    f4 = f1 * f2
    
    # 6. Substrate-pH interaction
    f5 = f1 * f3
    
    # 7. Temperature-pH interaction
    f6 = f2 * f3
    
    # 8. Density inhibition (simple)
    f7 = b / (1.0 + b)
    
    # 9. Density-substrate interaction
    f8 = f7 * f1
    
    # 10. Triple interaction: substrate * temp * pH
    f9 = f1 * f2 * f3

    return (p0 * f0 + p1 * f1 + p2 * f2 + p3 * f3 + p4 * f4 +
            p5 * f5 + p6 * f6 + p7 * f7 + p8 * f8 + p9 * f9)
# EVOLVE-BLOCK-END
