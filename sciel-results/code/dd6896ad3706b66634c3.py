"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a ReLU-based piecewise-linear basis with smooth temperature modulation
    to capture the elastic-plastic transition (yield kink) in stress-strain behavior.
    
    Parameters:
    p0: constant offset
    p1: elastic modulus (slope before yield)
    p2: plastic hardening slope (ReLU weight)
    p3: yield strain location (shift for ReLU)
    p4: temp coefficient for elastic modulus
    p5: temp coefficient for yield location
    p6: temp coefficient for plastic slope
    p7: secondary ReLU weight (multi-kink)
    p8: secondary yield location
    p9: temperature quadratic coupling
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent parameters
    # Elastic modulus varies with temperature
    E_eff = p1 * (1.0 + p4 * temp)
    # Yield strain shifts with temperature
    sy_loc = p3 + p5 * temp
    # Plastic slope varies with temperature
    hard_slope = p2 * (1.0 + p6 * temp)
    
    # Primary ReLU: captures the main yield kink
    relu1 = np.maximum(strain - sy_loc, 0.0)
    
    # Secondary ReLU: captures additional kinks or multi-stage hardening
    relu2 = np.maximum(strain - p8, 0.0)
    
    # Temperature quadratic interaction term
    t2 = temp * temp
    
    # Combine: elastic part + plastic hardening + secondary kink + temp effects
    prediction = (
        p0 +
        E_eff * strain +
        hard_slope * relu1 +
        p7 * relu2 +
        p9 * t2
    )
    
    # Ensure finite output
    prediction = np.where(np.isfinite(prediction), prediction, 0.0)
    
    return prediction
# EVOLVE-BLOCK-END
