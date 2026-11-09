"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate from population density, substrate, temperature and pH.
    
    Uses a 2-expert mixture model gated by temperature, with Monod substrate
    kinetics, quadratic pH response, and density inhibition.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-based soft gate: sigmoid centered at p0 with sharpness p1
    gate = 1.0 / (1.0 + np.exp(-p1 * (temp - p0)))
    
    # pH response: quadratic centered at p2 with width p3 (shared)
    ph_term = 1.0 / (1.0 + ((pH - p2) / (p3 + 1e-8)) ** 2)
    
    # Density inhibition: multiplicative factor (shared)
    density_factor = 1.0 / (1.0 + p8 * b / (p9 + 1e-8))
    
    # Expert 1 (low-temp regime): Monod kinetics
    monod_1 = s / (p4 + s)
    expert_1 = p5 * monod_1
    
    # Expert 2 (high-temp regime): Monod kinetics
    monod_2 = s / (p6 + s)
    expert_2 = p7 * monod_2
    
    # Combine experts with gate
    combined = (1.0 - gate) * expert_1 + gate * expert_2
    
    # Apply pH and density effects
    result = combined * ph_term * density_factor
    
    return result
# EVOLVE-BLOCK-END
