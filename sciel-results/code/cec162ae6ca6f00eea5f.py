"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses triple-tanh regime model (elastic, yield plateau, hardening)
    with softplus yield offset and temperature-dependent corrections.
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure inputs are arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Softplus for non-negative yield stress offset
    # sp(x) = log(1 + exp(x)), approximates max(0, x)
    def softplus(x):
        return np.log1p(np.exp(x))
    
    # Three tanh terms for three regimes
    # Elastic: linear response
    # Yield plateau: transition from elastic to plastic
    # Hardening: saturation in plastic region
    
    # p1: scale for yield transition (sharp)
    # p2: scale for hardening saturation (slower)
    # p3: scale for elastic-to-plastic transition
    
    s_yield = np.tanh(p1 * strain)
    s_hard = np.tanh(p2 * strain)
    s_elastic = np.tanh(p3 * strain)
    
    # Temperature terms
    t1 = temp
    t2 = t1 * t1
    
    # Yield stress with temperature dependence
    # Higher temperature reduces yield strength
    yield_stress = softplus(p0 + p6 * t1 + p7 * t2)
    
    # Elastic modulus with temperature dependence
    # Higher temperature reduces elastic modulus
    elastic_mod = p4 + p5 * t1
    
    # Hardening coefficient
    hardening_coeff = p8 + p9 * t1
    
    # Build the model
    # Elastic contribution: only active when strain is small
    elastic_part = elastic_mod * strain * (1.0 - np.abs(s_yield))
    
    # Yield plateau contribution
    yield_part = yield_stress * s_yield
    
    # Hardening contribution: activates after yield
    hardening_part = hardening_coeff * s_hard * np.abs(s_yield)
    
    result = elastic_part + yield_part + hardening_part
    
    return result
# EVOLVE-BLOCK-END
