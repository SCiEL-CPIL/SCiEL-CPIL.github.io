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
    # np.log1p(exp(x)) is stable softplus
    yield_offset = np.log1p(np.exp(p0))
    
    # Three tanh terms for three regimes:
    # 1. Elastic-to-yield transition (sharp, p1 scales)
    # 2. Yield plateau (p2 scales)  
    # 3. Hardening saturation (p3 scales)
    s_elastic = np.tanh(p1 * strain)
    s_yield = np.tanh(p2 * strain)
    s_hard = np.tanh(p3 * strain)
    
    # Linear strain for elastic region
    s1 = strain
    
    # Temperature terms
    t1 = temp
    t2 = t1 * t1
    
    # Gates that reduce correction terms in saturated regions
    gate_yield = 1.0 - np.abs(s_yield)
    gate_hard = 1.0 - np.abs(s_hard)
    gate_all = gate_yield * gate_hard
    
    # Build the model:
    # p4 * s_elastic: elastic-to-yield transition
    # p5 * s_yield: yield plateau level
    # p6 * s_hard: hardening saturation
    # p7 * s1: linear elastic slope
    # yield_offset: baseline stress level (non-negative via softplus)
    # p8 * t1: temperature linear correction
    # p9 * t2: temperature quadratic correction
    # Additional: strain-temp interaction gated by yield gate
    # Additional: yield-temperature interaction
    
    result = (yield_offset
              + p4 * s_elastic
              + p5 * s_yield
              + p6 * s_hard
              + p7 * s1
              + p8 * t1
              + p9 * t2
              + 0.5 * p4 * s1 * t1 * gate_yield
              + 0.5 * p5 * s_yield * t1 * gate_all)
    
    return result
# EVOLVE-BLOCK-END
