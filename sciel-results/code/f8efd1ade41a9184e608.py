"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a fixed-basis additive model (linear in params) combining:
    - Polynomial strain terms for elastic + strain hardening
    - Tanh saturation terms for yield behavior at multiple scales
    - Temperature terms for thermal softening
    - Interaction terms for thermo-mechanical coupling
    
    All 10 parameters are fitted by BFGS on a smooth, convex-like loss landscape.
    The tanh terms provide smooth approximations of piecewise-linear behavior
    (elastic plateau, yield transition, hardening) while remaining differentiable.
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Strain powers (elastic + hardening)
    s1 = strain
    s2 = s1 * s1
    s3 = s2 * s1
    
    # Tanh saturation terms for yield behavior
    # Multiple scales to capture different transition regimes:
    # t1: sharp transition near yield point (~0.01 strain)
    # t2: broader saturation (plastic flow plateau)
    # t3: very sharp, captures elastic-plastic boundary
    t1 = np.tanh(100.0 * s1)       # moderate sharpness
    t2 = np.tanh(20.0 * s1)        # broader transition
    t3 = np.tanh(500.0 * s1)       # very sharp, near-step function
    
    # Temperature effects
    # temp is in range [0.067, 1.0]
    t1_temp = temp
    t2_temp = temp * temp
    
    # Interaction terms
    st1 = s1 * t1_temp              # linear strain-temperature coupling
    st2 = t1 * t1_temp              # yield-temperature coupling
    st3 = s2 * t1_temp              # hardening-temperature coupling
    
    # Build the model: linear combination of basis functions
    # p0: intercept
    # p1: linear elastic response
    # p2: quadratic hardening
    # p3: primary yield saturation (sharp)
    # p4: secondary saturation (broad)
    # p5: very sharp elastic-plastic transition
    # p6: linear temperature effect
    # p7: strain-temperature interaction
    # p8: yield-temperature interaction
    # p9: quadratic temperature effect
    
    stress = (
        p0 +
        p1 * s1 +
        p2 * s2 +
        p3 * t1 +
        p4 * t2 +
        p5 * t3 +
        p6 * t1_temp +
        p7 * st1 +
        p8 * st2 +
        p9 * t2_temp
    )
    
    return stress
# EVOLVE-BLOCK-END
