"""Symbolic regression: stress-strain of aluminium rod with piecewise-linear yield model."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a piecewise-linear model with:
    - Elastic region (linear in strain)
    - Yield point (kink at strain = p[3])
    - Work-hardening region (different slope)
    - Temperature dependence (thermal softening)
    
    Parameters (10 total):
    p[0]: Elastic modulus (slope before yield)
    p[1]: Thermal coefficient (stress reduction per unit temp)
    p[2]: Base offset / yield stress baseline
    p[3]: Yield strain (breakpoint)
    p[4]: Work-hardening slope (after yield)
    p[5]: Temperature interaction with strain
    p[6]: Additional hardening term coefficient
    p[7]: Saturation strain scale
    p[8]: Saturation stress level
    p[9]: Nonlinear correction coefficient
    """
    # Ensure inputs are arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Extract parameters
    p0 = params[0]  # Elastic modulus
    p1 = params[1]  # Thermal coefficient
    p2 = params[2]  # Base offset
    p3 = params[3]  # Yield strain
    p4 = params[4]  # Work-hardening slope
    p5 = params[5]  # Temp-strain interaction
    p6 = params[6]  # Additional hardening
    p7 = params[7]  # Saturation strain scale
    p8 = params[8]  # Saturation stress
    p9 = params[9]  # Nonlinear correction
    
    # Effective strain (temperature-adjusted)
    # Temperature reduces effective elastic response
    strain_eff = strain * (1.0 + p5 * (temp - 0.5))
    
    # Piecewise-linear yield model using abs and min/max
    # Elastic part: p0 * strain_eff
    # Post-yield part: p4 * max(strain_eff - p3, 0)
    # This creates a kink at strain_eff = p3
    
    elastic = p0 * strain_eff
    post_yield = p4 * np.maximum(strain_eff - p3, 0.0)
    
    # Saturation term: approaches p8 as strain increases
    # Using a smooth-like approximation via min
    saturation = p8 * np.minimum(1.0, strain_eff / (p7 + 1e-12))
    
    # Temperature softening: reduces stress as temp increases
    thermal = p1 * (temp - 0.5)
    
    # Nonlinear correction: quadratic-like term for curvature
    nonlinear = p9 * strain_eff * np.abs(strain_eff - p3)
    
    # Additional hardening contribution
    hardening = p6 * np.maximum(strain_eff - p3, 0.0) ** 2
    
    # Combine all terms
    stress = (elastic + post_yield + saturation - thermal + nonlinear + hardening + p2)
    
    # Ensure output has correct shape
    return stress
# EVOLVE-BLOCK-END
