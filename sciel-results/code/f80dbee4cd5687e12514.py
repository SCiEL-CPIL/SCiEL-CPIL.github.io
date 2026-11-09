"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator modeled as a generalized Duffing-van der Pol system.
    
    This approach uses a physics-informed structure:
    a(x, v) = -k*x - beta*x^3 + mu*v*(1 - x^2) + A*sin(w*t)
    
    However, we don't have time 't' as an input, only x and v. 
    In a phase-space representation of an autonomous or periodically driven system, 
    the "driving" term is often implicit or we model the effective force field.
    
    Since we only have x and v, we model the autonomous part of the dynamics 
    plus potential state-dependent forcing terms that can be learned.
    
    Structure:
    1. Linear spring: -k1 * x
    2. Cubic nonlinearity (Duffing): -k2 * x^3
    3. Quintic saturation (optional, helps bound energy): -k3 * x^5
    4. Linear damping: -c1 * v
    5. Van der Pol style nonlinear damping: c2 * x^2 * v
    6. Cross-term coupling: c3 * x * v
    7. Constant offset (bias): b0
    8. Quadratic velocity term (drag): c4 * v^2 (can be signed)
    9. Higher order cross: c5 * x^2 * v^2 (weak coupling)
    10. Higher order position: c6 * x^4 (symmetric correction)
    
    Parameters (10 total):
    p[0]: k1 (linear stiffness)
    p[1]: k2 (cubic stiffness)
    p[2]: k3 (quintic stiffness)
    p[3]: c1 (linear damping)
    p[4]: c2 (nonlinear damping coeff)
    p[5]: c3 (cross damping)
    p[6]: b0 (bias)
    p[7]: c4 (quadratic drag)
    p[8]: c5 (high-order cross)
    p[9]: c6 (quartic position)
    
    This structure enforces physical symmetries (e.g., odd powers for conservative forces, 
    even powers for dissipative forces) which helps BFGS converge faster and more robustly 
    than an unconstrained basis.
    """
    p = params
    
    # Precompute powers
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    
    v2 = v * v
    
    # Terms
    # Conservative (position dependent)
    t1 = p[0] * x
    t2 = p[1] * x3
    t3 = p[2] * x5
    t4 = p[9] * x4
    
    # Dissipative (velocity dependent)
    t5 = p[3] * v
    t6 = p[4] * (x2 * v)  # van der Pol term
    t7 = p[5] * (x * v)
    t8 = p[7] * v2
    t9 = p[8] * (x2 * v2)
    
    # Bias
    t10 = p[6]
    
    return t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9 + t10
# EVOLVE-BLOCK-END
