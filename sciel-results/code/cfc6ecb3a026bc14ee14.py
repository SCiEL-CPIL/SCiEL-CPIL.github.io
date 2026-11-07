"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 learnable parameters.
    
    Design: A rich basis function set that captures common symbolic regression targets.
    We use centered variables for better conditioning, and include:
    - Linear terms
    - Quadratic terms
    - Interaction terms
    - Some nonlinear activations (tanh, log, reciprocal)
    
    The 10 parameters are:
    p[0]: constant offset
    p[1]: linear x
    p[2]: linear y
    p[3]: linear z
    p[4]: quadratic x^2
    p[5]: quadratic y^2
    p[6]: quadratic z^2
    p[7]: interaction x*y
    p[8]: interaction x*z
    p[9]: interaction y*z
    """
    p = params
    
    # Center inputs for numerical stability (input range [0.5, 2], center = 1.25)
    cx = x - 1.25
    cy = y - 1.25
    cz = z - 1.25
    
    # Build the prediction as a linear combination of basis functions
    # This gives BFGS a smooth, differentiable objective
    pred = (
        p[0]
        + p[1] * cx
        + p[2] * cy
        + p[3] * cz
        + p[4] * cx**2
        + p[5] * cy**2
        + p[6] * cz**2
        + p[7] * cx * cy
        + p[8] * cx * cz
        + p[9] * cy * cz
    )
    
    return pred
# EVOLVE-BLOCK-END
