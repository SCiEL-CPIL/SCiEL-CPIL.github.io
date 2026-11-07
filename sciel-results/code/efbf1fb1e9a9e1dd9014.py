# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Multiplicative parametric form: product of two affine linear terms.
    
    pred = (p0*x + p1*y + p2*z + p3) * (p4*x + p5*y + p6*z + p7)
    
    This can represent:
    - Linear: if one factor is constant (p4=p5=p6=0)
    - Bilinear (xy, xz, yz): if factors are single-variable
    - Full quadratic: all terms
    - Separable structures
    
    We also add a small linear correction term using p8, p9 to help with
    non-separable linear components that the product form can't capture exactly.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Factor 1: affine linear in x, y, z
    f1 = p0 * x + p1 * y + p2 * z + p3
    # Factor 2: affine linear in x, y, z
    f2 = p4 * x + p5 * y + p6 * z + p7
    
    # Product term (captures bilinear and quadratic interactions)
    prod = f1 * f2
    
    # Add small linear correction to handle non-separable linear parts
    # Use p8, p9 as additional linear terms to improve fit
    linear_correction = p8 * (x + y + z) + p9
    
    return prod + linear_correction
# EVOLVE-BLOCK-END
