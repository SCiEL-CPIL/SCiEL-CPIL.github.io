"""Stress-strain model for aluminium rod using smooth hyperbolic tangent yield transition."""
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model: stress = E(T) * strain + Y(T) * smooth_saturation(strain, k)
    where smooth_saturation approximates the post-yield plateau behavior.
    
    Uses 10 parameters:
    p0: base elastic modulus E0
    p1: temperature coefficient for E
    p2: base yield stress Y0
    p3: temperature coefficient for Y
    p4: yield strain threshold (where transition begins)
    p5: sharpness of yield transition k
    p6: post-yield hardening slope
    p7: temperature interaction with strain
    p8: quadratic strain term coefficient
    p9: constant offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent elastic modulus
    E = p0 + p1 * temp
    
    # Temperature-dependent yield stress
    Y = p2 + p3 * temp
    
    # Ensure k is positive for stability
    k = np.abs(p5) + 1e-6
    
    # Smooth saturation: tanh-based transition from linear to plateau
    # For strain < p4: behaves like linear
    # For strain > p4: saturates toward Y
    x = (strain - p4) * k
    # Clip to avoid overflow in tanh
    x = np.clip(x, -50, 50)
    saturation = np.tanh(x)
    # Normalize so that at x=0, saturation=0, and as x->inf, saturation->1
    # We want: at strain=p4, contribution from yield term is 0
    # At large strain, contribution approaches Y
    
    # Post-yield hardening: additional linear term after yield
    hardening = p6 * np.maximum(strain - p4, 0)
    
    # Temperature-strain interaction
    temp_strain = p7 * temp * strain
    
    # Quadratic strain term for curvature
    quad_strain = p8 * strain**2
    
    # Combine all terms
    stress = E * strain + Y * saturation + hardening + temp_strain + quad_strain + p9
    
    return stress
