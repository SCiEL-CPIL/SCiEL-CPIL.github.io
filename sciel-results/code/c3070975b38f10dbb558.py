"""Stress-strain model for aluminium rod with temperature dependence."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a generalized logistic / smooth-sigmoid model with temperature-dependent
    parameters to capture the elastic rise, yield point, and post-yield behavior.
    
    Model:
        E(T)  = p0 + p1*T          (elastic slope)
        Y(T)  = p2 + p3*T          (yield stress / plateau level)
        k(T)  = exp(p4 + p5*T)     (transition sharpness)
        s0(T) = p6 + p7*T          (yield strain offset)
        h(T)  = p8 + p9*T          (hardening coefficient)
        
        z = k(T) * (strain - s0(T))
        sigma = E(T)*strain + Y(T)*sigmoid(z) + h(T)*strain*sigmoid(z)
    
    The sigmoid provides a smooth transition from elastic to plastic behavior.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent parameters
    E = p0 + p1 * temp
    Y = p2 + p3 * temp
    k_raw = p4 + p5 * temp
    k = np.exp(np.clip(k_raw, -20.0, 20.0))
    s0 = p6 + p7 * temp
    h = p8 + p9 * temp
    
    # Compute sigmoid argument
    z = k * (strain - s0)
    
    # Stable sigmoid computation
    # sigmoid(z) = 1 / (1 + exp(-z))
    # For large positive z: sigmoid ~ 1
    # For large negative z: sigmoid ~ 0
    # Use stable formulation to avoid overflow
    z_clamped = np.clip(z, -30.0, 30.0)
    sigmoid_z = 1.0 / (1.0 + np.exp(-z_clamped))
    
    # Combine terms
    elastic_part = E * strain
    plastic_part = Y * sigmoid_z
    hardening_part = h * strain * sigmoid_z
    
    stress = elastic_part + plastic_part + hardening_part
    
    return stress
# EVOLVE-BLOCK-END
