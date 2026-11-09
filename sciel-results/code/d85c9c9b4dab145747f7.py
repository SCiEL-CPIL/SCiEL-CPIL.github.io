"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a Gaussian RBF (radial basis function) model in 2D (strain, temp).
    This is a flexible non-parametric-in-shape model that can capture
    arbitrary smooth stress-strain-temperature relationships.
    
    10 parameters:
    p0: global intercept
    p1: amplitude of RBF center 1
    p2: x-center (strain) of RBF center 1
    p3: y-center (temp) of RBF center 1
    p4: amplitude of RBF center 2
    p5: x-center (strain) of RBF center 2
    p6: y-center (temp) of RBF center 2
    p7: shared RBF width (inverse sigma)
    p8: linear strain coefficient
    p9: linear temp coefficient
    
    The model is:
    y = p0 + p8*strain + p9*temp + p1*exp(-p7^2*((strain-p2)^2 + (temp-p3)^2)) 
              + p4*exp(-p7^2*((strain-p5)^2 + (temp-p6)^2))
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure width is positive to avoid numerical issues
    width = 1.0 + np.abs(p7)
    
    # Compute squared distances for two RBF centers
    dist1_sq = (strain - p2)**2 + (temp - p3)**2
    dist2_sq = (strain - p5)**2 + (temp - p6)**2
    
    # Gaussian RBFs
    rbf1 = np.exp(-(width**2) * dist1_sq)
    rbf2 = np.exp(-(width**2) * dist2_sq)
    
    # Linear terms + RBF contributions
    prediction = p0 + p8 * strain + p9 * temp + p1 * rbf1 + p4 * rbf2
    
    # Ensure finite output
    prediction = np.where(np.isfinite(prediction), prediction, 0.0)
    
    return prediction
# EVOLVE-BLOCK-END
