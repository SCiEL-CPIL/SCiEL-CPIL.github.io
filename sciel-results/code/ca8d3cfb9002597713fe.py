"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Strategy: Linear-in-parameters model with a physics-motivated basis.
    This approach ensures the optimization problem (fitting the 10 weights)
    is convex, allowing BFGS to reliably find the global optimum for this
    specific basis set, avoiding the local minima traps of fully non-linear
    physics models.
    
    The basis functions capture:
    - Elastic behavior (linear, quadratic, cubic strain terms)
    - Plastic yield and hardening (smoothed one-sided power laws starting near typical yield strain)
    - Temperature effects (interactions with strain and plastic strain)
    
    params (10 constants):
    p0: Intercept
    p1: Linear strain coeff (Elastic)
    p2: Quadratic strain coeff
    p3: Cubic strain coeff
    p4: Plastic strain coeff (1st order hardening)
    p5: Plastic strain squared coeff (2nd order hardening)
    p6: Temperature offset
    p7: Strain*Temp interaction
    p8: PlasticStrain*Temp interaction
    p9: QuadraticStrain*Temp interaction
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    e = strain
    t = temp
    
    # Define a smooth approximation for max(0, x)
    # Using s(x) = 0.5 * (x + sqrt(x^2 + eps))
    # This is smooth, differentiable, and approximates the kink at yield.
    eps = 1e-8
    
    def smooth_max(x):
        return 0.5 * (x + np.sqrt(x * x + eps))
    
    # Pre-compute basis functions
    # Elastic region terms
    phi0 = np.ones_like(e)
    phi1 = e
    phi2 = e * e
    phi3 = e * e * e
    
    # Plastic region terms (assuming yield strain ~ 0.005 for Aluminum)
    # This captures the deviation from linear elasticity
    plastic_strain = smooth_max(e - 0.005)
    phi4 = plastic_strain
    phi5 = plastic_strain * plastic_strain
    
    # Temperature terms
    phi6 = t
    
    # Interactions
    phi7 = e * t
    phi8 = plastic_strain * t
    phi9 = phi2 * t  # e^2 * t
    
    # Linear combination
    stress = (p0 * phi0 + 
              p1 * phi1 + 
              p2 * phi2 + 
              p3 * phi3 + 
              p4 * phi4 + 
              p5 * phi5 + 
              p6 * phi6 + 
              p7 * phi7 + 
              p8 * phi8 + 
              p9 * phi9)
              
    return stress
# EVOLVE-BLOCK-END
