"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature."""
    # params: length 10
    # We use a 3rd order polynomial in strain and temp to capture interactions.
    # This is a linear model in parameters, ensuring convex optimization landscape.
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute powers and interactions
    s = strain
    t = temp
    s2 = s * s
    t2 = t * t
    s3 = s2 * s
    t3 = t2 * t
    st = s * t
    s2t = s2 * t
    st2 = s * t2
    s3t = s3 * t
    
    # Construct the polynomial
    # Terms:
    # 1. Constant
    # 2. Linear S
    # 3. Linear T
    # 4. Quad S
    # 5. Interaction ST
    # 6. Quad T
    # 7. Cubic S
    # 8. Interaction S^2 T
    # 9. Interaction S T^2
    # 10. Interaction S^3 T (or T^3, let's pick S^3 T for higher order strain interaction)
    
    stress = (p0 
              + p1 * s 
              + p2 * t 
              + p3 * s2 
              + p4 * st 
              + p5 * t2 
              + p6 * s3 
              + p7 * s2t 
              + p8 * st2 
              + p9 * s3t)
              
    return stress
# EVOLVE-BLOCK-END
