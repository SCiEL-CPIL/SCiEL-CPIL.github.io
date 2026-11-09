import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model using linear background and softplus hinge functions.
    
    This model uses:
    1. A linear baseline to capture the elastic region and general trend.
    2. Two 'softplus' hinge functions to capture the yield point and hardening behavior.
       Softplus approximates the ReLU/hinge function but is smooth everywhere.
       By allowing the slope inside the softplus to be fitted, the model can 
       approximate hard kinks (large slope) or smooth transitions (small slope).
    3. A quadratic term to capture overall curvature.
    
    Parameters:
    p0: intercept
    p1: linear strain coefficient
    p2: linear temp coefficient
    p3: strain-temp interaction coefficient
    p4: amplitude of first hinge
    p5: slope of first hinge (controls sharpness)
    p6: offset of first hinge (controls yield location)
    p7: amplitude of second hinge
    p8: slope of second hinge
    p9: offset of second hinge
    p10: quadratic strain coefficient
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Linear background
    linear_part = p0 + p1 * strain + p2 * temp + p3 * strain * temp
    
    # Softplus function: log(1 + exp(x))
    # To avoid overflow in exp, we use a stable implementation or clip the input.
    # np.logaddexp(0, x) is stable for softplus(x).
    
    # Hinge 1: Primary yield transition
    x1 = p5 * (strain + p6 * temp)
    # Clip x to prevent extreme values that might cause issues in logaddexp if not handled, 
    # though logaddexp is generally stable. 
    # However, if p5 is very large, x can be huge.
    # np.logaddexp(0, x) is stable.
    hinge1 = np.logaddexp(0.0, x1)
    
    # Hinge 2: Secondary hardening or correction
    x2 = p8 * (strain + p9 * temp)
    hinge2 = np.logaddexp(0.0, x2)
    
    # Quadratic correction
    quad_part = p10 * strain * strain
    
    stress = linear_part + p4 * hinge1 + p7 * hinge2 + quad_part
    
    return stress
