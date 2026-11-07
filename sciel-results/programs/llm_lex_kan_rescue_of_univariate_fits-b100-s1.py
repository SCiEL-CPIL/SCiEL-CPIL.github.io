import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Ensure params is a flat array of length 10
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
        
    # Extract parameters
    # Params 0-3: Polynomial coefficients (degree 0-3)
    b0, b1, b2, b3 = params[0], params[1], params[2], params[3]
    
    # Params 4-6: First Tanh unit (weight, center, slope)
    a1, c1, s1 = params[4], params[5], params[6]
    
    # Params 7-9: Second Tanh unit (weight, center, slope)
    a2, c2, s2 = params[7], params[8], params[9]
    
    # Compute polynomial part
    # Using Horner's method for stability and speed
    poly = b0 + x * (b1 + x * (b2 + x * b3))
    
    # Compute Tanh part
    # np.tanh is stable and handles large inputs well (saturates to +/- 1)
    # We use two units to allow for more complex non-linear shapes
    # The centers c1, c2 and slopes s1, s2 allow the units to shift and sharpen
    t1 = a1 * np.tanh(s1 * (x - c1))
    t2 = a2 * np.tanh(s2 * (x - c2))
    
    return poly + t1 + t2
