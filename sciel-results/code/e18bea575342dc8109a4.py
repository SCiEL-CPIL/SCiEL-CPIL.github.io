import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to improve conditioning of the basis functions
    # Ranges from prompt: x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
    # Scale to approx [-1, 1]
    x_n = x / 0.6
    v_n = v / 0.3
    
    # Construct basis terms
    # We use 10 parameters.
    # Term 0: Constant
    # Term 1: x
    # Term 2: v
    # Term 3: x^2
    # Term 4: x^3
    # Term 5: x*v
    # Term 6: v^2
    # Term 7: x^2*v
    # Term 8: x*v^2
    # Term 9: x^4
    
    t0 = np.ones_like(x)
    t1 = x_n
    t2 = v_n
    t3 = x_n * x_n
    t4 = x_n * x_n * x_n
    t5 = x_n * v_n
    t6 = v_n * v_n
    t7 = x_n * x_n * v_n
    t8 = x_n * v_n * v_n
    t9 = x_n * x_n * x_n * x_n
    
    # Combine with params
    pred = (params[0] * t0 +
            params[1] * t1 +
            params[2] * t2 +
            params[3] * t3 +
            params[4] * t4 +
            params[5] * t5 +
            params[6] * t6 +
            params[7] * t7 +
            params[8] * t8 +
            params[9] * t9)
            
    return pred
