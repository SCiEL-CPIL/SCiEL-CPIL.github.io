import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Models acceleration for a nonlinear oscillator using smooth nonlinear functions.
    Uses tanh for bounded nonlinearities and higher odd powers for stiffness/damping.
    This structure is smooth everywhere, aiding BFGS convergence.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Smooth nonlinear transformations
    # tanh(x) provides a smooth, bounded nonlinearity for stiffness
    tanh_x = np.tanh(p0 * x)
    
    # Higher odd powers for capturing stronger nonlinear effects
    x2 = x * x
    v2 = v * v
    x3 = x2 * x
    v3 = v2 * v
    x5 = x3 * x2
    v5 = v3 * v2
    
    # Construct the model:
    # Linear terms: p1*x + p2*v
    # Smooth nonlinear stiffness: p3*tanh_x + p4*x3
    # Smooth nonlinear damping: p5*tanh(p6*v) + p7*v3
    # Higher order corrections: p8*x5 + p9*v5
    # Note: using p6 as scaling for v in tanh to keep it smooth and parameterized
    
    tanh_v = np.tanh(p6 * v)
    
    return (p1 * x + 
            p2 * v + 
            p3 * tanh_x + 
            p4 * x3 + 
            p5 * tanh_v + 
            p7 * v3 + 
            p8 * x5 + 
            p9 * v5)
