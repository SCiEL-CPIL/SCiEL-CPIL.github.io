import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-informed basis using smooth, bounded nonlinearities (sin, tanh).
    This avoids ill-conditioning from high-degree polynomials and captures
    periodic restoring forces and saturation effects typical of nonlinear oscillators.
    """
    # Compute basis functions
    sin_x = np.sin(x)
    tanh_x = np.tanh(x)
    sin_v = np.sin(v)
    tanh_v = np.tanh(v)
    
    # Linear coupling
    x_v = x * v
    
    # Cross terms
    x_sin_v = x * sin_v
    v_sin_x = v * sin_x
    x_tanh_v = x * tanh_v
    v_tanh_x = v * tanh_x
    
    # Higher order coupling
    x_v_sin_x = x * v * sin_x

    # Combine with parameters
    # params[0] * sin(x) + params[1] * tanh(x) + params[2] * sin(v) + params[3] * tanh(v)
    # + params[4] * x*v + params[5] * x*sin(v) + params[6] * v*sin(x)
    # + params[7] * x*tanh(v) + params[8] * v*tanh(x) + params[9] * x*v*sin(x)
    
    return (params[0] * sin_x + 
            params[1] * tanh_x + 
            params[2] * sin_v + 
            params[3] * tanh_v + 
            params[4] * x_v + 
            params[5] * x_sin_v + 
            params[6] * v_sin_x + 
            params[7] * x_tanh_v + 
            params[8] * v_tanh_x + 
            params[9] * x_v_sin_x)
