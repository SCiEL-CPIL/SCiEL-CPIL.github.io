# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    1-3-1 Tanh MLP with 10 parameters.
    
    Structure:
    - Input: x (scaled to [-1, 1])
    - Hidden Layer: 3 units, tanh activation
    - Output Layer: 1 unit, linear activation
    
    Parameters:
    - params[0:3]: Weights from input to hidden (W_in)
    - params[3:6]: Biases for hidden layer (b_in)
    - params[6:9]: Weights from hidden to output (W_out)
    - params[9]:   Bias for output (b_out)
    
    This model is a universal approximator and can capture complex non-linear
    trends without fixed basis centers, relying on BFGS to optimize the
    non-linear parameter landscape.
    """
    x = np.asarray(x, dtype=float).flatten()
    params = np.asarray(params, dtype=float).flatten()
    
    # Ensure exactly 10 parameters
    if len(params) < 10:
        params = np.concatenate([params, np.ones(10 - len(params))])
    else:
        params = params[:10]
        
    # Scale input x from [0.1, 6] to [-1, 1] for numerical stability
    # Center: 3.05, Half-Range: 2.95
    x_center = 3.05
    x_scale = 2.95
    x_scaled = (x - x_center) / x_scale
    
    # Extract parameters
    W_in = params[0:3]    # shape (3,)
    b_in = params[3:6]    # shape (3,)
    W_out = params[6:9]   # shape (3,)
    b_out = params[9]     # scalar
    
    # Compute hidden layer activations
    # x_scaled: (N,), W_in: (3,) -> broadcast to (N, 3)
    z = x_scaled[:, np.newaxis] * W_in[np.newaxis, :] + b_in[np.newaxis, :]
    h = np.tanh(z)        # shape (N, 3)
    
    # Compute output
    y = h @ W_out + b_out # shape (N,)
    
    return y
# EVOLVE-BLOCK-END
