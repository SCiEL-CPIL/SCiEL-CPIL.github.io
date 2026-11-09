"""Fixed-architecture Random Fourier Feature Network.
The 10 parameters are the output-layer weights. The hidden layer consists of 
fixed random projections of the inputs (t, x, v) passed through sinusoidal activations.
This makes the model linear in the parameters, ensuring BFGS converges to the 
global minimum of the convex MSE landscape, avoiding local minima found in 
nonlinear basis searches."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 135.0
_GLOBAL = {}

def _initialize_hidden():
    """Initialize fixed random hidden-layer weights and biases."""
    if 'hidden_W' in _GLOBAL:
        return
    
    rng = np.random.default_rng(42)
    
    # Inputs: t, x, v. 
    # We need to map these to a fixed basis. 
    # Since we don't know the scale of t, x, v exactly, we use random projections.
    # However, t is likely in [30, 50], x in [-0.2, 0.2], v in [-0.3, 0.36].
    # To ensure the random features have reasonable variance, we scale inputs roughly to unit variance.
    # t ~ 40, x ~ 0.2, v ~ 0.3.
    # Let's create a fixed set of basis functions.
    
    # We have 10 parameters. 
    # Strategy: Use a small number of hidden units K, and let the 10 params be 
    # weights for K features + 1 bias + (10-K-1) extra features? 
    # No, the interface requires exactly 10 params.
    # Let's use K=9 hidden units and 1 bias term = 10 params total.
    # Or K=10 hidden units, no bias? Bias is usually important.
    # Let's use K=9 random Fourier features and 1 bias.
    
    K = 9
    D = 3 # Dimensionality of input (t, x, v)
    
    # Random projections matrix W_hidden (K x D)
    # Scale: We want the arguments to sin/cos to be of order 1.
    # Input scales: t~40, x~0.2, v~0.3.
    # If we project directly, t dominates. We should normalize inputs first?
    # We cannot modify the input inside equation() easily without overhead, 
    # but we can bake the normalization into W_hidden.
    
    # Let's assume typical scales:
    # t_mean ~ 40, t_std ~ 5
    # x_mean ~ 0, x_std ~ 0.1
    # v_mean ~ 0, v_std ~ 0.1
    
    # We will construct basis functions: phi_i(t,x,v) = cos(w_i^T [t,x,v] + b_i)
    # To make this effective, w_i should be scaled inversely to input std.
    
    # Let's generate random weights.
    # We'll use a fixed seed for reproducibility.
    
    # W_hidden: shape (K, D)
    # We want w_i^T z to be roughly N(0, 1).
    # z = [t, x, v].
    # Let's approximate scales:
    # scale_t = 1.0 / 5.0
    # scale_x = 1.0 / 0.1
    # scale_v = 1.0 / 0.1
    
    scales = np.array([0.2, 10.0, 10.0]) # 1/std approx
    
    # Generate random directions
    W_raw = rng.normal(size=(K, D))
    # Normalize rows to unit length
    norms = np.linalg.norm(W_raw, axis=1, keepdims=True)
    W_unit = W_raw / norms
    
    # Scale by frequency parameter sigma. 
    # For RFF, sigma is often 1/sqrt(D) or tuned. Let's try sigma = 1.0.
    sigma = 1.0
    W_hidden = W_unit * sigma * scales
    
    # Biases
    b_hidden = rng.uniform(0, 2 * np.pi, size=K)
    
    _GLOBAL['hidden_W'] = W_hidden
    _GLOBAL['hidden_b'] = b_hidden
    _GLOBAL['K'] = K

_initialize_hidden()


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of fixed Random Fourier Features.
    
    params[0] is the bias.
    params[1:10] are the weights for the 9 hidden units.
    """
    W = _GLOBAL['hidden_W']
    b = _GLOBAL['hidden_b']
    K = _GLOBAL['K']
    
    # Stack inputs: (N, 3)
    # t, x, v are 1D arrays of length N
    inputs = np.column_stack((t, x, v))
    
    # Project: (N, K)
    proj = inputs @ W.T + b
    
    # Basis functions: cos(proj)
    # We use cos. Could mix sin/cos but cos is sufficient for span if biases are random.
    features = np.cos(proj)
    
    # Linear combination
    # params[0] is bias, params[1:] are weights
    bias = params[0]
    weights = params[1:1+K]
    
    pred = bias + features @ weights
    
    return pred
# EVOLVE-BLOCK-END
