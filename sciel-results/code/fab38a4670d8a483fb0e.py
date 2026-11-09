# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Neural ODE-inspired Residual Basis with tanh activation.
    
    Uses a single hidden layer with fixed random weights (seeded for reproducibility)
    and tanh activation, combined with a linear skip connection.
    This provides a universal approximator structure that avoids ill-conditioning
    of high-degree polynomials while maintaining compactness.
    
    Structure:
    - 3 input features: t (normalized), x (normalized), v (normalized)
    - 7 hidden units with fixed weights, tanh activation
    - Linear output layer with 10 parameters
    - Skip connections for linear terms
    
    The 10 parameters control:
    - 7 weights from hidden layer to output
    - 3 weights for direct linear skip connections (t, x, v)
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    n = len(t)
    
    # Normalize inputs to roughly [-1, 1] range based on training ranges
    # t: [30, 50] -> center 40, scale 10
    # x: [-0.2056, 0.1749] -> center -0.0154, scale 0.19
    # v: [-0.3391, 0.3624] -> center 0.0117, scale 0.35
    t_norm = (t - 40.0) / 10.0
    x_norm = (x + 0.0154) / 0.19
    v_norm = (v - 0.0117) / 0.35
    
    # Fixed random weight matrix for hidden layer (7 units x 3 inputs)
    # Use a deterministic seed for reproducibility
    rng = np.random.RandomState(42)
    W_hidden = rng.randn(7, 3) * np.sqrt(2.0 / 3.0)
    b_hidden = rng.randn(7) * 0.1
    
    # Build input matrix: [t_norm, x_norm, v_norm]
    X = np.column_stack([t_norm, x_norm, v_norm])
    
    # Hidden layer: tanh(X @ W_hidden.T + b_hidden)
    Z = X @ W_hidden.T + b_hidden
    H = np.tanh(Z)
    
    # Output: linear combination of hidden units + skip connections
    # params[0:7] for hidden units, params[7:10] for skip connections
    pred = (H @ params[0:7] +
            t_norm * params[7] +
            x_norm * params[8] +
            v_norm * params[9])
    
    return pred
# EVOLVE-BLOCK-END
