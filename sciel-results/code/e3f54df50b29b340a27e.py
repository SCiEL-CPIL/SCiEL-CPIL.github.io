"""Symbolic regression with 1 input; Neural Network with 10 ReLU hidden units."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Neural Network with a single hidden layer of 10 ReLU units.
    
    The 10 parameters are interpreted as:
    - params[0:5]: Hidden layer weights (w1_0, w1_1, w1_2, w1_3, w1_4)
    - params[5:10]: Hidden layer biases (b1_0, b1_1, b1_2, b1_3, b1_4)
    
    Wait, that's only 10 params for 5 hidden units. We need output weights too.
    Let's reconsider the parameter allocation for a 1-10-1 network:
    - 10 weights (input to hidden)
    - 10 biases (hidden)
    - 10 weights (hidden to output)
    - 1 bias (output)
    Total: 31 params. But we only have 10.
    
    So we must use a smaller network or a different parameterization.
    
    Option A: 1-2-1 network (2 w1, 2 b1, 2 w2, 1 b2 = 7 params) + 3 extra params for something else?
    Option B: 1-3-1 network (3 w1, 3 b1, 3 w2, 1 b2 = 10 params)
    Option C: 1-4-1 network (4 w1, 4 b1, 4 w2, 1 b2 = 13 params) - too many
    
    The directive says "Neural Network with a single hidden layer (e.g., 10 ReLU or tanh units)".
    But with only 10 parameters, we cannot have 10 hidden units with full connectivity.
    
    Let's re-read the directive: "Use a Neural Network with a single hidden layer (e.g., 10 ReLU or tanh units) where the parameters define the hidden layer weights, biases, and output weights".
    
    This implies the parameters ARE the weights and biases. With 10 params, we can't have 10 hidden units.
    Perhaps the directive means a network where the TOTAL number of parameters is 10, and the architecture is chosen to fit that.
    
    Let's try a 1-3-1 tanh network, which uses exactly 10 parameters:
    - w1_0, w1_1, w1_2 (3 weights)
    - b1_0, b1_1, b1_2 (3 biases)
    - w2_0, w2_1, w2_2 (3 weights)
    - b2 (1 bias)
    Total: 10.
    
    This is a structural change from the current program (1-2-1 MLP + 3 RBF) to a pure 1-3-1 MLP.
    The inspiration with score 5.1947 used a 1-3-1 MLP and got a good score.
    The current program (1-2-1 MLP + 3 RBF) has score 4460.
    
    Let's try the pure 1-3-1 MLP as a structural change.
    """
    x = np.asarray(x, dtype=float).flatten()
    params = np.asarray(params, dtype=float).flatten()
    
    # Ensure exactly 10 parameters
    if len(params) < 10:
        params = np.concatenate([params, np.ones(10 - len(params))])
    else:
        params = params[:10]
    
    # Scale input x from [0.1, 6] to roughly [-1, 1] for numerical stability
    x_scaled = (x - 3.05) / 3.0
    
    # --- 1-3-1 Tanh MLP ---
    # params[0:3]: w1 (weights input to hidden)
    # params[3:6]: b1 (biases hidden)
    # params[6:9]: w2 (weights hidden to output)
    # params[9]:   b2 (output bias)
    
    w1 = params[0:3]
    b1 = params[3:6]
    w2 = params[6:9]
    b2 = params[9]
    
    # Hidden layer: tanh(x_scaled * w1 + b1)
    # Shape: (N, 3)
    hidden = np.tanh(np.outer(x_scaled, w1) + b1)
    
    # Output layer: hidden @ w2 + b2
    # Shape: (N,)
    y = hidden @ w2 + b2
    
    return y
# EVOLVE-BLOCK-END
