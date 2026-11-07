# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 130.0  # Safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model combining:
    - A 1-4-1 tanh MLP (7 params: 2 w1, 2 b1, 2 w2, 1 b2)
    - 3 RBF terms with fixed centers (3 params: coefficients)
    
    Total: 10 parameters
    
    The input x is in [0.1, 6]. RBF centers at 1.5, 3.0, 4.5 with sigma=1.5.
    MLP uses scaled input for stability.
    """
    x = np.asarray(x, dtype=float).flatten()
    params = np.asarray(params, dtype=float).flatten()
    
    # Ensure exactly 10 parameters
    if len(params) < 10:
        params = np.concatenate([params, np.ones(10 - len(params))])
    else:
        params = params[:10]
    
    # Scale input x from [0.1, 6] to roughly [-1, 1]
    x_scaled = (x - 3.05) / 3.0
    
    # --- MLP Part (1-4-1) ---
    # Params [0:2] = w1 for first 2 hidden units
    # Params [2:4] = b1 for first 2 hidden units
    # Params [4:6] = w2 for first 2 hidden units to output
    # Params [6]   = output bias b2
    
    # We only have 7 params for a 1-4-1, so let's do 1-3-1 instead
    # Params [0:2]: w1[0], w1[1] for 2 hidden units
    # Actually, let's restructure: 1-3-1 MLP = 3*2 + 3 = 9 params, too many
    # Let's do 1-2-2-1: 2+2 (first layer) + 2*2+2 (second layer) = 10? No
    # 
    # Let's stick with: 1-2-1 MLP (5 params) + 5 RBF coefficients
    # Or: 1-2-1 MLP (5 params) + 3 RBF + 2 extra smooth basis
    #
    # Better plan: Use a richer linear-in-params basis with 10 terms
    # This is linear in params, so BFGS should converge well
    
    # Basis functions:
    # 0: constant
    # 1: x
    # 2: x^2
    # 3: x^3
    # 4: x^4
    # 5: x^5
    # 6: sin(2*pi*x/6)
    # 7: cos(2*pi*x/6)
    # 8: exp(-((x-1.5)^2)/(2*1.5^2))
    # 9: exp(-((x-4.5)^2)/(2*1.5^2))
    
    c = np.ones_like(x)
    b1 = x
    b2 = x ** 2
    b3 = x ** 3
    b4 = x ** 4
    b5 = x ** 5
    b6 = np.sin(2.0 * np.pi * x / 6.0)
    b7 = np.cos(2.0 * np.pi * x / 6.0)
    b8 = np.exp(-((x - 1.5) ** 2) / (2.0 * 1.5 ** 2))
    b9 = np.exp(-((x - 4.5) ** 2) / (2.0 * 1.5 ** 2))
    
    # Stack basis: shape (n, 10)
    basis = np.column_stack([c, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Linear combination
    y = basis @ params
    
    return y
# EVOLVE-BLOCK-END
