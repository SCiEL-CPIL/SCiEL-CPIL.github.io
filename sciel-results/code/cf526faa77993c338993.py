"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    
    # Deadline guard: 150s limit, leave 10s margin for safety
    t0 = time.time()
    deadline = t0 + 135.0
    
    # Initial parameter vector: 10 weights + 10 centers
    # The evaluator passes 10 params (all ones). We need 20.
    # We interpret the 10 input params as initial weights.
    # We initialize centers to a spread grid.
    
    # If params length is not 20, we resize.
    # The contract says params is length 10.
    # We will use the 10 params as weights.
    # We will generate centers internally.
    
    if len(params) == 10:
        w_init = params
        # Initial centers: evenly spaced
        c_init = np.linspace(-2.0, 2.0, 10)
        theta_init = np.concatenate([w_init, c_init])
    elif len(params) == 20:
        theta_init = params
    else:
        # Fallback
        theta_init = np.ones(20)
        
    # Fixed width for stability
    sigma = 0.4
    
    def compute_rbf(theta, x_eval):
        w = theta[:10]
        c = theta[10:]
        # RBF: sum w_i * exp(-((x - c_i)^2) / (2 * sigma^2))
        # Vectorized
        # x_eval shape (N,), c shape (10,)
        # diff shape (N, 10)
        diff = x_eval[:, None] - c[None, :]
        rbf = np.exp(-diff**2 / (2 * sigma**2))
        return rbf @ w

    # Initial prediction
    pred = compute_rbf(theta_init, x)
    
    # We don't have y, so we can't compute MSE directly.
    # However, the evaluator minimizes MSE. 
    # Wait, the `equation` function is called *during* the optimization?
    # No, the evaluator calls `equation(x, params)` to get predictions, 
    # then computes MSE against hidden y.
    # The evaluator uses BFGS to fit `params`.
    # I cannot change the optimizer the evaluator uses.
    # I can only change the function `equation` computes.
    #
    # Re-reading the contract:
    # "The evaluator fits params ... with scipy.optimize.minimize(method="BFGS")"
    # This means I CANNOT run my own optimizer inside `equation` to find better params 
    # for the *current* x, because `equation` is just the model.
    # The evaluator calls `equation(x, params)` repeatedly with different `params` 
    # to evaluate the loss landscape for BFGS.
    #
    # Therefore, any "search" I do inside `equation` would have to be 
    # deterministic given `x` and `params`, or it would break the gradient 
    # estimation of the outer BFGS loop (if I do stochastic search, 
    # the loss surface becomes noisy/non-differentiable).
    #
    # If I do a full Nelder-Mead inside `equation`, it will:
    # 1. Be extremely slow (calling `equation` many times, each doing N-M).
    # 2. Produce non-differentiable jumps in the loss landscape, 
    #    likely breaking the outer BFGS.
    #
    # So, I must stick to a fixed, smooth, differentiable function of `params`.
    #
    # The current RBF (score ~939) is convex in weights. 
    # The Padé (score ~61393) is non-convex.
    #
    # To improve on 939, I need a better basis or a way to handle 
    # non-convexity that BFGS can navigate from all-ones start.
    #
    # Idea: Use a basis that is more flexible than fixed RBFs but still well-behaved.
    # Maybe a Fourier series? 
    # Or a Neural Network with fixed random weights (Nips)?
    # A 2-layer NN with fixed random weights is linear in the output layer weights.
    # If I fix the hidden layer weights and biases, the function is linear in the 10 output weights.
    # This is convex! And much more flexible than 10 RBFs if the hidden layer is large.
    # But I only have 10 params.
    # If I use 10 params for output weights, I can have, say, 50 hidden units 
    # with fixed random weights.
    # This gives a 10-dimensional linear subspace of functions spanned by 
    # 50 fixed random sigmoid activations.
    # This is strictly convex in the 10 params.
    #
    # Let's try this: Random Feature Network.
    # Fixed random matrix W (100 x 10)? No, I need 10 params total.
    # So, I have 10 output weights.
    # I can define 10 fixed "basis functions" which are actually 
    # complex random feature maps.
    # e.g., phi_i(x) = cos(w_i * x + b_i) + sin(w_i * x + b_i)
    # where w_i, b_i are fixed constants.
    # This is a linear model: y = sum w_i * phi_i(x).
    # Convex.
    #
    # Why might this beat RBF?
    # RBFs are localized. Fourier features are global.
    # If the function is smooth and global (like sin, poly), Fourier might be better.
    # If the function has sharp local features, RBF is better.
    #
    # Let's try a Hybrid: 
    # 5 params for a low-order polynomial (global trend).
    # 5 params for RBFs (local corrections).
    #
    # Polynomial: 1, x, x^2, x^3, x^4.
    # RBFs: 5 centers fixed at -2, -1, 0, 1, 2.
    #
    # This is still linear in params -> Convex.
    #
    # Let's implement this Hybrid Basis.
    
    # Basis 1: Polynomial terms (indices 0-4)
    p_coeffs = theta_init[:5] # Just for init, we use params directly in the function below
    
    # Basis 2: RBF centers (indices 5-9)
    rbf_centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # Re-evaluate: The `params` passed in are the ones being optimized.
    # I should just define the basis functions and return the linear combination.
    
    # Extract current params
    p = params[:5]
    r = params[5:10]
    
    # Polynomial part
    x0 = np.ones_like(x)
    x1 = x
    x2 = x**2
    x3 = x**3
    x4 = x**4
    
    poly_part = p[0]*x0 + p[1]*x1 + p[2]*x2 + p[3]*x3 + p[4]*x4
    
    # RBF part
    # Fixed centers
    # Fixed width
    sigma = 0.5
    rbf_vals = np.zeros_like(x)
    for i, c in enumerate(rbf_centers):
        rbf_vals += r[i] * np.exp(-(x - c)**2 / (2 * sigma**2))
        
    return poly_part + rbf_vals

# EVOLVE-BLOCK-END
