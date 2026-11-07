"""Symbolic regression: an equation for E. coli growth rate."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with a small Multi-Layer Perceptron (MLP).
    
    This strategy uses a neural network representation where the 10 parameters
    are the weights of a single hidden layer. This is a structural departure
    from the previous additive basis expansion:
    
    1. Non-linear optimization: The problem is no longer convex/linear in params.
       BFGS will navigate a complex landscape, potentially finding better local
       minima that capture non-additive interactions.
    2. Flexible interaction: The hidden layer allows for arbitrary non-linear
       combinations of inputs (b, s, temp, pH), not just the specific terms
       (Monod, Gaussian, etc.) chosen in the previous approach.
    3. Universal approximation: Even with 1 hidden unit, it can approximate
       a wide range of functions better than a fixed 10-term polynomial-like
       basis if the basis functions were mismatched to the true data generation.
    
    Architecture:
    Input: [b, s, temp, pH] (4 features)
    Hidden: 1 neuron with tanh activation
    Output: 1 neuron (linear)
    
    Parameters mapping (10 total):
    p0: Input bias for hidden layer
    p1-p4: Weights from inputs [b, s, temp, pH] to hidden layer
    p5: Hidden layer bias
    p6: Weight from hidden layer to output
    p7: Output bias
    p8, p9: Auxiliary parameters (kept to satisfy length 10 constraint, 
            multiplied by 0 to not affect output, or used for scaling if needed.
            Here we just ignore them or use them as dummy to ensure length 10).
            Actually, to be safe and use all params effectively if possible, 
            we can make it 2 hidden units? No, 4 inputs + 1 bias = 5 params for W1+b1.
            1 hidden + 1 bias = 2 params for W2+b2. Total 7.
            Let's stick to 1 hidden unit to keep it simple and robust, 
            and map the extra 3 params to 0 or use them for a second hidden unit?
            
    Let's use 2 hidden units to utilize all 10 parameters more meaningfully?
    Input (4) -> Hidden (2) -> Output (1)
    W1 (4x2) = 8 params
    b1 (2) = 2 params
    W2 (2x1) = 2 params
    b2 (1) = 1 param
    Total = 13 params. Too many.
    
    Back to 1 hidden unit:
    W1 (4x1) = 4 params
    b1 (1) = 1 param
    W2 (1x1) = 1 param
    b2 (1) = 1 param
    Total = 7 params.
    
    We have 10 params. Let's use 1 hidden unit and add 3 "dummy" params that are 
    multiplied by 0, OR we can use them to scale the inputs? 
    Scaling inputs is risky for BFGS stability.
    Let's just use the first 7 and ignore the last 3, or better, use a 
    slightly more complex structure that fits exactly 10.
    
    How about 1 hidden unit, but we allow the input to be transformed?
    No, keep it simple. 
    Let's use 2 hidden units but with shared weights? No.
    
    Alternative: 
    Input (4) -> Hidden (1) -> Output (1)
    Params:
    p0: w_b
    p1: w_s
    p2: w_temp
    p3: w_ph
    p4: b_hidden
    p5: w_out
    p6: b_out
    p7, p8, p9: Unused (set to 0 in effect)
    
    To ensure BFGS doesn't get confused by unused params, we can multiply them by 0.0.
    """
    # Unpack params
    w_b, w_s, w_temp, w_ph, b_h, w_o, b_o, p7, p8, p9 = params
    
    # Normalize inputs to help optimization (center and scale roughly)
    # b range [1, 4.998] -> center ~3, scale ~2
    # s range [1, 4.999] -> center ~3, scale ~2
    # temp range [1.002, 39.98] -> center ~20, scale ~20
    # pH range [2, 12] -> center ~7, scale ~5
    
    # Simple normalization
    b_norm = (b - 3.0) / 2.0
    s_norm = (s - 3.0) / 2.0
    temp_norm = (temp - 20.0) / 20.0
    ph_norm = (pH - 7.0) / 5.0
    
    # Hidden layer (1 unit, tanh)
    # z = w_b*b_norm + w_s*s_norm + w_temp*temp_norm + w_ph*ph_norm + b_h
    z = w_b * b_norm + w_s * s_norm + w_temp * temp_norm + w_ph * ph_norm + b_h
    h = np.tanh(z)
    
    # Output layer (linear)
    y = w_o * h + b_o
    
    # Add unused params multiplied by 0 to ensure they are part of the graph 
    # (though mathematically they don't change anything, this ensures the 
    # function signature uses all inputs if strict checking is done, 
    # though typically unused variables are fine in Python).
    # However, to be safe against any "unused variable" linters or to 
    # ensure the optimizer sees them as part of the cost function if it 
    # tracks gradients for all params (it does), we add them as 0.
    y = y + 0.0 * (p7 + p8 + p9)
    
    return y
# EVOLVE-BLOCK-END
