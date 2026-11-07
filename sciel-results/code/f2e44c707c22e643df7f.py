"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate using a Neural Network (MLP) representation.
    
    Strategy:
    Instead of a fixed linear basis (like RBFs or Polynomials), we interpret the 10 
    parameters as the weights of a small neural network. Specifically, we use a 
    single-layer network with 10 hidden units (tanh activation) and a single output 
    unit. The parameters p[0:10] are the input-to-hidden weights. The hidden-to-output 
    weights are fixed to 1.0 to keep the parameter count at 10.
    
    This approach is structurally different because:
    1. It creates a non-linear relationship between params and prediction (due to tanh).
    2. It allows the model to represent functions that are not in the linear span of 
       the fixed basis functions used in previous attempts.
    3. BFGS can still optimize this, although the landscape is non-convex. The 
       initialization at ones often provides a decent starting point for smooth functions.
    
    Architecture:
    Input x (scalar) -> Weights p[0:10] -> Hidden layer (10 units, tanh) -> Output (sum of hidden)
    
    Wait, if output is just sum of hidden, that's equivalent to a linear model with 
    non-linear basis. To make it more powerful, we need a non-linear output combination 
    or more layers. However, with only 10 params, we are constrained.
    
    Let's try a different structure:
    2 hidden layers?
    Layer 1: 5 units. Weights p[0:5]. Bias 0. Activation tanh.
    Layer 2: 5 units. Weights p[5:10]. Bias 0. Activation tanh.
    Output: Sum of Layer 2 units.
    
    This gives depth. The non-linearity is in the composition of tanhs.
    
    Let's verify the parameter mapping:
    p[0:5] -> Weights for first layer (x -> h1)
    p[5:10] -> Weights for second layer (h1 -> h2)
    Output = sum(h2)
    
    This is a valid 10-parameter non-linear model.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Ensure p has length 10
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), mode='constant')
    else:
        p = p[:10]
        
    # Split parameters
    w1 = p[0:5]  # Weights for first layer
    w2 = p[5:10] # Weights for second layer
    
    # Compute first hidden layer
    # x is (N,), w1 is (5,)
    # h1 is (N, 5)
    h1 = x[:, np.newaxis] * w1[np.newaxis, :]
    h1 = np.tanh(h1)
    
    # Compute second hidden layer
    # h1 is (N, 5), w2 is (5,)
    # h2 is (N, 5)
    # Each unit in h2 is dot product of h1 and w2? 
    # If w2 is just 5 scalars, then h2[i] = sum(h1[i, :] * w2).
    # This would make all units in h2 identical if w2 is the same vector.
    # We need w2 to be a matrix (5x5) to have distinct units, but we only have 5 params left.
    
    # Alternative:
    # Use the 10 params as:
    # 5 centers for RBFs? No, that's what we had.
    
    # Let's try a "Deep RBF" or similar.
    # Or, simply a higher-order polynomial with non-linear parameterization?
    
    # Actually, the prompt says "Different design or modelling family".
    # Neural net with 10 params is very small.
    
    # Let's try a **Fourier Series** with **learnable frequencies**.
    # But BFGS starts from 1. If frequencies are params, the landscape is very rugged.
    
    # Let's stick to the **MLP** idea but structure it to use 10 params effectively.
    # Structure:
    # Layer 1: 10 units. Input x. Weights are fixed to [-1, 1] spread? No, weights must be params.
    
    # How about:
    # y = sum_{i=1}^{10} p_i * tanh(x * c_i)
    # Where c_i are fixed constants? That's just a fixed basis with non-linear activation.
    # That is a "Neural Basis". It is structurally different from RBFs (Gaussian) and Polynomials.
    # It is linear in params, so BFGS converges reliably (convex).
    # The basis functions are tanh(cx). These are sigmoid-like step functions.
    # They can approximate piecewise linear functions well.
    
    # Let's use this: **Tanh Basis Functions** with fixed centers/scales.
    # This is a "Neural Network with linear output and fixed hidden weights".
    # It is different from RBFs (local vs global step) and Polynomials (smooth vs kinked).
    
    # Define 10 fixed scales/centers for the tanh basis.
    # To cover [-3, 3], we want transitions across the domain.
    # tanh(x * scale + shift)
    
    # Let's use 10 different scales and shifts.
    # scales = np.linspace(0.5, 3.0, 10)
    # shifts = np.linspace(-2.0, 2.0, 10)
    
    # This creates a rich set of non-linear basis functions.
    
    scales = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0])
    shifts = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    
    # Compute basis
    # shape (N, 10)
    basis = np.tanh(x[:, np.newaxis] * scales[np.newaxis, :] + shifts[np.newaxis, :])
    
    # Prediction
    pred = basis @ p
    
    # Safety check
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
