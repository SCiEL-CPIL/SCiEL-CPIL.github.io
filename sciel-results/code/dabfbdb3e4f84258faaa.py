import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation using a 1-3-1 Multilayer Perceptron with Softplus activation.
    y = sum_{i=1}^{3} w2_i * softplus(w1_i * x + b1_i) + b2
    
    Parameters:
    p0: w1_1, p1: b1_1
    p2: w1_2, p3: b1_2
    p4: w1_3, p5: b1_3
    p6: w2_1, p7: w2_2, p8: w2_3
    p9: b2
    """
    x = np.asarray(x, dtype=float)
    
    # Unpack parameters
    w1_1, b1_1 = params[0], params[1]
    w1_2, b1_2 = params[2], params[3]
    w1_3, b1_3 = params[4], params[5]
    w2_1, w2_2, w2_3 = params[6], params[7], params[8]
    b2 = params[9]
    
    # Calculate hidden layer activations (Softplus for smooth non-linearity)
    # Using np.log1p(np.exp(z)) for numerical stability
    def softplus(z):
        return np.log1p(np.exp(np.clip(z, -50, 50)))
        
    z1 = w1_1 * x + b1_1
    z2 = w1_2 * x + b1_2
    z3 = w1_3 * x + b1_3
    
    h1 = softplus(z1)
    h2 = softplus(z2)
    h3 = softplus(z3)
    
    # Output layer
    y = w2_1 * h1 + w2_2 * h2 + w2_3 * h3 + b2
    
    return y
