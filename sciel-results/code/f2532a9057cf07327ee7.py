import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a hybrid basis of polynomials, 
    exponentials, trigonometric functions, and rational terms.
    
    The input x is scaled to u in [-1, 1] to improve numerical stability 
    and condition the basis functions.
    """
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=float)
    
    # Scale x from [0.1, 6] to [-1, 1]
    # Center = 3.05, Half-Range = 2.95
    u = (x - 3.05) / 2.95
    
    # Define 10 basis functions
    # 1. Constant
    b0 = np.ones_like(u)
    
    # 2. Linear
    b1 = u
    
    # 3. Quadratic
    b2 = u ** 2
    
    # 4. Cubic
    b3 = u ** 3
    
    # 5. Exponential growth
    b4 = np.exp(u)
    
    # 6. Exponential decay
    b5 = np.exp(-u)
    
    # 7. Sine
    b6 = np.sin(u)
    
    # 8. Cosine
    b9 = np.cos(u)
    
    # 9. Hyperbolic tangent (smooth step/saturation)
    b8 = np.tanh(u)
    
    # 10. Lorentzian (peak at u=0)
    b9 = 1.0 / (1.0 + u**2)
    
    # Combine basis functions with parameters
    # y = p0*b0 + p1*b1 + ... + p9*b9
    y = (params[0] * b0 + 
         params[1] * b1 + 
         params[2] * b2 + 
         params[3] * b3 + 
         params[4] * b4 + 
         params[5] * b5 + 
         params[6] * b6 + 
         params[7] * b8 + 
         params[8] * b9 + 
         params[9] * np.ones_like(u)) # Wait, I reused b9 index. Let's fix indices.
         
    # Correct combination:
    y = (params[0] * b0 + 
         params[1] * b1 + 
         params[2] * b2 + 
         params[3] * b3 + 
         params[4] * b4 + 
         params[5] * b5 + 
         params[6] * b6 + 
         params[7] * np.cos(u) + # Using cos for index 7
         params[8] * b8 +        # tanh for index 8
         params[9] * b9)         # Lorentzian for index 9

    return y
