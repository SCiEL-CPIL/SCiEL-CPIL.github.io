import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure x is float
    x = np.asarray(x, dtype=np.float64)
    
    # Compute basis functions
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = x2
    b3 = x3
    b4 = x4
    b5 = np.sin(x)
    b6 = np.cos(x)
    b7 = np.exp(x)
    b8 = np.exp(-x2)
    b9 = np.tanh(x)
    
    # Linear combination
    # Using dot product or manual sum for clarity and speed
    # params is shape (10,)
    # We can stack basis into a matrix (N, 10) and dot with params
    # But manual sum is also fast and avoids memory overhead for small N
    y = (params[0] * b0 + 
         params[1] * b1 + 
         params[2] * b2 + 
         params[3] * b3 + 
         params[4] * b4 + 
         params[5] * b5 + 
         params[6] * b6 + 
         params[7] * b7 + 
         params[8] * b8 + 
         params[9] * b9)
         
    return y
