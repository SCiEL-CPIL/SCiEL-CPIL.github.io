import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
        # Compute basis functions
        # Polynomials
        x2 = x**2
        x3 = x2 * x
        x4 = x2 * x2
        x5 = x4 * x
        
        # Trig
        sin_x = np.sin(x)
        cos_x = np.cos(x)
        sin_2x = np.sin(2*x)
        cos_2x = np.cos(2*x)
        
        # Linear combination
        y = (params[0] * 1 + 
             params[1] * x + 
             params[2] * x2 + 
             params[3] * x3 + 
             params[4] * x4 + 
             params[5] * x5 + 
             params[6] * sin_x + 
             params[7] * cos_x + 
             params[8] * sin_2x + 
             params[9] * cos_2x)
             
        return y
