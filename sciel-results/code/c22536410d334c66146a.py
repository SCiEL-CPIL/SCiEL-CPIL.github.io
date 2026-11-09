import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Basis functions
    # 1. Constant
    phi1 = np.ones_like(x)
    # 2. Linear
    phi2 = x
    # 3. Quadratic
    phi3 = x**2
    # 4. Cubic
    phi4 = x**3
    # 5. Exponential decay
    phi5 = np.exp(-0.5 * x)
    # 6. Exponential growth (scaled)
    phi6 = np.exp(0.5 * x)
    # 7. Sine
    phi7 = np.sin(x)
    # 8. Cosine
    phi8 = np.cos(x)
    # 9. Inverse
    phi9 = 1.0 / (x + 1.0)  # Avoid singularity at 0, x>=0.1
    # 10. Log
    phi10 = np.log(1.0 + x)

    # Linear combination
    return (params[0] * phi1 + 
            params[1] * phi2 + 
            params[2] * phi3 + 
            params[3] * phi4 + 
            params[4] * phi5 + 
            params[5] * phi6 + 
            params[6] * phi7 + 
            params[7] * phi8 + 
            params[8] * phi9 + 
            params[9] * phi10)
