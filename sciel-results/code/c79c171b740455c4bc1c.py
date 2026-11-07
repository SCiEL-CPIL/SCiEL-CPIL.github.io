# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Mixed basis functions to capture various functional forms
    # 1. Constant
    b0 = 1.0
    # 2. Linear
    b1 = x
    # 3. Quadratic
    b2 = x * x
    # 4. Cubic
    b3 = b2 * x
    # 5. Sine
    b4 = np.sin(x)
    # 6. Cosine
    b5 = np.cos(x)
    # 7. Hyperbolic tangent (saturating)
    b6 = np.tanh(x)
    # 8. Double frequency sine
    b7 = np.sin(2.0 * x)
    # 9. Double frequency cosine
    b8 = np.cos(2.0 * x)
    # 10. Modulated sine
    b9 = x * np.sin(x)

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
# EVOLVE-BLOCK-END
