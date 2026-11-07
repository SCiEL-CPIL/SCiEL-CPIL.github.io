# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs (x, y) drawn from [-1, 1]^2.
    
    Strategy: Use a 2-hidden-layer neural network (MLP) with tanh activations.
    10 parameters are split as:
      - params[0:6] : weights from input (x, y) to first hidden layer (2 neurons)
                      [w_x1, w_y1, b1, w_x2, w_y2, b2]
      - params[6:10]: weights from first hidden layer to output
                      [w_1_out, w_2_out, bias_out, scale]
    
    Architecture:
      h1 = tanh(w_x1*x + w_y1*y + b1)
      h2 = tanh(w_x2*x + w_y2*y + b2)
      output = scale * (w_1_out * h1 + w_2_out * h2 + bias_out)
    
    This provides non-linear capacity to approximate a wider class of functions
    than a fixed linear basis. The scale parameter helps the optimizer adjust
    the overall magnitude.
    """
    p = params
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # First hidden layer (2 neurons)
    h1 = np.tanh(p[0] * x + p[1] * y + p[2])
    h2 = np.tanh(p[3] * x + p[4] * y + p[5])
    
    # Output layer
    output = p[8] * (p[6] * h1 + p[7] * h2 + p[9])
    
    return output
# EVOLVE-BLOCK-END
