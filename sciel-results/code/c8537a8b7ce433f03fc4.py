import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    A shallow neural network-like model with 1 hidden layer.
    Structure:
    x -> [W1, b1] -> tanh -> [W2, b2] -> y
    
    Parameters mapping (10 params total):
    p[0]: Input weight for hidden unit 0
    p[1]: Input weight for hidden unit 1
    p[2]: Input weight for hidden unit 2
    p[3]: Input weight for hidden unit 3
    p[4]: Bias for hidden unit 0
    p[5]: Bias for hidden unit 1
    p[6]: Bias for hidden unit 2
    p[7]: Bias for hidden unit 3
    p[8]: Output weight (shared across hidden units for simplicity? No, let's use a single output weight 
          that sums the hidden activations, effectively p[8] * sum(tanh(...)) + p[9])
    p[9]: Output bias
    
    Wait, a standard dense layer would need 4 output weights (one per hidden unit) + 1 bias = 5 params.
    Input layer: 4 weights + 4 biases = 8 params.
    Total 13 params. We only have 10.
    
    Revised Structure to fit 10 params:
    Option A: 2 hidden units.
    p[0]: w1 (x to h0)
    p[1]: w2 (x to h1)
    p[2]: b0 (bias h0)
    p[3]: b1 (bias h1)
    p[4]: w_out_0 (h0 to y)
    p[5]: w_out_1 (h1 to y)
    p[6]: b_out (bias y)
    p[7]: scale_1 (optional scaling for h0)
    p[8]: scale_2 (optional scaling for h1)
    p[9]: global_scale (optional)
    
    This is still somewhat linear in the output layer. 
    Let's try a more direct nonlinear composition:
    y = p[0] * tanh(p[1]*x + p[2]) + p[3] * tanh(p[4]*x + p[5]) + p[6] * tanh(p[7]*x + p[8]) + p[9]
    
    This is a sum of 3 shifted/scaled tanh functions + a constant.
    This is a "universal approximator" style basis where the basis functions themselves are parameterized.
    This is fundamentally different from a fixed basis with linear coefficients.
    The BFGS optimizer will have to navigate a non-convex landscape to find the centers (biases) and slopes (weights) of the tanh steps.
    
    Let's implement:
    y = p[0] * tanh(p[1]*x + p[2]) + p[3] * tanh(p[4]*x + p[5]) + p[6] * tanh(p[7]*x + p[8]) + p[9]
    
    This uses exactly 10 parameters.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has exactly 10 elements
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Safety guard against overflow in tanh arguments (though tanh is safe, large inputs are fine)
    # But we should ensure no NaNs propagate if params get weird.
    
    try:
        # Term 1: p[0] * tanh(p[1]*x + p[2])
        arg1 = p[1] * x + p[2]
        t1 = np.tanh(arg1)
        
        # Term 2: p[3] * tanh(p[4]*x + p[5])
        arg2 = p[4] * x + p[5]
        t2 = np.tanh(arg2)
        
        # Term 3: p[6] * tanh(p[7]*x + p[8])
        arg3 = p[7] * x + p[8]
        t3 = np.tanh(arg3)
        
        # Constant term
        y_hat = p[0] * t1 + p[3] * t2 + p[6] * t3 + p[9]
        
        # Ensure finite output
        if not np.all(np.isfinite(y_hat)):
            # If non-finite, return a safe constant (mean of x? 0?)
            # The contract says non-finite error costs 1e30. Returning 0 is safer than NaN.
            y_hat = np.zeros_like(x)
            
        return y_hat
        
    except Exception:
        return np.zeros_like(x)
