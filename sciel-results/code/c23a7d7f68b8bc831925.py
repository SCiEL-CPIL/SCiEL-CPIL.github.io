# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
        # RBF Network with 3x3 grid of Gaussians
        # params[0] is log(inverse_variance) to ensure positive width
        # params[1:10] are the weights for the 9 Gaussian centers
        # Centers are at (-1, -1), (-1, 0), ..., (1, 1)
        
        # Inverse variance alpha = exp(params[0])
        # This allows the optimizer to adjust the kernel width
        # Starting at params[0]=1 -> alpha=e ~ 2.718 (narrower than before)
        # Wait, previous was alpha=1. 
        # If I want to start at alpha=1, I should start params[0] at 0.
        # But the evaluator starts params at all ones.
        # So params[0] = 1.
        # alpha = exp(1) = 2.718.
        # This is a narrower kernel than the previous alpha=1.
        # Is that good?
        # Narrower kernels are more local.
        # If the function is smooth, wider might be better.
        # If the function has details, narrower is better.
        # Since we don't know, let's trust the optimizer.
        # However, if the initial guess is too narrow, it might get stuck.
        # Maybe use `alpha = params[0]` and clip it?
        # Or `alpha = 1.0 + np.tanh(params[0])`? No, that limits range.
        # `alpha = np.exp(params[0] - 1.0)` would start at 1.0.
        # Let's do that. `alpha = np.exp(params[0] - 1.0)`.
        # If params[0]=1, alpha=1.
        # If params[0]=0, alpha=0.36.
        # If params[0]=2, alpha=2.71.
        
        alpha = np.exp(params[0] - 1.0)
        
        w0 = params[1]
        w1 = params[2]
        w2 = params[3]
        w3 = params[4]
        w4 = params[5]
        w5 = params[6]
        w6 = params[7]
        w7 = params[8]
        w8 = params[9]
        
        # Calculate Gaussians
        # x, y are 1D arrays
        
        # To improve speed, we can precompute x^2, y^2, x, y
        x2 = x * x
        y2 = y * y
        
        # g0: center (-1, -1)
        # r^2 = (x+1)^2 + (y+1)^2 = x^2+2x+1 + y^2+2y+1
        r2_00 = x2 + y2 + 2.0*x + 2.0*y + 2.0
        g0 = np.exp(-alpha * r2_00)
        
        # g1: center (-1, 0)
        # r^2 = (x+1)^2 + y^2 = x^2+2x+1 + y^2
        r2_01 = x2 + y2 + 2.0*x + 1.0
        g1 = np.exp(-alpha * r2_01)
        
        # g2: center (-1, 1)
        # r^2 = (x+1)^2 + (y-1)^2 = x^2+2x+1 + y^2-2y+1
        r2_02 = x2 + y2 + 2.0*x - 2.0*y + 2.0
        g2 = np.exp(-alpha * r2_02)
        
        # g3: center (0, -1)
        # r^2 = x^2 + (y+1)^2 = x^2 + y^2+2y+1
        r2_10 = x2 + y2 + 2.0*y + 1.0
        g3 = np.exp(-alpha * r2_10)
        
        # g4: center (0, 0)
        # r^2 = x^2 + y^2
        r2_11 = x2 + y2
        g4 = np.exp(-alpha * r2_11)
        
        # g5: center (0, 1)
        # r^2 = x^2 + (y-1)^2 = x^2 + y^2-2y+1
        r2_12 = x2 + y2 - 2.0*y + 1.0
        g5 = np.exp(-alpha * r2_12)
        
        # g6: center (1, -1)
        # r^2 = (x-1)^2 + (y+1)^2 = x^2-2x+1 + y^2+2y+1
        r2_20 = x2 + y2 - 2.0*x + 2.0*y + 2.0
        g6 = np.exp(-alpha * r2_20)
        
        # g7: center (1, 0)
        # r^2 = (x-1)^2 + y^2 = x^2-2x+1 + y^2
        r2_21 = x2 + y2 - 2.0*x + 1.0
        g7 = np.exp(-alpha * r2_21)
        
        # g8: center (1, 1)
        # r^2 = (x-1)^2 + (y-1)^2 = x^2-2x+1 + y^2-2y+1
        r2_22 = x2 + y2 - 2.0*x - 2.0*y + 2.0
        g8 = np.exp(-alpha * r2_22)
        
        return (w0 * g0 + w1 * g1 + w2 * g2 + 
                w3 * g3 + w4 * g4 + w5 * g5 + 
                w6 * g6 + w7 * g7 + w8 * g8)
    # EVOLVE-BLOCK-END
