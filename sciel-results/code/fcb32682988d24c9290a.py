"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    1-hidden-layer Neural Network with tanh activation.
    
    Architecture:
    Inputs: x, y (normalized to [-1, 1] which they already are)
    Hidden Layer: 10 units, tanh activation
    Output: 1 unit, linear activation
    
    Parameter Mapping (10 params total):
    params[0:5]  -> Weights for first 5 hidden units (x, y, bias)
    params[5:10] -> Weights for last 5 hidden units (x, y, bias)
    
    Note: To strictly adhere to the 10-parameter constraint while maintaining
    expressiveness, we use a fixed, symmetric initialization strategy for the
    second set of weights or use a specific factorization. However, a standard
    10-param NN is tricky.
    
    Alternative Strategy: 
    Use a 5-unit hidden layer with 2 inputs + bias = 3 weights/unit = 15 weights. 
    This exceeds 10.
    
    Let's use a 3-unit hidden layer? 3*(2+1) = 9 weights for hidden, 1 for output = 10.
    This is very small.
    
    Let's try a different structure: 
    2 hidden units.
    Unit 1: w11*x + w12*y + w13 -> tanh -> h1
    Unit 2: w21*x + w22*y + w23 -> tanh -> h2
    Output: w31*h1 + w32*h2 + w33
    
    Total params: 3 + 3 + 3 = 9. We have 10.
    Add another input feature? No, only x,y.
    
    Let's use 4 hidden units?
    4 * 3 = 12 weights. Too many.
    
    Let's go back to the Inspiration 1 (RBF) which scored 66.8. 
    The current score is ~3.5e8 (terrible). The seed (polynomial) was 1.0.
    Wait, the prompt says "CURRENT PROGRAM (score 354096458.3157)". 
    This implies the cubic polynomial FAILED or diverged catastrophically.
    
    The "Knowledge from this run" states: "Bounded tanh-based model is well-conditioned 
    for BFGS from all-ones... Measured: 1 -> 6.837e+05."
    This suggests a tanh model improved things significantly but is still far from 1.0.
    
    Why is it still so bad? 
    If the target is something like sin(10*x), a simple NN with 10 params might not have enough capacity.
    
    Let's look at Inspiration 3: "multi-scale Fourier basis". Score 0.9908.
    This is extremely close to 1.0 (perfect).
    
    The task asks for a strategy DIFFERENT from the current (Cubic Poly).
    Inspiration 3 (Fourier) is different. 
    However, the current program is the Cubic Poly.
    
    Let's implement a **Fourier Basis** approach, which is structurally different from 
    polynomials and RBFs, and is known to handle periodic/smooth functions well on 
    bounded domains like [-1,1].
    
    Basis:
    1 (constant)
    cos(k * x), sin(k * x)
    cos(k * y), sin(k * y)
    cos(k * (x+y)), sin(k * (x+y))
    
    With 10 params, we can afford:
    1. Constant (1)
    2. cos(x), sin(x) (2)
    3. cos(y), sin(y) (2)
    4. cos(x+y), sin(x+y) (2)
    5. cos(2x), sin(2x) (2)
    
    Total: 1 + 2 + 2 + 2 + 2 = 9.
    Add 10th: cos(2y) or sin(2y) or a cross term like cos(x-y).
    
    Let's use:
    p0: 1
    p1: cos(x)
    p2: sin(x)
    p3: cos(y)
    p4: sin(y)
    p5: cos(x+y)
    p6: sin(x+y)
    p7: cos(x-y)
    p8: sin(x-y)
    p9: cos(2*x + 2*y)  # Higher frequency term
    
    This spans a wide range of smooth, periodic-like behaviors which are common 
    in synthetic function recovery tasks (like sin, cos, exp approximations).
    """
    p = params
    
    # Pre-compute trigonometric terms for efficiency and clarity
    cx = np.cos(x)
    sx = np.sin(x)
    cy = np.cos(y)
    sy = np.sin(y)
    
    # Linear combinations of arguments
    s_xy = np.sin(x + y)
    c_xy = np.cos(x + y)
    s_xm = np.sin(x - y)
    c_xm = np.cos(x - y)
    
    # Higher frequency
    c_2xy = np.cos(2 * x + 2 * y)
    
    pred = (
        p[0]
        + p[1] * cx
        + p[2] * sx
        + p[3] * cy
        + p[4] * sy
        + p[5] * c_xy
        + p[6] * s_xy
        + p[7] * c_xm
        + p[8] * s_xm
        + p[9] * c_2xy
    )
    
    return pred
# EVOLVE-BLOCK-END
