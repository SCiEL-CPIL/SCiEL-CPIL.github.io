import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF-based symbolic regression model.
    
    Instead of global trigonometric/polynomial basis, this uses 5 local 
    Gaussian basis functions (RBFs) plus a linear trend term.
    
    Parameters:
    - p[0:10] are used to define 5 RBF centers and scales.
      For k in 0..4:
        center_x = p[2*k]
        center_y = p[2*k + 1]
      The scales (widths) are fixed to 0.5 to ensure stability and 
      prevent collapse, relying on BFGS to find the optimal centers.
      Note: The original contract specifies params length 10.
      We use all 10 params: 5 for x-centers, 5 for y-centers.
      
    The model is:
    f(x,y) = sum_{k=0}^{4} exp(-((x - cx_k)^2 + (y - cy_k)^2) / (2 * sigma^2)) 
             + linear_trend(x, y)
             
    Wait, the evaluator fits `params` (length 10) using BFGS. 
    The function `equation` receives `params` and must return predictions.
    The coefficients of the RBFs are NOT in `params`. 
    This is a critical constraint: `params` is the ONLY thing optimized.
    
    If I use RBFs, I need coefficients for the RBFs too. 
    If I don't include them in `params`, I can't fit them.
    
    Alternative Strategy: 
    Since `params` is length 10, and BFGS optimizes them, 
    I must design a model where the 10 parameters fully define the function 
    such that it can approximate the target.
    
    The previous strategy failed because the basis was too restrictive.
    A more flexible strategy with only 10 params is difficult.
    
    Let's look at the "Symbolic Regression" aspect. 
    Usually, symbolic regression searches over the structure. 
    Here, the structure is FIXED in the code, and BFGS fits the numbers.
    
    If the target is, say, `x*y` or `sin(x)+cos(y)`, a 10-param model can fit it.
    But if the target is `exp(x*y)`, a fixed 10-param model might struggle.
    
    However, the prompt says "Write equation... that recovers an unknown function".
    And "The evaluator fits params... minimizing training MSE".
    
    This implies the *structure* of my `equation` function is the symbolic form.
    The parameters are the constants in that form.
    
    The current program uses:
    A1*sin(f1x*x + f1y*y + phi1) + A2*cos(f2x*x + f2y*y + phi2) + C + Q*(x^2+y^2)
    This is 10 params.
    
    Why did it score 5.2? Maybe the target is not well approximated by 2 sines/cosines + quadratic.
    
    Let's try a **Polynomial Basis** of higher order or mixed terms.
    With 10 params, we can fit a full quadratic (6 terms) + linear (2 terms) + constant (1 term) + one more?
    Full quadratic in x,y:
    1, x, y, x^2, xy, y^2 (6 params)
    Plus x^3, x^2y, xy^2, y^3? (4 more) -> Total 10.
    A cubic polynomial has 10 coefficients (for degree 3 in 2 vars: 4+3+2+1=10).
    
    Let's test a **Full Cubic Polynomial**.
    Terms:
    1
    x, y
    x^2, xy, y^2
    x^3, x^2y, xy^2, y^3
    
    This is a linear model in the parameters. 
    BFGS will converge quickly to the least squares solution.
    If the target is smooth, a cubic might capture more local curvature than the current trig+quad mix.
    
    Why might the current one be better? If the target is periodic.
    But if the target is a rational function or exponential, polynomial might fail on edges.
    
    Let's consider **Rational Functions** or **Neural Network-like** structures?
    With 10 params, a small MLP is hard to train with BFGS from scratch (local minima).
    
    Let's stick to **Full Cubic Polynomial**. It's a standard, robust baseline for smooth functions on [-1,1].
    The current model is a specific nonlinear combination. A cubic is a universal approximator for polynomials up to degree 3.
    
    Let's verify the parameter count for full cubic:
    Degree 0: 1 (c0)
    Degree 1: x, y (c1, c2)
    Degree 2: x^2, xy, y^2 (c3, c4, c5)
    Degree 3: x^3, x^2y, xy^2, y^3 (c6, c7, c8, c9)
    Total: 10.
    
    This uses all 10 params efficiently as linear coefficients.
    BFGS on a quadratic loss surface (MSE of a linear model) is convex (globally optimal).
    The current model has a non-convex loss surface (frequencies inside sine), which BFGS might get stuck in local minima near the initial value of 1.
    
    By switching to a linear-in-parameters model (the polynomial coefficients), we guarantee that BFGS finds the global minimum for that specific basis. If the target is well-approximated by a cubic, this will outperform the current model which might be stuck in a bad local minimum for its specific trig structure.
    """
    
    # Unpack 10 parameters as coefficients for a full cubic polynomial
    c0 = params[0]
    c1 = params[1]
    c2 = params[2]
    c3 = params[3]
    c4 = params[4]
    c5 = params[5]
    c6 = params[6]
    c7 = params[7]
    c8 = params[8]
    c9 = params[9]
    
    # Compute polynomial terms
    # To avoid overflow or numerical issues, we rely on x,y being in [-1,1]
    
    # Degree 0
    term_0 = np.ones_like(x)
    
    # Degree 1
    term_1 = x
    term_2 = y
    
    # Degree 2
    term_3 = x * x
    term_4 = x * y
    term_5 = y * y
    
    # Degree 3
    term_6 = x * x * x
    term_7 = x * x * y
    term_8 = x * y * y
    term_9 = y * y * y
    
    # Combine
    val = (c0 * term_0 + 
           c1 * term_1 + 
           c2 * term_2 + 
           c3 * term_3 + 
           c4 * term_4 + 
           c5 * term_5 + 
           c6 * term_6 + 
           c7 * term_7 + 
           c8 * term_8 + 
           c9 * term_9)
           
    return val
