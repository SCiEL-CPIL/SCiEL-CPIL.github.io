"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Universal approximator basis for nonlinear oscillator dynamics.
    
    Strategy:
    Instead of a sparse, hand-picked physics basis (which might miss specific 
    interaction terms or suffer from ill-conditioning if terms are correlated), 
    we use a dense, orthogonalized basis consisting of:
    1. Constant term (bias)
    2. Linear terms: t, x, v
    3. Fourier terms for t: sin(t), cos(t), sin(2t), cos(2t)
       - Captures periodic forcing and resonant modes.
       - t is in [30, 50], so frequency 1 and 2 are relevant.
    4. Radial/Polynomial terms for x and v: x^2, v^2, x*v
       - Captures Duffing-like (x^3 is approximated by linear combo of x, x^2, x^3? 
         No, we need odd/even separation or just a dense enough set).
       - Actually, let's use a Chebyshev-like or just simple monomials up to degree 3 
         but orthogonalized? 
       
    Let's refine the basis to be more robust for BFGS starting from 1s.
    A common issue with dense bases is that BFGS might struggle if the Hessian 
    is poorly scaled. However, the inputs are bounded:
    t in [30, 50] -> sin/cos are bounded [-1, 1]. t itself is large, so p0*t 
    might be large. But the target y is in [-0.71, 0.66]. 
    If we include 't' as a linear term, the coefficient must be very small 
    (approx 0) to not blow up the output. This creates a scaling problem.
    
    Better approach: Center and scale t? 
    We cannot modify inputs. But we can use sin/cos of t to handle the periodic 
    nature. If there is a secular drift, it's likely small or zero in a 
    steady-state oscillator.
    
    Let's construct a basis that covers:
    - Constant: 1
    - x, v (linear restoring/damping)
    - x^2, v^2, x*v (quadratic nonlinearities)
    - x^3, v^3 (cubic nonlinearities, Duffing/vdP)
    - sin(t), cos(t) (forcing)
    - sin(2t), cos(2t) (harmonic forcing)
    - sin(t)*x, cos(t)*x (parametric forcing)
    - sin(t)*v, cos(t)*v (parametric damping)
    
    That's 1+3+3+2+2+2+2 = 15 terms. We only have 10 params.
    
    We need to select the most important 10.
    1. 1 (bias)
    2. x
    3. v
    4. x^3 (Duffing)
    5. v^3 (vdP)
    6. x*v (cross term)
    7. sin(t)
    8. cos(t)
    9. sin(2t)
    10. cos(2t)
    
    This is similar to the previous one but drops 't' (secular) and adds 2nd 
    harmonic. The previous one had 't' which might have caused scaling issues 
    or just wasn't needed. The previous score was 57.9, which is an improvement 
    over 1.0 (seed). Wait, the seed score is 1.0? 
    "1.0 equals the seed program and above 1.0 is an improvement"
    Current score: 57.9167. This is a huge improvement over 1.0.
    
    Why is it so high? 
    Maybe the seed program was a very simple linear model or random.
    
    Let's look at the inspiration 3 (score 0.9795): "Replace the fixed... with a 
    universal approximator basis". It scored 0.98, which is WORSE than 1.0.
    This suggests that a generic dense basis might be harder for BFGS to converge 
    than a physically motivated sparse one.
    
    The current program (score 57.9) uses:
    p0 + p1*x + p2*x^3 + p3*v + p4*v^3 + p5*x*v + p6*sin(t) + p7*cos(t) + p8*t + p9
    
    It includes 't'. If the data has no secular trend, p8 should go to 0. 
    But 't' is in [30, 50]. If p8 is not exactly 0, the error is p8*(t - t_mean). 
    Variance of t is high. This could make the Hessian ill-conditioned if p8 
    is coupled with other terms.
    
    Alternative Strategy: 
    Use a **Gaussian Process-like** basis? No, too complex for BFGS.
    
    How about **RBF (Radial Basis Functions)** in (x, v) space?
    Centers at (0,0), (0.1, 0.1), (-0.1, -0.1), etc.
    This provides local flexibility.
    
    Let's try a **Hybrid**: 
    Keep the strong physics terms (x, v, x^3, v^3, x*v) but replace the 
    time-dependence with a more robust Fourier series and drop the linear 't'.
    
    Basis:
    1. 1
    2. x
    3. v
    4. x^3
    5. v^3
    6. x*v
    7. sin(t)
    8. cos(t)
    9. sin(2t)
    10. cos(2t)
    
    This removes 't' (secular) which is risky for scaling, and adds 2nd harmonic 
    which is common in nonlinear oscillators (Duffing generates 3rd harmonic, 
    but forcing might be 1st or 2nd).
    
    Let's also consider that the previous program had `p0` as a constant and `p9` 
    as another constant? 
    `result = (p0 + ... + p9)`. 
    If p0 and p9 are both constants, they are perfectly collinear. BFGS might 
    struggle or just split the bias arbitrarily. This is a bad sign for 
    optimization stability.
    
    Revised Basis (Orthogonal/Independent):
    1. 1 (Bias)
    2. x
    3. v
    4. x^2
    5. v^2
    6. x*v
    7. x^3
    8. v^3
    9. sin(t)
    10. cos(t)
    
    This drops x^3? No, Duffing is x^3. 
    It drops 2nd harmonic. 
    It includes x^2, v^2 which might be useful for asymmetric potentials or 
    rectification.
    
    Let's go with this "Full Quadratic + Cubic Diagonals + Linear Forcing" basis.
    It is dense enough to capture most oscillator behaviors but sparse enough 
    for BFGS to handle without extreme ill-conditioning (assuming inputs are 
    scaled reasonably, which they are: x,v ~ 0.3, x^2 ~ 0.1, x^3 ~ 0.03).
    
    Scaling check:
    x ~ 0.3, v ~ 0.3
    x^2 ~ 0.09
    v^2 ~ 0.09
    x^3 ~ 0.027
    v^3 ~ 0.027
    sin/cos ~ 1.0
    
    The magnitudes of basis terms vary by ~40x. This might still be an issue.
    However, BFGS is quasi-Newton and can handle some scaling issues better 
    than GD.
    
    To improve convergence, we can **scale the basis functions** inside the 
    equation so that all terms are O(1).
    e.g., use (x/0.3)^3 instead of x^3? 
    But the params are fitted. If I scale the basis by 1/0.3^3, the param 
    will be fitted to be 1000x larger. This is fine for BFGS as long as the 
    Hessian is well-conditioned. Scaling the basis to O(1) improves the 
    condition number of the Gram matrix (X^T X).
    
    Let's implement a scaled basis:
    x_s = x / 0.3
    v_s = v / 0.3
    
    Basis:
    1. 1
    2. x_s
    3. v_s
    4. x_s^2
    5. v_s^2
    6. x_s * v_s
    7. x_s^3
    8. v_s^3
    9. sin(t)
    10. cos(t)
    
    This ensures all basis functions are in [-1, 1] roughly (since x in [-0.34, 0.36], x_s in [-1.1, 1.2]).
    This should significantly help BFGS convergence compared to the raw 
    polynomials or the unscaled mix.
    """
    start_time = time.time()
    
    # Unpack parameters
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Scale inputs to O(1) for better conditioning
    # Ranges: x ~ [-0.34, 0.36], v ~ [-0.21, 0.17] (from prompt: x [-0.3391, 0.3624], v [-0.2056, 0.1749])
    # Let's use a scale of 0.35 for both to be safe and simple.
    scale = 0.35
    xs = x / scale
    vs = v / scale
    
    # Compute basis terms
    b0 = 1.0
    b1 = xs
    b2 = vs
    b3 = xs * xs
    b4 = vs * vs
    b5 = xs * vs
    b6 = xs * xs * xs
    b7 = vs * vs * vs
    b8 = np.sin(t)
    b9 = np.cos(t)
    
    # Combine
    result = (p0 * b0 + 
              p1 * b1 + 
              p2 * b2 + 
              p3 * b3 + 
              p4 * b4 + 
              p5 * b5 + 
              p6 * b6 + 
              p7 * b7 + 
              p8 * b8 + 
              p9 * b9)
    
    # Safety check
    if not np.all(np.isfinite(result)):
        return np.zeros_like(t)
        
    return result
# EVOLVE-BLOCK-END
