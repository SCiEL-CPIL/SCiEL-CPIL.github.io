import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a Duffing-van der Pol inspired structure.
    """
    # Deadline guard
    start_time = time.time()
    # The function is called many times during optimization. 
    # The 150s limit is for the whole evaluation (fitting + testing).
    # This function itself should be fast.
    
    # Scaling inputs to O(1) range
    # x in [-0.5728, 0.5275] -> scale by 0.6
    # v in [-0.1754, 0.2821] -> scale by 0.3
    xs = x / 0.6
    vs = v / 0.3
    
    # Precompute powers
    xs2 = xs * xs
    vs2 = vs * vs
    xs3 = xs2 * xs
    vs3 = vs2 * vs
    xs4 = xs2 * xs2
    vs4 = vs2 * vs2
    
    # Duffing-van der Pol structure
    # Standard form: a = -x - beta*x^3 - mu*(v^2 - A^2)*v
    # Expanded: a = -x - beta*x^3 - mu*v^3 + mu*A^2*v
    # This uses: x, x^3, v, v^3
    # We add cross terms and higher orders to use all 10 params and capture more complexity.
    
    # p0: Linear x (Restoring force)
    term0 = xs
    # p1: Linear v (Damping/Excitation)
    term1 = vs
    # p2: Cubic x (Duffing nonlinearity)
    term2 = xs3
    # p3: Cubic v (vdP nonlinearity)
    term3 = vs3
    # p4: x * v^2 (Cross term)
    term4 = xs * vs2
    # p5: v * x^2 (Cross term)
    term5 = vs * xs2
    # p6: x^2 (Quadratic x, if asymmetric)
    term6 = xs2
    # p7: v^2 (Quadratic v, if asymmetric)
    term7 = vs2
    # p8: x^4 (Higher order stiffness)
    term8 = xs4
    # p9: v^4 (Higher order damping)
    term9 = vs4
    
    # Linear combination
    # Note: The order of params in the evaluator is fixed (0 to 9).
    return (params[0] * term0 +
            params[1] * term1 +
            params[2] * term2 +
            params[3] * term3 +
            params[4] * term4 +
            params[5] * term5 +
            params[6] * term6 +
            params[7] * term7 +
            params[8] * term8 +
            params[9] * term9)
