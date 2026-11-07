import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-informed basis for a nonlinear oscillator (Duffing / Van der Pol style).
    
    The model is structured to align with the underlying ODE:
        dv/dt = -x - alpha * x^3 - beta * (1 - x^2) * v
    
    We express the target y (presumably dv/dt or an acceleration-like quantity) as:
        y = -x - alpha * x^3 - beta * v + beta * x^2 * v
    
    To give the optimizer (BFGS) more flexibility while maintaining the physical structure,
    we parameterize the coefficients and add small correction terms.
    
    The 10 parameters are used as follows:
    p0: offset (should be ~0)
    p1: linear restoring force coeff (should be ~1)
    p2: cubic stiffness coeff (alpha)
    p3: linear damping coeff (beta)
    p4: amplitude-dependent damping coeff (beta, same as p3 ideally, but free)
    p5-p9: higher-order correction terms for asymmetry or numerical noise.
    
    To improve conditioning, we scale inputs to [-1, 1].
    """
    
    # Input scaling
    # x range: [-0.5728, 0.5275]
    # v range: [-0.1754, 0.2821]
    
    x_center = -0.02265
    x_scale = 0.55015
    v_center = 0.05335
    v_scale = 0.22875
    
    X = (x - x_center) / x_scale
    V = (v - v_center) / v_scale
    
    # Clip for safety
    X = np.clip(X, -1.0, 1.0)
    V = np.clip(V, -1.0, 1.0)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute powers
    X2 = X * X
    X3 = X2 * X
    X4 = X2 * X2
    V2 = V * V
    
    # Core Physics Terms (Duffing + Van der Pol)
    # Standard Duffing: -X - alpha * X^3
    # Standard Van der Pol damping: -beta * (1 - X^2) * V = -beta * V + beta * X^2 * V
    
    # We let the optimizer find the best combination.
    # Note: The sign and magnitude of these terms will be determined by BFGS.
    
    # Term 0: Constant
    t0 = 1.0
    
    # Term 1: Linear X (Restoring force)
    t1 = X
    
    # Term 2: Cubic X (Nonlinear stiffness)
    t2 = X3
    
    # Term 3: Linear V (Damping)
    t3 = V
    
    # Term 4: X^2 * V (Amplitude dependent damping)
    t4 = X2 * V
    
    # Term 5: X * V (Cross damping)
    t5 = X * V
    
    # Term 6: X^4 (Higher order stiffness)
    t6 = X4
    
    # Term 7: V^2 (Nonlinear damping)
    t7 = V2
    
    # Term 8: X^2 (Asymmetry/offset in stiffness)
    t8 = X2
    
    # Term 9: X^5 (Saturation)
    t9 = X4 * X
    
    # Linear combination
    return (p0 * t0 + p1 * t1 + p2 * t2 + p3 * t3 + p4 * t4 +
            p5 * t5 + p6 * t6 + p7 * t7 + p8 * t8 + p9 * t9)
