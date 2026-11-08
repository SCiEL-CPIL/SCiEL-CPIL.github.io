"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-Informed Nonlinear Oscillator Model.
    
    Strategy:
    Instead of a dense, fixed-grid RBF approximation which can lead to ill-conditioned 
    linear systems for the weight fitting (BFGS on MSE), we use a sparse, physically 
    motivated basis. The target is the acceleration (dv/dt) of a nonlinear oscillator.
    
    A generic driven nonlinear oscillator can be described by:
    dv/dt = -k(x) - c(v) + F(t)
    
    We approximate:
    1. k(x): Nonlinear spring force. Terms: x, x^3 (Duffing), x^5 (stronger nonlinearity).
       Note: x^3 is often sufficient for Duffing, but we include higher order if params allow.
       Actually, with only 10 params, we must be selective.
       
    Let's select the most impactful terms for a general nonlinear oscillator:
    - Constant: 1
    - Drive: sin(t), cos(t)  (assuming periodic drive, t is in [30, 50])
    - State Linear: x, v
    - State Quadratic: x^2, v^2
    - Couplings: x*v (damping/nonlinear coupling), x^2*v (Van der Pol type), v^3 (nonlinear damping)
    
    Basis Functions (10 terms):
    1. 1.0 (Bias)
    2. sin(t)
    3. cos(t)
    4. x
    5. v
    6. x^2
    7. v^2
    8. x*v
    9. x^3
    10. v^3
    
    This basis is:
    - Sparse: Only 10 terms.
    - Orthogonal-ish: Trigonometric and polynomial terms in different variables are often well-conditioned.
    - Physically Relevant: Captures linear damping, nonlinear stiffness, nonlinear damping, and periodic drive.
    
    Why this might outperform RBF:
    - BFGS on a linear-in-parameters model is equivalent to least squares. 
    - The condition number of the design matrix for these standard monomials/trig functions 
      is likely much lower than for 10 overlapping Gaussians in 3D space, leading to 
      more stable and accurate parameter recovery from the "all ones" start.
    - If the true function is smooth and locally polynomial-like, this global basis 
      generalizes better than a local RBF with fixed centers that might not align with 
      the data's effective support.
    """
    
    # Pre-compute basis terms
    # Ensure inputs are arrays
    t = np.asarray(t)
    x = np.asarray(x)
    v = np.asarray(v)
    
    # 1. Bias
    b1 = np.ones_like(t)
    
    # 2. Drive terms (Periodic in t)
    b2 = np.sin(t)
    b3 = np.cos(t)
    
    # 3. Linear state terms
    b4 = x
    b5 = v
    
    # 4. Quadratic state terms
    b6 = x * x
    b7 = v * v
    
    # 5. Coupling term (Linear in both, captures energy transfer)
    b8 = x * v
    
    # 6. Cubic terms (Strong nonlinearity)
    b9 = x * x * x
    b10 = v * v * v
    
    # Construct design matrix (N, 10)
    # Using column_stack for efficiency
    features = np.column_stack((
        b1, b2, b3, b4, b5, b6, b7, b8, b9, b10
    ))
    
    # Linear combination with params
    # params: (10,)
    # features: (N, 10)
    # Result: (N,)
    return features @ params

# EVOLVE-BLOCK-END
