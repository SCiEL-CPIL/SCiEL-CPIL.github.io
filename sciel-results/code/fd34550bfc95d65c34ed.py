"""Hybrid Chebyshev-RBF basis for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    p = params.ravel()
    
    # Ensure p has length 10
    if p.shape[0] != 10:
        # Fallback if shape is weird, though evaluator guarantees length 10
        p = np.resize(p, 10)
        
    # Basis 1: Chebyshev Polynomials of the first kind (T_0 to T_3)
    # T_0(x) = 1
    # T_1(x) = x
    # T_2(x) = 2x^2 - 1
    # T_3(x) = 4x^3 - 3x
    # This captures global smooth trends efficiently.
    T0 = np.ones_like(x)
    T1 = x
    T2 = 2.0 * x**2 - 1.0
    T3 = 4.0 * x**3 - 3.0 * x
    
    # Basis 2: Gaussian RBFs (6 centers)
    # Centers spaced across [-2, 2]
    # Using 6 centers leaves 4 params for Chebyshev.
    # This split (4 global + 6 local) is a heuristic balance.
    centers = np.linspace(-1.8, 1.8, 6)
    # Sigma chosen to be wide enough to overlap but distinct enough to be local.
    # Domain width is 4. 6 centers -> spacing ~0.6.
    # Sigma ~0.4-0.5 is reasonable.
    sigma = 0.45
    
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    # Compute Gaussians: exp(-0.5 * (diff/sigma)^2)
    # Clip exponent to avoid underflow issues (though exp handles small numbers fine, 
    # we just want to ensure no NaNs/Infs)
    exponent = -0.5 * (diff / sigma) ** 2
    # np.exp is safe for negative large numbers (goes to 0)
    G = np.exp(exponent)
    
    # Construct design matrix Phi: shape (N, 10)
    # Columns 0-3: Chebyshev
    # Columns 4-9: Gaussians
    # T0, T1, T2, T3 are (N,)
    # G is (N, 6)
    Phi = np.hstack((T0[:, np.newaxis], T1[:, np.newaxis], T2[:, np.newaxis], T3[:, np.newaxis], G))
    
    # Prediction is linear in params: pred = Phi @ p
    pred = Phi @ p
    
    return pred
# EVOLVE-BLOCK-END
