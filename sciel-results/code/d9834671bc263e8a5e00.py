import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _compute_entropies(P):
    """
    Computes D = H(x-y) and M = max(H(x), H(x+y), H(x+2y)) for a 10x10 grid P[y, x].
    """
    n = 10
    ys = np.arange(n)
    xs = np.arange(n)
    Y, X = np.meshgrid(ys, xs, indexing='ij')
    
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    P_flat = P.flatten()
    
    # D: distribution of x - y
    val_d = X_flat - Y_flat
    # Values in [-9, 9]
    counts_d = np.zeros(19)
    for v in range(-9, 10):
        mask = (val_d == v)
        counts_d[v + 9] = np.sum(P_flat[mask])
        
    # H0: distribution of x
    val_h0 = X_flat
    counts_h0 = np.zeros(n)
    for v in range(n):
        mask = (val_h0 == v)
        counts_h0[v] = np.sum(P_flat[mask])
        
    # H1: distribution of x + y
    val_h1 = X_flat + Y_flat
    # Values in [0, 18]
    counts_h1 = np.zeros(19)
    for v in range(19):
        mask = (val_h1 == v)
        counts_h1[v] = np.sum(P_flat[mask])
        
    # H2: distribution of x + 2y
    val_h2 = X_flat + 2*Y_flat
    # Values in [0, 27]
    counts_h2 = np.zeros(28)
    for v in range(28):
        mask = (val_h2 == v)
        counts_h2[v] = np.sum(P_flat[mask])

    def shannon_entropy(counts):
        p = counts[counts > 1e-15]
        if len(p) == 0:
            return 0.0
        p = p / np.sum(p)
        return -np.sum(p * np.log2(p))
    
    D = shannon_entropy(counts_d)
    H0 = shannon_entropy(counts_h0)
    H1 = shannon_entropy(counts_h1)
    H2 = shannon_entropy(counts_h2)
    
    M = max(H0, H1, H2)
    
    return D, M

def _get_P_from_params(params):
    """
    Construct P from quadratic parameters.
    params: [A11, A12, A22, b1, b2, c]
    W[y, x] = A11*x^2 + 2*A12*x*y + A22*y^2 + b1*x + b2*y + c
    """
    A11, A12, A22, b1, b2, c = params
    n = 10
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys, indexing='xy') # Note: P[y,x], so X is rows? No, meshgrid with indexing='xy' gives X[i,j] = xs[j], Y[i,j]=ys[i].
    # We want P[y, x]. So we want X to vary along columns (axis 1) and Y along rows (axis 0).
    # np.meshgrid(xs, ys, indexing='ij') gives Y (10x10) and X (10x10) where Y[i,j]=ys[i], X[i,j]=xs[j].
    # This matches P[y, x] = P[i, j].
    
    Y, X = np.meshgrid(ys, xs, indexing='ij')
    
    W = A11 * X**2 + 2 * A12 * X * Y + A22 * Y**2 + b1 * X + b2 * Y + c
    
    # Softmax
    W_shifted = W - np.max(W)
    P_flat = np.exp(W_shifted)
    P_flat /= np.sum(P_flat)
    P = P_flat.reshape(10, 10)
    return P

def _objective(params):
    P = _get_P_from_params(params)
    D, M = _compute_entropies(P)
    if M < 1e-9 or D < 1e-9:
        return 1e10
    return -D / M

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    timeout = 630.0 # Evaluator kills at 630s
    safety_margin = 10.0
    deadline = start_time + timeout - safety_margin
    
    best_P = np.ones((10, 10)) / 100.0
    best_score = 0.0
    
    # Define bounds for parameters
    # A11, A12, A22 control shape. Negative values spread out.
    # b1, b2 control center.
    # c is irrelevant for softmax (shifts all by same amount), but keep it 0 or small.
    bounds = [
        (-5.0, 5.0),   # A11
        (-5.0, 5.0),   # A12
        (-5.0, 5.0),   # A22
        (-5.0, 5.0),   # b1
        (-5.0, 5.0),   # b2
        (0.0, 0.0)     # c (fixed)
    ]
    
    # We will use differential evolution for global search in the low-dim space,
    # then refine with L-BFGS-B.
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: Differential Evolution (Global)
    # DE is slow but good for 6D.
    # Let's try a few random restarts of L-BFGS-B from structured points first, 
    # as it's faster.
    
    # Structured Initial Points:
    # 1. Uniform (all zeros)
    # 2. Spread in x-y: A11 > 0, A22 > 0, A12 < 0 (to make (x-y)^2 term positive? No.)
    #    (x-y)^2 = x^2 - 2xy + y^2. So A11=1, A12=-1, A22=1 gives (x-y)^2.
    #    We want to MAXIMIZE entropy of x-y. So we want P to be uniform over x-y values.
    #    This means W should be flat along x-y directions? No, if W is flat, P is uniform.
    #    If W = -k(x-y)^2, then P is concentrated on x=y. Entropy of x-y is low.
    #    If W = +k(x-y)^2, P is spread out to corners. Entropy of x-y is high.
    #    So we want A11, A22 positive, A12 negative to encourage spread in x-y.
    
    init_points = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Uniform
        [2.0, -2.0, 2.0, 0.0, 0.0, 0.0], # Spread x-y
        [2.0, -2.0, 2.0, 1.0, -1.0, 0.0], # Shifted
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],   # Spread x+y (bad for M?)
        [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],   # Mild spread
        [3.0, -3.0, 3.0, 0.0, 0.0, 0.0],  # Strong spread x-y
        [1.0, -0.5, 1.0, 0.0, 0.0, 0.0],  # Moderate
    ]
    
    # Add random points
    for _ in range(20):
        p = rng.uniform([-5, -5, -5, -5, -5, 0], [5, 5, 5, 5, 5, 0])
        init_points.append(p)
    
    for p_init in init_points:
        if time.time() > deadline:
            break
            
        try:
            # Use L-BFGS-B for local refinement
            res = minimize(_objective, p_init, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.fun < 1e9:
                P = _get_P_from_params(res.x)
                D, M = _compute_entropies(P)
                if M > 1e-9 and D > 1e-9:
                    score = D / M
                    if score > best_score:
                        best_score = score
                        best_P = P.copy()
                        
        except Exception:
            continue

    # If we have time, try Differential Evolution for a global search
    if time.time() < deadline - 60:
        try:
            # DE bounds
            de_bounds = [
                (-5.0, 5.0),
                (-5.0, 5.0),
                (-5.0, 5.0),
                (-5.0, 5.0),
                (-5.0, 5.0),
                (0.0, 0.0)
            ]
            
            # Run DE with a reasonable number of generations
            # maxiter is generations. popsize is multiplier of len(bounds).
            res_de = differential_evolution(_objective, de_bounds, seed=42, maxiter=50, popsize=10, tol=1e-6)
            
            if res_de.fun < 1e9:
                P = _get_P_from_params(res_de.x)
                D, M = _compute_entropies(P)
                if M > 1e-9 and D > 1e-9:
                    score = D / M
                    if score > best_score:
                        best_score = score
                        best_P = P.copy()
                        
        except Exception:
            pass

    # Final Check and Cleanup
    # Ensure P is valid: shape (10,10), no NaN/Inf, sum 1, entries >= 0
    best_P = np.asarray(best_P, dtype=float)
    if best_P.shape != (10, 10):
        best_P = np.ones((10, 10)) / 100.0
        
    # Clip negatives (should be none from softmax, but just in case)
    best_P = np.clip(best_P, 0, None)
    
    # Normalize
    s = np.sum(best_P)
    if s <= 1e-9:
        best_P = np.ones((10, 10)) / 100.0
    else:
        best_P = best_P / s
        
    # Replace any remaining NaN/Inf
    if not np.all(np.isfinite(best_P)):
        best_P = np.ones((10, 10)) / 100.0

    return best_P
