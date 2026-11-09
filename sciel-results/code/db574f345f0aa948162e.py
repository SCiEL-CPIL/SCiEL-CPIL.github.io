import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist

N = 100

def _energy(X):
    """Calculate logarithmic energy using math.fsum for accuracy."""
    XTX = X @ X.T
    D2 = 2.0 - 2.0 * XTX
    np.fill_diagonal(D2, 1.0)
    D = np.sqrt(np.clip(D2, 1e-30, None))
    iu = np.triu_indices(N, 1)
    vals = np.log(1.0 / D[iu])
    return math.fsum(vals)

def _energy_fast(X):
    """Fast energy for optimization (no fsum, vectorized)."""
    XTX = X @ X.T
    D2 = 2.0 - 2.0 * XTX
    np.fill_diagonal(D2, 1.0)
    D = np.sqrt(np.clip(D2, 1e-30, None))
    iu = np.triu_indices(N, 1)
    return np.sum(np.log(1.0 / D[iu]))

def _gradient(X):
    """Gradient of energy w.r.t. X."""
    XTX = X @ X.T
    D2 = 2.0 - 2.0 * XTX
    np.fill_diagonal(D2, 1.0)
    M = 1.0 / D2
    np.fill_diagonal(M, 0.0)
    S = np.sum(M, axis=1)
    W = M @ X
    G = S[:, np.newaxis] * X - W
    return G

def _project_tangent(G, X):
    G_dot_X = np.sum(G * X, axis=1, keepdims=True)
    return G - G_dot_X * X

def _normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return X / norms

def _fibonacci_start():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_start(seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    X = rng.randn(N, 3)
    return _normalize(X)

def _jittered_fibonacci(jitter=0.1, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    base = _fibonacci_start()
    X = base + jitter * rng.randn(N, 3)
    return _normalize(X)

def _lbfgs_optimize(X0, max_iter=5000, time_limit=100.0):
    """L-BFGS optimization on sphere with normalization."""
    start = time.time()
    X = _normalize(X0.copy())
    
    def objective(params):
        Xp = _normalize(params.reshape(N, 3))
        return _energy_fast(Xp)
    
    def grad(params):
        Xp = _normalize(params.reshape(N, 3))
        G = _gradient(Xp)
        G_tan = _project_tangent(G, Xp)
        return G_tan.reshape(-1)
    
    best_X = X.copy()
    best_E = _energy_fast(X)
    
    params = X.reshape(-1).copy()
    
    for restart in range(5):
        if time.time() - start > time_limit - 5.0:
            break
        
        result = minimize(
            objective,
            params,
            jac=grad,
            method='L-BFGS-B',
            options={'maxiter': max_iter // 5, 'gtol': 1e-12, 'factr': 1e12}
        )
        
        X_new = _normalize(result.x.reshape(N, 3))
        E_new = _energy_fast(X_new)
        
        if E_new < best_E:
            best_E = E_new
            best_X = X_new.copy()
        
        if time.time() - start > time_limit - 10.0:
            break
        
        # Add small perturbation for next restart
        if restart < 4:
            params = (X_new + 0.01 * np.random.randn(N, 3)).reshape(-1)
            params = _normalize(params.reshape(N, 3)).reshape(-1)
    
    return best_X, best_E

def _conjugate_gradient(X0, max_iter=10000, time_limit=100.0):
    """Conjugate gradient on tangent space with improved line search."""
    start = time.time()
    X = _normalize(X0.copy())
    E_old = _energy_fast(X)
    best_X = X.copy()
    best_E = E_old
    
    G = _gradient(X)
    G_tan = _project_tangent(G, X)
    p = -G_tan
    d = p.copy()
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 5.0:
            break
        
        # Improved line search: expand if possible, then contract
        alpha = 0.1
        # First try expanding
        for _ in range(5):
            X_trial = _normalize(X + alpha * d)
            E_trial = _energy_fast(X_trial)
            if E_trial < E_old:
                break
            alpha *= 2.0
        
        # Now contract to find good step
        alpha_prev = alpha
        for _ in range(30):
            X_new = _normalize(X + alpha * d)
            E_new = _energy_fast(X_new)
            if E_new < E_old:
                break
            alpha *= 0.5
        else:
            break
        
        X = X_new
        E_old = E_new
        
        if E_old < best_E:
            best_E = E_old
            best_X = X.copy()
        
        G_new = _gradient(X)
        G_tan_new = _project_tangent(G_new, X)
        
        if np.linalg.norm(G_tan_new) < 1e-10:
            break
        
        # Fletcher-Reeves update
        ratio = np.sum(G_tan_new * G_tan_new) / max(np.sum(G_tan * G_tan), 1e-15)
        d = -G_tan_new + ratio * d
        p = G_tan_new
    
    return best_X, best_E

def _differential_evolution_init(time_limit=60.0):
    """Use DE to find diverse good starting points (reduced iterations for speed)."""
    start = time.time()
    
    # DE operates on 300-dim space (N x 3), but we project to sphere
    def de_objective(params):
        X = _normalize(params.reshape(N, 3))
        return _energy_fast(X)
    
    bounds = [(-1.0, 1.0)] * (N * 3)
    
    try:
        result = differential_evolution(
            de_objective,
            bounds,
            maxiter=30,
            popsize=8,
            tol=1e-5,
            seed=42,
            workers=-1,
            polish=False,
            init='sobol'
        )
        X_de = _normalize(result.x.reshape(N, 3))
        return [X_de]
    except Exception:
        return []

def _final_polish(X0, time_limit=30.0):
    """High-precision final polish with very tight tolerance."""
    start = time.time()
    X = _normalize(X0.copy())
    
    def objective(params):
        Xp = _normalize(params.reshape(N, 3))
        return _energy_fast(Xp)
    
    def grad(params):
        Xp = _normalize(params.reshape(N, 3))
        G = _gradient(Xp)
        G_tan = _project_tangent(G, Xp)
        return G_tan.reshape(-1)
    
    params = X.reshape(-1).copy()
    
    result = minimize(
        objective,
        params,
        jac=grad,
        method='L-BFGS-B',
        options={'maxiter': 10000, 'gtol': 1e-14, 'factr': 1e14}
    )
    
    X_new = _normalize(result.x.reshape(N, 3))
    return X_new

def construct():
    start_time = time.time()
    deadline = 120.0
    time_margin = 10.0
    
    best_X = None
    best_E = float('inf')
    
    # Generate diverse starting points
    starts = [_fibonacci_start()]
    for s in range(5):
        starts.append(_random_start(seed=100 + s))
    for s in range(3):
        starts.append(_jittered_fibonacci(jitter=0.05 + 0.05 * s, seed=200 + s))
    
    # Try DE initialization if time permits
    if time.time() < deadline - time_margin - 25.0:
        de_starts = _differential_evolution_init(time_limit=deadline - time.time() - time_margin - 15.0)
        for ds in de_starts:
            if _energy_fast(ds) < 1100.0:  # Only keep good ones
                starts.append(ds)
    
    # Optimize from each start
    for i, X0 in enumerate(starts):
        remaining = deadline - time.time() - time_margin
        if remaining < 10.0:
            break
        
        # Use L-BFGS for main optimization
        X_opt, E_opt = _lbfgs_optimize(X0, max_iter=5000, time_limit=remaining - 5.0)
        
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt.copy()
        
        # If we have time, do a CG polish pass on the best result
        if i == 0 or E_opt < best_E:
            remaining2 = deadline - time.time() - time_margin
            if remaining2 > 15.0:
                X_polish, E_polish = _conjugate_gradient(X_opt, max_iter=5000, time_limit=remaining2 - 5.0)
                if E_polish < best_E:
                    best_E = E_polish
                    best_X = X_polish.copy()
    
    # Final high-precision polish if time permits
    if best_X is not None:
        remaining_final = deadline - time.time() - time_margin
        if remaining_final > 15.0:
            X_final = _final_polish(best_X, time_limit=remaining_final - 5.0)
            E_final = _energy_fast(X_final)
            if E_final < best_E:
                best_E = E_final
                best_X = X_final
    
    if best_X is None:
        best_X = _fibonacci_start()
    
    return _normalize(best_X)
