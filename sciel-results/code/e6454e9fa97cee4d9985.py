"""Optimize 14 points in 3D to minimize (max_dist/min_dist)^2 via Gram matrix + smooth surrogates."""
import time
import numpy as np
from scipy.optimize import minimize

N = 14
D = 3
DEADLINE = 120.0  # seconds
SAFETY_MARGIN = 15.0  # stop searching this far before deadline

def _pairwise_dist_sq(G):
    """Compute all pairwise squared distances from Gram matrix G (n x n)."""
    n = G.shape[0]
    diag = np.diag(G)
    # d_ij^2 = G_ii + G_jj - 2 G_ij
    D2 = diag[:, None] + diag[None, :] - 2.0 * G
    np.fill_diagonal(D2, 0.0)
    # clip tiny negatives from numerical error
    np.maximum(D2, 0.0, out=D2)
    return D2

def _score_from_D2(D2):
    """Compute (max/min)^2 from squared distance matrix (off-diagonal)."""
    n = D2.shape[0]
    # off-diagonal entries
    mask = ~np.eye(n, dtype=bool)
    off = D2[mask]
    if off.size == 0:
        return np.inf
    dmin = np.min(off)
    dmax = np.max(off)
    if dmin <= 1e-12:
        return np.inf
    ratio = dmax / dmin
    return ratio * ratio

def _soft_max(x, lam=50.0):
    """Soft maximum via log-sum-exp."""
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(x - x_max))) / lam

def _soft_min(x, lam=50.0):
    """Soft minimum = -soft_max(-x)."""
    return -_soft_max(-x, lam)

def _smooth_objective(G, lam=50.0):
    """Smooth surrogate: (soft_max(d2) / soft_min(d2))^2 with PSD penalty."""
    G = 0.5 * (G + G.T)  # symmetrize
    D2 = _pairwise_dist_sq(G)
    n = D2.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off = D2[mask]
    
    smax = _soft_max(off, lam)
    smin = _soft_min(off, lam)
    
    if smin <= 1e-10:
        smin = 1e-10
    
    obj = (smax / smin) ** 2
    
    # PSD penalty: penalize negative eigenvalues
    eigvals = np.linalg.eigvalsh(G)
    neg_penalty = np.sum(np.maximum(0, -eigvals)) ** 2 * 1e4
    
    return obj + neg_penalty

def _smooth_gradient(G, lam=50.0):
    """Numerical gradient of smooth objective (for robustness)."""
    eps = 1e-7
    grad = np.zeros_like(G)
    flat = G.ravel()
    for i in range(flat.size):
        G_plus = G.copy()
        G_plus.flat[i] += eps
        G_minus = G.copy()
        G_minus.flat[i] -= eps
        grad.flat[i] = (_smooth_objective(G_plus, lam) - _smooth_objective(G_minus, lam)) / (2 * eps)
    return grad

def _exact_objective_and_grad(coords_flat, lam=0.0):
    """Exact objective with subgradient for final polish. coords_flat is (n*d,)."""
    X = coords_flat.reshape(N, D)
    # Pairwise distances
    diff = X[:, None, :] - X[None, :, :]
    D2 = np.sum(diff ** 2, axis=2)
    n = N
    mask = ~np.eye(n, dtype=bool)
    off = D2[mask]
    
    dmin = np.min(off)
    dmax = np.max(off)
    
    if dmin <= 1e-12:
        return 1e10, None
    
    ratio2 = (dmax / dmin) ** 2
    
    # Subgradient computation
    # f = (max/min)^2
    # df/dX = 2 * max/min^2 * d(max)/dX - 2 * max^2/min^3 * d(min)/dX
    # d(max)/dX_i = 2*(X_i - X_j) for the pair achieving max
    # d(min)/dX_i = 2*(X_i - X_j) for the pair achieving min
    
    grad = np.zeros_like(X)
    # Find indices of max and min
    off_idx = np.argwhere(mask)
    vals = D2[off_idx[:, 0], off_idx[:, 1]]
    max_idx = np.argmax(vals)
    min_idx = np.argmin(vals)
    
    i_max, j_max = off_idx[max_idx]
    i_min, j_min = off_idx[min_idx]
    
    dmax_dX = np.zeros_like(X)
    dmax_dX[i_max] += 2.0 * (X[i_max] - X[j_max])
    dmax_dX[j_max] += 2.0 * (X[j_max] - X[i_max])
    
    dmin_dX = np.zeros_like(X)
    dmin_dX[i_min] += 2.0 * (X[i_min] - X[j_min])
    dmin_dX[j_min] += 2.0 * (X[j_min] - X[i_min])
    
    grad = (2.0 * (dmax / dmin) ** 2) * dmax_dX - (2.0 * (dmax / dmin) ** 4) * dmin_dX
    
    return ratio2, grad.ravel()

def _gram_to_coords(G):
    """Convert Gram matrix to coordinates via Cholesky/eigh."""
    G = 0.5 * (G + G.T)
    # Ensure PSD
    eigvals, eigvecs = np.linalg.eigh(G)
    eigvals = np.maximum(eigvals, 0.0)
    G_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    G_psd = 0.5 * (G_psd + G_psd.T)
    try:
        L = np.linalg.cholesky(G_psd + 1e-10 * np.eye(N))
        X = L.T  # shape (N, N), take first D columns
        X = X[:D, :].T  # shape (N, D)
    except np.linalg.LinAlgError:
        # Fallback: use eigendecomposition
        X = eigvecs[:, :D] * np.sqrt(eigvals[:D])[None, :]
    return X

def _init_tetrahedron():
    """Initialize with tetrahedron-based structure."""
    # Start with a regular simplex-like configuration
    X = np.zeros((N, D))
    # Place points on a grid with jitter, then refine
    # Use a 2x2x2 cube plus extra points
    base = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]], dtype=float)
    # Scale to unit-ish
    base = base * 0.5
    # Add 6 more points on a sphere
    theta = np.linspace(0, 2*np.pi, 6, endpoint=False)
    extra = np.column_stack([np.cos(theta), np.sin(theta), np.ones(6)*0.5]) * 0.7
    X = np.vstack([base, extra])
    # Center
    X -= X.mean(axis=0)
    return X

def _optimize_from_coords(X0, max_time=30.0):
    """Optimize from initial coordinates using L-BFGS-B with exact objective."""
    x0 = X0.ravel()
    
    def obj_and_grad(x):
        val, grad = _exact_objective_and_grad(x)
        if grad is None:
            return val, np.zeros_like(x)
        return val, grad
    
    try:
        res = minimize(
            obj_and_grad, x0, method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000}
        )
        return res.x.reshape(N, D), res.fun
    except Exception:
        return X0, _score_from_D2(_pairwise_dist_sq(X0.T @ X0))

def _optimize_gram(G0, max_time=30.0, lam=50.0):
    """Optimize in Gram matrix space using smooth surrogate, then convert and polish."""
    g0 = G0.ravel()
    
    def obj(g):
        G = g.reshape(N, N)
        return _smooth_objective(G, lam)
    
    def grad(g):
        G = g.reshape(N, N)
        return _smooth_gradient(G, lam).ravel()
    
    try:
        res = minimize(
            obj, g0, jac=grad, method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-8, 'maxfun': 500}
        )
        G_opt = res.x.reshape(N, N)
        X = _gram_to_coords(G_opt)
    except Exception:
        X = _gram_to_coords(G0)
    
    # Polish with exact objective in coordinate space
    X, score = _optimize_from_coords(X, max_time=max_time)
    return X, score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_X = None
    best_score = np.inf
    
    # Strategy 1: Multiple random restarts with Gram matrix optimization
    rng = np.random.default_rng(42)
    
    # Precompute some initializations
    inits = []
    
    # Tetrahedron-based init
    X_tet = _init_tetrahedron()
    G_tet = X_tet.T @ X_tet
    inits.append(G_tet)
    
    # Random inits
    for i in range(8):
        X_rand = rng.standard_normal((N, D))
        X_rand -= X_rand.mean(axis=0)
        # Add some structure: place on a sphere
        norms = np.linalg.norm(X_rand, axis=1, keepdims=True)
        X_rand = X_rand / norms * rng.uniform(0.5, 1.0, size=(N, 1))
        G_rand = X_rand.T @ X_rand
        inits.append(G_rand)
    
    # Also try grid-based init
    side = 2
    grid = np.array([[x, y, z] for x in range(side) for y in range(side) for z in range(side)], dtype=float)
    if len(grid) >= N:
        grid = grid[:N]
        grid -= grid.mean(axis=0)
        G_grid = grid.T @ grid
        inits.append(G_grid)
    
    for G0 in inits:
        if time.time() > deadline:
            break
        
        X, score = _optimize_gram(G0, max_time=10.0)
        
        if score < best_score:
            best_score = score
            best_X = X
    
    # Final polish on best
    if best_X is not None and time.time() < deadline:
        best_X, best_score = _optimize_from_coords(best_X, max_time=20.0)
    
    if best_X is None:
        # Fallback: simple grid
        side = int(np.ceil(N ** (1 / D)))
        grid = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    grid.append([i, j, k])
                    if len(grid) == N:
                        break
                if len(grid) == N:
                    break
            if len(grid) == N:
                break
        best_X = np.array(grid, dtype=float)
    
    return best_X
