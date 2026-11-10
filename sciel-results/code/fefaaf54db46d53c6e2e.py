import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150


def _energy(pts):
    """Compute log energy using math.fsum for maximum precision (matches evaluator)."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return 1e18
            terms.append(-math.log(d))
    return math.fsum(terms)


def _energy_fast(pts):
    """Fast energy for internal comparison."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu[0], iu[1]]))


def _energy_and_grad_tangent(X):
    """
    Compute log energy and tangent-space gradient for points on unit sphere.
    X: (n, 3) array of points (will be normalized internally).
    Returns (energy, grad_tangent) where grad_tangent is the gradient projected
    onto the tangent space at each point.
    """
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n = len(X)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    
    # Energy: -sum_{i<j} log(r_ij) = -0.5 * sum_{i!=j} log(r_ij)
    # Use upper triangle for efficiency
    iu = np.triu_indices(n, k=1)
    r = np.sqrt(r2[iu[0], iu[1]])
    E = -np.sum(np.log(r))
    
    # Gradient: dE/dX_i = sum_{j != i} (X_i - X_j) / r_ij^2
    # This is the Euclidean gradient; we project to tangent space
    inv_r2 = np.zeros((n, n))
    inv_r2[iu[0], iu[1]] = 1.0 / r2[iu[0], iu[1]]
    inv_r2[iu[1], iu[0]] = 1.0 / r2[iu[0], iu[1]]
    
    # dE_dX[i] = sum_j (X[i] - X[j]) * inv_r2[i,j]
    dE_dX = np.sum(diff * inv_r2[:, :, np.newaxis], axis=1)
    
    # Project to tangent space: g_tan = g - (g . x) x
    g_tan = dE_dX - np.sum(dE_dX * X, axis=1, keepdims=True) * X
    
    return E, g_tan


def _project_to_sphere(points):
    """Project points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms


def _optimize_lbfgs(X0, maxiter=1000, ftol=1e-14, gtol=1e-12, deadline=None):
    """
    Optimize using L-BFGS-B in Cartesian coordinates with tangent-space gradient.
    The projection to sphere happens inside the objective/gradient function.
    """
    def objective(x_flat):
        pts = x_flat.reshape(-1, 3)
        e, g = _energy_and_grad_tangent(pts)
        return e, g.ravel()
    
    try:
        result = minimize(
            objective,
            X0.reshape(-1),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol,
                     'maxls': 50}
        )
        if result.x is not None and np.all(np.isfinite(result.x)):
            return _project_to_sphere(result.x.reshape(-1, 3))
    except Exception:
        pass
    return X0


def _riemannian_cg(X0, max_iter=3000, tol=1e-12, deadline=None):
    """
    Riemannian conjugate gradient on the sphere with geodesic line search.
    This works directly on the manifold and can be more robust than
    projected L-BFGS for this problem.
    """
    X = _project_to_sphere(X0.copy())
    n = len(X)
    
    E_prev, g = _energy_and_grad_tangent(X)
    p = -g.copy()
    best_E = E_prev
    best_X = X.copy()
    
    for k in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Geodesic line search: move along p on the sphere
        # X_new(t) = (X + t*p) / ||X + t*p||
        # Use backtracking line search
        alpha = 1.0
        for ls_iter in range(30):
            X_trial = X + alpha * p
            X_trial = _project_to_sphere(X_trial)
            E_trial, g_trial = _energy_and_grad_tangent(X_trial)
            
            if E_trial < E_prev - 1e-4 * alpha * np.sum(g * p):
                break
            alpha *= 0.5
        
        else:
            # Line search failed; try smaller step
            alpha = 1e-6
            X_trial = X + alpha * p
            X_trial = _project_to_sphere(X_trial)
            E_trial, g_trial = _energy_and_grad_tangent(X_trial)
            if E_trial >= E_prev:
                break
        
        # Update
        X = X_trial
        E_prev = E_trial
        g = g_trial
        
        if E_prev < best_E:
            best_E = E_prev
            best_X = X.copy()
        
        # Check convergence
        if np.linalg.norm(g) < tol:
            break
        
        # Update direction (Polak-Ribiere)
        g_new = g
        beta = np.sum(g_new * (g_new - p)) / (np.sum(p * p) + 1e-30)
        if beta < 0:
            beta = 0
        p = -g_new + beta * p
        
    return best_X


def _fibonacci_spiral():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(pts)


def construct():
    start = time.time()
    deadline = start + 135.0  # 15s safety margin before 150s limit
    
    best_points = None
    best_energy = 1e18
    
    # Generate diverse initializations
    np.random.seed(42)
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(_fibonacci_spiral())
    
    # 2-7. Random initializations
    for seed in range(6):
        rng = np.random.RandomState(100 + seed)
        X = rng.randn(N, 3)
        X = _project_to_sphere(X)
        initializations.append(X)
    
    # 8. Slightly perturbed Fibonacci
    fib = _fibonacci_spiral()
    perturbed = fib + 0.1 * np.random.randn(N, 3)
    initializations.append(_project_to_sphere(perturbed))
    
    # Optimize each initialization
    for idx, X0 in enumerate(initializations):
        if time.time() > deadline - 10.0:
            break
        
        # First pass: L-BFGS with moderate settings
        X_opt = _optimize_lbfgs(X0, maxiter=800, ftol=1e-13, gtol=1e-11,
                                deadline=deadline - 10.0)
        X_opt = _project_to_sphere(X_opt)
        E = _energy_fast(X_opt)
        
        if E < best_energy:
            best_energy = E
            best_points = X_opt.copy()
        
        # If time permits, refine best with Riemannian CG
        if time.time() < deadline - 20.0 and best_points is not None:
            X_cg = _riemannian_cg(best_points, max_iter=2000, tol=1e-13,
                                  deadline=deadline - 10.0)
            X_cg = _project_to_sphere(X_cg)
            E_cg = _energy_fast(X_cg)
            if E_cg < best_energy:
                best_energy = E_cg
                best_points = X_cg.copy()
    
    # Final refinement: multiple passes with tighter tolerances
    if best_points is not None and time.time() < deadline - 5.0:
        for _ in range(3):
            if time.time() > deadline - 3.0:
                break
            X_ref = _optimize_lbfgs(best_points, maxiter=500, ftol=1e-15,
                                    gtol=1e-13, deadline=deadline - 3.0)
            X_ref = _project_to_sphere(X_ref)
            E_ref = _energy_fast(X_ref)
            if E_ref < best_energy:
                best_energy = E_ref
                best_points = X_ref.copy()
            else:
                break
    
    # Ensure no points are too close
    best_points = _project_to_sphere(best_points)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                best_points[j] += 1e-8 * np.random.randn(3)
                best_points = _project_to_sphere(best_points)
    
    # Final validation
    if not np.all(np.isfinite(best_points)):
        best_points = _fibonacci_spiral()
    
    return best_points
