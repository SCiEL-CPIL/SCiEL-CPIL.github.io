"""Rewritten program: optimize 14-point configuration in 3D to minimize (max/min pairwise distance)^2."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 14
D = 3


def _pairwise_distances(pts):
    """Compute all pairwise distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists, 0)
    return dists


def _objective(x):
    """
    Objective: minimize (max pairwise dist / min pairwise dist)^2.
    x is a flat array of 42 values (14 points x 3 coords).
    We center the points to remove translation invariance.
    """
    pts = x.reshape(N, D)
    # Center
    pts = pts - pts.mean(axis=0)
    
    dists = _pairwise_distances(pts)
    # Get off-diagonal distances
    mask = ~np.eye(N, dtype=bool)
    off_diags = dists[mask]
    
    if len(off_diags) == 0:
        return 1e10
    
    d_min = np.min(off_diags)
    d_max = np.max(off_diags)
    
    if d_min < 1e-12:
        return 1e10
    
    ratio = d_max / d_min
    return ratio ** 2


def _make_initial_guess(seed=42):
    """Create a diverse initial guess using icosahedron-like structure."""
    np.random.seed(seed)
    # Start with points roughly on a sphere
    angles_theta = np.linspace(0, np.pi, N, endpoint=False)
    angles_phi = np.linspace(0, 2 * np.pi, N)
    r = np.ones(N)
    
    pts = np.zeros((N, D))
    for i in range(N):
        pts[i, 0] = r[i] * np.sin(angles_theta[i]) * np.cos(angles_phi[i])
        pts[i, 1] = r[i] * np.sin(angles_theta[i]) * np.sin(angles_phi[i])
        pts[i, 2] = r[i] * np.cos(angles_theta[i])
    
    # Add some randomness
    pts += np.random.randn(N, D) * 0.1
    return pts.flatten()


def construct():
    """
    Construct 14 points in 3D to minimize squared max/min pairwise distance ratio.
    Returns array of shape (14, 3).
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    best_score = np.inf
    best_x = None
    
    # Strategy 1: Try multiple random restarts with local optimization
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Different initial guesses
        if restart == 0:
            # Icosahedron-like structure
            phi = (1 + np.sqrt(5)) / 2
            ico_pts = np.array([
                [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
                [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
                [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
            ], dtype=float)
            # Normalize
            ico_pts /= np.linalg.norm(ico_pts, axis=1, keepdims=True)
            # Add 2 more points (poles with perturbation)
            extra = np.array([[0, 0, 1], [0, 0, -1]], dtype=float)
            extra += np.random.randn(2, 3) * 0.05
            init = np.vstack([ico_pts, extra]).flatten()
        else:
            init = _make_initial_guess(seed=restart * 100 + 42)
        
        # Local optimization using Powell
        try:
            result = minimize(
                _objective,
                init,
                method='Powell',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_x = result.x
        except Exception:
            pass
        
        if time.time() > deadline - 50:
            break
    
    # Strategy 2: If we still have time, try differential evolution for global search
    if time.time() < deadline - 60:
        try:
            bounds = [(-2, 2)] * (N * D)
            de_result = differential_evolution(
                _objective,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=False
            )
            
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_x = de_result.x
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if best_x is not None and time.time() < deadline - 10:
        try:
            polish_result = minimize(
                _objective,
                best_x,
                method='Powell',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            if polish_result.fun < best_score:
                best_score = polish_result.fun
                best_x = polish_result.x
        except Exception:
            pass
    
    # If no valid result found, fall back to a simple configuration
    if best_x is None:
        # Use cube + octahedron structure
        cube = np.array([
            [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1],
        ], dtype=float)
        octa = np.array([
            [-1, 0, 0], [1, 0, 0],
            [0, -1, 0], [0, 1, 0],
            [0, 0, -1], [0, 0, 1],
        ], dtype=float)
        best_x = np.vstack([cube, octa]).flatten()
    
    # Convert back to points and center
    pts = best_x.reshape(N, D)
    pts = pts - pts.mean(axis=0)
    
    # Check deadline
    if time.time() > deadline:
        # Emergency: return what we have
        pass
    
    return pts
