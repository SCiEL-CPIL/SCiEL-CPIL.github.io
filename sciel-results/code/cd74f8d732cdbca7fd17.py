"""Fifty equal circles in a circle – optimised packing via simulated annealing + local refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # safety margin before 150s kill


def _compute_score(pts):
    """Return the packing radius for a set of points, or -inf if invalid."""
    if pts.shape != (N, 2):
        return -np.inf
    if not np.all(np.isfinite(pts)):
        return -np.inf
    # Distance to boundary
    dists_boundary = 1.0 - np.linalg.norm(pts, axis=1)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists, axis=1)
    min_pair = np.min(min_pair)
    r = min(min_pair / 2.0, np.min(dists_boundary))
    return r


def _score_and_grad(pts, target_r):
    """
    Objective: maximize the minimum of (pairwise distances / 2, boundary distances).
    We use a smooth approximation: minimize the negative of a soft-min of the constraints.
    Actually, for gradient ascent we want to maximize r.
    Let's define the objective as: for each constraint c_i (distance/2 or boundary dist),
    we want c_i >= target_r. The violation is max(0, target_r - c_i).
    We minimize the sum of squared violations, or use a soft-min.
    
    For L-BFGS-B, we need a smooth objective.
    Let's use: f(x) = -log(sum(exp(-k * c_i))) / k  (soft-min of c_i)
    Maximizing this is equivalent to maximizing the min c_i.
    But L-BFGS-B minimizes. So we minimize -soft_min.
    
    Actually, simpler: just use a potential function that penalizes violations.
    V = sum_{i<j} max(0, target_r - d_ij/2)^2 + sum_i max(0, target_r - (1-|c_i|))^2
    This is not smooth at the boundary. Use a smooth max: soft_max(x) = log(1+exp(k*x))/k
    Or just use: (target_r - c)^2 if target_r > c else 0, but that's not differentiable.
    
    Better approach: Use the fact that we want to maximize the minimum distance.
    Let's use a different strategy: binary search on r, and for each r, try to find a feasible packing.
    For a fixed r, the problem is: find points such that d_ij >= 2r and |c_i| <= 1-r.
    This is a feasibility problem. We can use gradient descent to minimize the violation.
    
    Violation function (smooth):
    For each pair i<j: v_ij = max(0, 2r - d_ij)^2  (but max is not smooth)
    Use: v_ij = (max(0, 2r - d_ij))^2  -> not smooth at 2r=d_ij
    Use smooth: v_ij = log(1 + exp(k*(2r - d_ij)))^2 / k^2  or similar.
    
    Let's use a simpler approach: 
    Objective to minimize: sum_{i<j} (max(0, 2r - d_ij/2))^2 + sum_i (max(0, r - (1-|c_i|)))^2
    But this is non-smooth.
    
    Alternative: Use the "energy" approach where we repel overlapping circles.
    E = sum_{i<j} (1/d_ij)^p  for d_ij < 2r, and boundary penalty.
    Minimize E. If E is small enough, the packing is feasible.
    
    Let's use: 
    E = sum_{i<j} max(0, 2r - d_ij)^2 + sum_i max(0, r - (1 - |c_i|))^2
    And use a smooth approximation for max: 
    smooth_max(a, b) = (a + b + sqrt((a-b)^2 + eps))/2
    So max(0, x) ≈ smooth_max(0, x)
    
    Actually, let's just use gradient ascent on the min distance directly, with random restarts.
    """
    pass


def _simulate_annealing(initial_pts, target_r, steps=5000, T0=0.05, T1=0.001, dt=0.01):
    """
    Simulated annealing to find a feasible packing for target_r.
    Returns the best points found.
    """
    pts = initial_pts.copy()
    best_pts = pts.copy()
    best_score = _compute_score(pts)
    
    for step in range(steps):
        # Random perturbation
        i = np.random.randint(N)
        delta = np.random.randn(2) * dt
        new_pts = pts.copy()
        new_pts[i] += delta
        
        # Keep within bounds
        norm = np.linalg.norm(new_pts[i])
        if norm > 1 - target_r:
            new_pts[i] = new_pts[i] * (1 - target_r) / norm
        
        # Accept if better or with probability exp(-delta_E / T)
        # For simplicity, accept if score doesn't decrease too much
        new_score = _compute_score(new_pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
        
        T = T0 * (T1 / T0) ** (step / steps)
        if new_score >= _compute_score(pts) or np.random.rand() < np.exp((new_score - _compute_score(pts)) / T):
            pts = new_pts
    
    return best_pts


def construct():
    start_time = time.time()
    
    # Initial guess: hexagonal packing
    def hexagonal_init():
        pts = []
        spacing = 0.2
        for j in range(-5, 6):
            for i in range(-5, 6):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3) / 2
                if np.hypot(x, y) <= 1 - spacing/2:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            # Take the N closest to center
            dists = np.linalg.norm(pts, axis=1)
            idx = np.argsort(dists, kind='stable')[:N]
            return pts[idx]
        else:
            # Fallback: random
            return np.random.randn(N, 2) * 0.5
    
    # Try multiple random initializations
    best_pts = None
    best_r = 0.0
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        if restart == 0:
            pts = hexagonal_init()
        else:
            # Random initialization with some structure
            angles = np.sort(np.random.uniform(0, 2*np.pi, N))
            radii = np.random.uniform(0, 0.8, N)
            pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        # Local optimization using gradient ascent
        def objective(x):
            """Negative of the packing radius (to minimize)."""
            pts_2d = x.reshape(N, 2)
            # Boundary distances
            bdist = 1.0 - np.linalg.norm(pts_2d, axis=1)
            # Pairwise distances
            diff = pts_2d[:, np.newaxis, :] - pts_2d[np.newaxis, :, :]
            pdist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(pdist, np.inf)
            min_pd = np.min(pdist, axis=1)
            min_pd = np.min(min_pd) / 2.0
            r = min(min_pd, np.min(bdist))
            return -r
        
        def objective_grad(x):
            """Gradient of the objective (negative packing radius)."""
            pts_2d = x.reshape(N, 2)
            bdist = 1.0 - np.linalg.norm(pts_2d, axis=1)
            diff = pts_2d[:, np.newaxis, :] - pts_2d[np.newaxis, :, :]
            pdist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(pdist, np.inf)
            
            # Find the critical constraints
            min_pd = np.min(pdist, axis=1)
            min_pd = np.min(min_pd) / 2.0
            min_b = np.min(bdist)
            
            r = min(min_pd, min_b)
            
            grad = np.zeros((N, 2))
            
            if min_pd <= min_b:
                # Pairwise constraint is critical
                for i in range(N):
                    for j in range(i+1, N):
                        if abs(pdist[i, j]/2.0 - min_pd) < 1e-10:
                            d = pts_2d[i] - pts_2d[j]
                            dn = d / (pdist[i, j] + 1e-15)
                            grad[i] += dn / 2.0
                            grad[j] -= dn / 2.0
            else:
                # Boundary constraint is critical
                for i in range(N):
                    if abs(bdist[i] - min_b) < 1e-10:
                        norm_i = np.linalg.norm(pts_2d[i])
                        if norm_i > 1e-15:
                            grad[i] += pts_2d[i] / norm_i
            
            return grad.flatten()
        
        # Use L-BFGS-B with bounds
        x0 = pts.flatten()
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                             jac=objective_grad, options={'maxiter': 100, 'gtol': 1e-6})
            if result.success or result.fun < 0:
                new_pts = result.x.reshape(N, 2)
                new_r = _compute_score(new_pts)
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
        except Exception:
            pass
        
        # Try simulated annealing from the current best
        if time.time() - start_time < DEADLINE - 20:
            sa_pts = _simulate_annealing(best_pts if best_pts is not None else pts, 
                                         best_r if best_r > 0 else 0.1,
                                         steps=2000, T0=0.03, T1=0.005, dt=0.02)
            sa_r = _compute_score(sa_pts)
            if sa_r > best_r:
                best_r = sa_r
                best_pts = sa_pts.copy()
    
    # Final local refinement
    if best_pts is not None and time.time() - start_time < DEADLINE - 10:
        def obj2(x):
            pts_2d = x.reshape(N, 2)
            bdist = 1.0 - np.linalg.norm(pts_2d, axis=1)
            diff = pts_2d[:, np.newaxis, :] - pts_2d[np.newaxis, :, :]
            pdist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(pdist, np.inf)
            min_pd = np.min(pdist, axis=1)
            min_pd = np.min(min_pd) / 2.0
            r = min(min_pd, np.min(bdist))
            return -r
        
        x0 = best_pts.flatten()
        bounds = [(-1.0, 1.0)] * (2 * N)
        try:
            result = minimize(obj2, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 200, 'gtol': 1e-8})
            new_pts = result.x.reshape(N, 2)
            new_r = _compute_score(new_pts)
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback
        best_pts = hexagonal_init()
    
    return best_pts
