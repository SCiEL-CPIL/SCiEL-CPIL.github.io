"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    Uses dot-product trick for speed: |x_i - x_j|^2 = 2 - 2*x_i·x_j for unit vectors.
    Returns energy and the full (Euclidean) gradient.
    """
    pts = pts.reshape(-1, 3)
    n = pts.shape[0]
    
    # Compute pairwise dot products
    dot = pts @ pts.T  # (n, n)
    
    # Squared distances: |x_i - x_j|^2 = 2 - 2*dot_ij for unit vectors
    diff_sq = 2.0 - 2.0 * dot
    # Clamp to avoid negative values from floating point
    diff_sq = np.maximum(diff_sq, 1e-15)
    
    # Mask diagonal
    np.fill_diagonal(diff_sq, np.inf)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(diff_sq[iu])
    energy = -np.sum(np.log(d))
    
    # Gradient: grad_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    inv_dist_sq = np.zeros_like(diff_sq)
    np.divide(1.0, diff_sq, out=inv_dist_sq, where=diff_sq < np.inf)
    np.fill_diagonal(inv_dist_sq, 0)
    
    row_sums = np.sum(inv_dist_sq, axis=1)
    grad = inv_dist_sq @ pts - pts * row_sums[:, np.newaxis]
    
    return energy, grad


def _tangent_grad(pts, grad):
    """Project gradient onto tangent space of unit sphere at each point."""
    return grad - np.sum(grad * pts, axis=1, keepdims=True) * pts


def _project_and_normalize(pts):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_spiral(n=N):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n=N, seed=42):
    """Generate random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _perturbed_fibonacci(n=N, seed=42, scale=0.1):
    """Generate perturbed Fibonacci spiral points."""
    rng = np.random.RandomState(seed)
    pts = _fibonacci_spiral(n).copy()
    for i in range(n):
        noise = rng.randn(3) * scale
        pts[i] += noise
        pts[i] /= np.linalg.norm(pts[i])
    return pts


def _ring_initialization(n=N, n_rings=10):
    """Generate ring-based initialization: points distributed in latitude rings."""
    rng = np.random.RandomState(99)
    pts = np.zeros((n, 3))
    ring_sizes = np.array_split(np.arange(n), n_rings)
    for k, idx in enumerate(ring_sizes):
        # Latitude for ring k: spread from pole to pole
        lat = np.pi * (k + 0.5) / n_rings  # latitude angle from north pole
        theta = (np.arange(len(idx)) / len(idx)) * 2 * np.pi + rng.randn() * 0.1
        pts[idx, 0] = np.sin(lat) * np.cos(theta)
        pts[idx, 1] = np.sin(lat) * np.sin(theta)
        pts[idx, 2] = np.cos(lat)
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _local_optimize(pts, deadline, maxiter=5000, ftol=1e-14, gtol=1e-10):
    """Run L-BFGS-B local optimization using tangent-space parameterization.
    
    Uses the fact that for unit vectors, moving in tangent direction and
    re-normalizing is equivalent to a geodesic step. We optimize in R^3
    but constrain via projection each iteration, with tangent-projected gradients.
    """
    if time.time() > deadline:
        return pts, _energy_and_grad(pts)[0]
    
    x0 = pts.flatten()
    
    def objective(x):
        p = _project_and_normalize(x.reshape(-1, 3))
        e, _ = _energy_and_grad(p)
        return e
    
    def grad(x):
        p = _project_and_normalize(x.reshape(-1, 3))
        _, g = _energy_and_grad(p)
        g_tangent = _tangent_grad(p, g)
        return g_tangent.flatten()
    
    try:
        res = minimize(
            objective, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxfun': maxiter * 2}
        )
        cand = _project_and_normalize(res.x.reshape(-1, 3))
        cand_e = objective(cand.flatten())
        return cand, cand_e
    except Exception:
        return pts, _energy_and_grad(pts)[0]


def _simulated_annealing(pts, deadline, max_iter=2000, T0=0.1, T1=1e-6):
    """Simulated annealing with per-point adaptive step sizes and periodic re-optimization."""
    best_pts = pts.copy()
    best_e = _energy_and_grad(best_pts)[0]
    cur_e = best_e
    cur_pts = best_pts.copy()
    n = N
    rng = np.random.RandomState(12345)
    
    # Per-point adaptive step sizes (start at 0.05)
    step_sizes = np.full(n, 0.05)
    
    accepted = 0
    total = 0
    reopt_interval = 50  # Re-optimize every 50 SA steps
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Adaptive temperature
        if total > 50:
            acc_rate = accepted / total
            if acc_rate > 0.35:
                T0 *= 0.85
            elif acc_rate < 0.10:
                T0 *= 1.15
        
        T = max(T1, T0 * np.exp(-0.5 * it / max_iter))
        
        # Perturb a random subset of points
        n_perturb = max(2, n // 15)
        idx = rng.choice(n, size=n_perturb, replace=False)
        
        new_pts = cur_pts.copy()
        for i in idx:
            noise = rng.randn(3) * step_sizes[i]
            new_pts[i] = cur_pts[i] + noise
            new_pts[i] /= np.linalg.norm(new_pts[i])
        
        new_e = _energy_and_grad(new_pts)[0]
        delta_e = new_e - cur_e
        total += 1
        
        accept = delta_e < 0 or rng.rand() < np.exp(-delta_e / max(T, 1e-12))
        
        if accept:
            cur_pts = new_pts
            cur_e = new_e
            accepted += 1
            # Increase step sizes for accepted moves
            step_sizes[idx] *= 1.05
            if cur_e < best_e:
                best_pts = cur_pts.copy()
                best_e = cur_e
        else:
            # Decrease step sizes for rejected moves
            step_sizes[idx] *= 0.90
        
        # Clamp step sizes
        step_sizes = np.clip(step_sizes, 0.005, 0.3)
        
        # Periodic re-optimization to polish current position
        if (it + 1) % reopt_interval == 0 and time.time() < deadline - 10:
            reopt_pts, reopt_e = _local_optimize(cur_pts, deadline, maxiter=500, ftol=1e-12, gtol=1e-8)
            if reopt_e < best_e:
                best_pts = reopt_pts
                best_e = reopt_e
            # Update current state to the re-optimized position
            cur_pts = best_pts.copy()
            cur_e = best_e
    
    return best_pts, best_e


def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin before 150s limit
    
    best_pts = None
    best_energy = np.inf
    
    # Phase 1: Try multiple diverse starting points with local optimization
    starts = [
        _fibonacci_spiral(N),
        _random_points(N, seed=42),
        _random_points(N, seed=123),
        _random_points(N, seed=777),
        _perturbed_fibonacci(N, seed=7, scale=0.05),
        _perturbed_fibonacci(N, seed=13, scale=0.1),
        _ring_initialization(N, n_rings=10),
        _ring_initialization(N, n_rings=14),
    ]
    
    for i, start in enumerate(starts):
        if time.time() > deadline - 35:
            break
        cand, cand_e = _local_optimize(start, deadline, maxiter=2000, ftol=1e-13, gtol=1e-9)
        if cand_e < best_energy:
            best_pts = cand
            best_energy = cand_e
    
    # Phase 2: Simulated annealing to escape local minima
    if time.time() < deadline - 35:
        try:
            sa_pts, sa_e = _simulated_annealing(
                best_pts, deadline,
                max_iter=1200, T0=0.08, T1=1e-7
            )
            if sa_e < best_energy:
                best_pts = sa_pts
                best_energy = sa_e
        except Exception:
            pass
    
    # Phase 3: High-precision polish with tighter tolerances
    if time.time() < deadline - 12:
        try:
            cand, cand_e = _local_optimize(best_pts, deadline, maxiter=20000, ftol=1e-15, gtol=1e-11)
            if cand_e < best_energy:
                best_pts = cand
                best_energy = cand_e
        except Exception:
            pass
    
    # Phase 4: Final ultra-tight polish if time permits
    if time.time() < deadline - 4:
        try:
            cand, cand_e = _local_optimize(best_pts, deadline, maxiter=10000, ftol=1e-16, gtol=1e-13)
            if cand_e < best_energy:
                best_pts = cand
                best_energy = cand_e
        except Exception:
            pass
    
    # Final check and return
    result = _project_and_normalize(best_pts)
    
    # Validate: check for near-duplicate points
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(result[i] - result[j])
            if d < 1e-9:
                # Add small perturbation
                rng = np.random.RandomState(0)
                result[i] += rng.normal(0, 1e-8, 3)
                result[i] /= np.linalg.norm(result[i])
    
    # Ensure non-finite values are handled
    if not np.all(np.isfinite(result)):
        result = _fibonacci_spiral(N)
    
    return result
# EVOLVE-BLOCK-END
