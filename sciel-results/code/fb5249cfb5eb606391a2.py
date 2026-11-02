"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


def _pairwise_distances_sq(points):
    """Compute squared pairwise distances."""
    pts = np.asarray(points, dtype=float)
    sq_norms = np.sum(pts * pts, axis=1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (pts @ pts.T)
    np.maximum(dist_sq, 0.0, out=dist_sq)
    return dist_sq


def _score(points):
    """Return (max_dist / min_dist)^2 for the given points."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return np.inf
    d2 = _pairwise_distances_sq(pts)
    mask = ~np.eye(len(pts), dtype=bool)
    off_diag = d2[mask]
    if len(off_diag) == 0:
        return np.inf
    min_d2 = np.min(off_diag)
    max_d2 = np.max(off_diag)
    if min_d2 < 1e-15:
        return np.inf
    return max_d2 / min_d2


def _soft_score(points, beta=50.0):
    """Smooth surrogate using log-sum-exp for max and min."""
    pts = np.asarray(points, dtype=float)
    d2 = _pairwise_distances_sq(pts)
    mask = ~np.eye(len(pts), dtype=bool)
    off_diag = d2[mask]
    if len(off_diag) == 0:
        return np.inf
    # Soft max: (1/beta) * log(sum(exp(beta*x)))
    max_val = np.max(off_diag)
    soft_max = (1.0 / beta) * np.log(np.sum(np.exp(beta * off_diag - beta * max_val))) + max_val
    # Soft min: -(1/beta) * log(sum(exp(-beta*x)))
    min_val = np.min(off_diag)
    soft_min = -(1.0 / beta) * np.log(np.sum(np.exp(-beta * off_diag + beta * min_val))) + min_val
    if soft_min < 1e-15:
        return np.inf
    return soft_max / soft_min


def _objective(x, beta=50.0):
    """Objective function for optimizer. x is flattened (N*D,) array."""
    pts = x.reshape(N, D)
    return _soft_score(pts, beta)


def _objective_and_grad(x, beta=50.0):
    """Objective with analytical gradient. x is flattened (N*D,) array."""
    pts = x.reshape(N, D)
    d2 = _pairwise_distances_sq(pts)
    mask = ~np.eye(N, dtype=bool)
    off_diag = d2[mask]
    
    if len(off_diag) == 0:
        return np.inf, np.zeros_like(x)
    
    max_val = np.max(off_diag)
    min_val = np.min(off_diag)
    
    # Soft max and soft min
    exp_max = np.exp(beta * off_diag - beta * max_val)
    sum_exp_max = np.sum(exp_max)
    soft_max = (1.0 / beta) * np.log(sum_exp_max) + max_val
    
    exp_min = np.exp(-beta * off_diag + beta * min_val)
    sum_exp_min = np.sum(exp_min)
    soft_min = -(1.0 / beta) * np.log(sum_exp_min) + min_val
    
    if soft_min < 1e-15:
        return np.inf, np.zeros_like(x)
    
    score = soft_max / soft_min
    
    # Gradient of soft_max w.r.t. off_diag
    # d(soft_max)/d(off_diag[k]) = exp_max[k] / sum_exp_max
    grad_soft_max = exp_max / sum_exp_max
    
    # Gradient of soft_min w.r.t. off_diag
    # d(soft_min)/d(off_diag[k]) = -exp_min[k] / sum_exp_min
    grad_soft_min = -exp_min / sum_exp_min
    
    # d(score)/d(off_diag[k]) = grad_soft_max[k]/soft_min - soft_max * grad_soft_min[k] / soft_min^2
    grad_score = grad_soft_max / soft_min - score * grad_soft_min / soft_min
    
    # Now need d(off_diag)/d(pts)
    # off_diag is the off-diagonal elements of d2
    # d2 = sq_norms[:, None] + sq_norms[None, :] - 2 * pts @ pts.T
    # d(d2[i,j])/d(pts[k]) = 2*pts[i,k] + 2*pts[j,k] - 2*pts[i,k]*delta_{k,i}... 
    # Actually: d2[i,j] = sum_m (pts[i,m] - pts[j,m])^2
    # d(d2[i,j])/d(pts[k]) = 2*(pts[i,k] - pts[j,k]) if i==k or j==k... 
    # More carefully: d2[i,j] = ||pts[i] - pts[j]||^2
    # d(d2[i,j])/d(pts[k,m]) = 2*(pts[i,m] - pts[j,m]) if k==i, 2*(pts[j,m] - pts[i,m]) if k==j, 0 otherwise
    
    # Build gradient
    grad = np.zeros_like(pts)
    # We need to map off_diag back to (i,j) pairs
    # Get indices
    rows, cols = np.where(mask)
    for idx in range(len(rows)):
        i, j = rows[idx], cols[idx]
        g = grad_score[idx]
        # d(d2[i,j])/d(pts[i,m]) = 2*(pts[i,m] - pts[j,m])
        # d(d2[i,j])/d(pts[j,m]) = 2*(pts[j,m] - pts[i,m])
        grad[i] += g * 2.0 * (pts[i] - pts[j])
        grad[j] += g * 2.0 * (pts[j] - pts[i])
    
    return score, grad.reshape(-1)


def _fibonacci_sphere(n):
    """Generate n points on a sphere using Fibonacci lattice."""
    points = []
    for i in range(n):
        z = 1.0 - (2.0 * i + 1.0) / n
        r = np.sqrt(1.0 - z * z)
        phi = 2.0 * np.pi * i / n
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        points.append([x, y, z])
    return np.array(points)


def _optimize_from_start(start_pts, max_iters=5000, tol=1e-12, beta=50.0, use_grad=True):
    """Run L-BFGS-B optimization from a starting configuration."""
    x0 = start_pts.reshape(-1)
    if use_grad:
        result = minimize(_objective_and_grad, x0, method='L-BFGS-B',
                          jac=True,
                          options={'maxiter': max_iters, 'ftol': tol, 'gtol': 1e-8, 'maxfun': 100000},
                          args=(beta,))
    else:
        result = minimize(_objective, x0, method='L-BFGS-B',
                          options={'maxiter': max_iters, 'ftol': tol, 'gtol': 1e-8, 'maxfun': 100000},
                          args=(beta,))
    return result.x.reshape(N, D), result.fun


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Generate multiple starting configurations
    starts = []
    
    # 1. Fibonacci sphere scaled
    fib = _fibonacci_sphere(N)
    starts.append(fib * 1.0)
    
    # 2. Icosahedron + center + extra points
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    ico_ext = np.vstack([ico, [[0, 0, 0]], [[0, 0, 1.2]]])[:N]
    starts.append(ico_ext)
    
    # 3. Random spherical
    np.random.seed(42)
    random_pts = np.random.randn(N, 3)
    random_pts = random_pts / np.linalg.norm(random_pts, axis=1, keepdims=True) * 0.9
    starts.append(random_pts)
    
    # 4. Halton sequence
    def _halton_sequence(n, dims):
        points = []
        for i in range(1, n + 1):
            point = []
            for base in range(2, 2 + dims):
                f = 1.0
                r = 0.0
                i_val = i
                while i_val > 0:
                    i_val, rem = divmod(i_val, base)
                    f /= base
                    r += f * rem
                point.append(r)
            points.append(point)
        return np.array(points)
    
    halton = _halton_sequence(N, 3)
    halton_scaled = halton * 2.0 - 1.0
    norms = np.linalg.norm(halton_scaled, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    halton_scaled = halton_scaled / norms * 0.8
    starts.append(halton_scaled)
    
    # 5. Two interpenetrating tetrahedra (stella octangula) with perturbation
    # Regular tetrahedron vertices
    tetra = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ], dtype=float)
    tetra = tetra / np.linalg.norm(tetra, axis=1, keepdims=True)
    
    # Second tetrahedron rotated by 180 degrees around z-axis (or some angle)
    angle = np.pi / 3  # 60 degrees
    rot = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    tetra2 = tetra @ rot.T
    
    # Scale second tetrahedron slightly differently
    scale2 = 0.85
    combined = np.vstack([tetra, tetra2 * scale2])
    # Add 6 more points: place them at midpoints or other symmetric positions
    # For now, add 6 random points near the center
    extra = np.random.randn(6, 3) * 0.3
    stella = np.vstack([combined, extra])[:N]
    starts.append(stella)
    
    # Optimize from each start
    best_pts = None
    best_score = np.inf
    
    for idx, start in enumerate(starts):
        if time.time() > deadline - 30.0:
            break
        
        try:
            # First pass with moderate beta
            opt_pts, opt_soft_score = _optimize_from_start(start, max_iters=3000, tol=1e-14, beta=100.0, use_grad=True)
            actual_score = _score(opt_pts)
            
            if actual_score < best_score:
                best_score = actual_score
                best_pts = opt_pts.copy()
        except Exception:
            pass
    
    # Final polish with higher beta and gradient
    if best_pts is not None and time.time() < deadline - 20.0:
        try:
            polished_pts, _ = _optimize_from_start(best_pts, max_iters=5000, tol=1e-16, beta=200.0, use_grad=True)
            polished_score = _score(polished_pts)
            if polished_score < best_score:
                best_pts = polished_pts
                best_score = polished_score
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback to grid
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_pts = np.array(grid, dtype=float)
    
    return best_pts
# EVOLVE-BLOCK-END
