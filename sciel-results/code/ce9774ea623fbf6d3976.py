"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 130
DEADLINE = 125.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import math


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _covering_radius_deg(pts):
    """Compute covering radius in degrees via convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: use pairwise distances
        d = 1.0 - pts @ pts.T
        np.fill_diagonal(d, 0.0)
        d = np.maximum(d, 0.0)
        return np.degrees(np.arccos(np.clip(1.0 - np.max(d), -1.0, 1.0)))
    
    max_angle = 0.0
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        for sign in [1.0, -1.0]:
            x = sign * n
            da = np.dot(x, a)
            db = np.dot(x, b)
            dc = np.dot(x, c)
            if abs(da - db) < 1e-8 and abs(da - dc) < 1e-8:
                angle = np.arccos(np.clip(da, -1.0, 1.0))
                if angle > max_angle:
                    max_angle = angle
                break
    
    return np.degrees(max_angle)


def _fast_covering_radius(pts):
    """Fast approximation of covering radius using nearest-neighbor distances."""
    pts = _normalize(pts)
    # Use a grid-based approximation
    # Project to stereographic or use simple angular distance
    d = 1.0 - pts @ pts.T
    np.fill_diagonal(d, 0.0)
    d = np.maximum(d, 0.0)
    min_d = np.min(d, axis=1)
    max_d = np.max(min_d)
    return np.degrees(np.arccos(np.clip(1.0 - max_d, -1.0, 1.0)))


def _spectral_energy(pts, max_l=8):
    """Compute spectral energy using spherical harmonics up to degree max_l."""
    pts = _normalize(pts)
    n = len(pts)
    
    # Compute all pairwise dot products
    dots = pts @ pts.T
    
    # Spectral energy: sum over l of |sum_i Y_l(x_i)|^2 / (2l+1)
    # For simplicity, use a few low-order terms
    energy = 0.0
    
    # l=0: constant term
    s0 = np.sum(np.ones(n))
    energy += s0**2 / 1.0
    
    # l=1: linear terms
    for k in range(3):
        s1 = np.sum(pts[:, k])
        energy += s1**2 / 3.0
    
    # l=2: quadratic terms (simplified)
    # Use the fact that sum of Y_2m^2 over m is related to |x|^2 terms
    # For points on sphere, this simplifies
    for i in range(n):
        for j in range(n):
            d = dots[i, j]
            # Legendre polynomial P_2(d) = (3d^2 - 1)/2
            p2 = (3.0 * d**2 - 1.0) / 2.0
            energy += p2**2 / 5.0
            if i < j:
                energy += p2**2 / 5.0  # symmetric
    
    # l=3
    for i in range(n):
        for j in range(n):
            d = dots[i, j]
            # P_3(d) = (5d^3 - 3d)/2
            p3 = (5.0 * d**3 - 3.0 * d) / 2.0
            energy += p3**2 / 7.0
            if i < j:
                energy += p3**2 / 7.0
    
    return energy


def _combined_loss(pts, target_cr=11.5):
    """Combined loss: spectral energy + covering radius penalty."""
    pts = _normalize(pts)
    
    # Spectral energy term (spreads points)
    spec = _spectral_energy(pts, max_l=4)
    
    # Covering radius penalty
    cr = _fast_covering_radius(pts)
    penalty = 0.0
    if cr > target_cr:
        penalty = 50.0 * (cr - target_cr) ** 2
    
    return spec + penalty


def _fibonacci_sphere(n, phase=0.0, rng=None):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + phase
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    if rng is not None:
        pts += rng.randn(n, 3) * 1e-3
        pts = _normalize(pts)
    
    return pts


def _gradient_step(pts, loss_fn, step_size=0.01, rng=None):
    """Perform a gradient descent step using finite differences on the sphere."""
    pts = _normalize(pts).copy()
    n = len(pts)
    
    if rng is None:
        rng = np.random.RandomState(42)
    
    # Compute gradient using central differences for a subset of points
    grad = np.zeros_like(pts)
    eps = 1e-4
    
    # Only compute gradient for a random subset to save time
    subset_size = min(n, 20)
    subset_idx = rng.choice(n, size=subset_size, replace=False)
    
    for idx in subset_idx:
        for dim in range(3):
            # Perturb in tangent direction
            e = np.zeros(3)
            e[dim] = eps
            # Project to tangent plane
            tangent = e - np.dot(e, pts[idx]) * pts[idx]
            t_norm = np.linalg.norm(tangent)
            if t_norm < 1e-15:
                continue
            tangent /= t_norm
            
            # Move point
            delta = eps / t_norm
            new_pt = pts[idx] + delta * tangent
            new_pt = _normalize(new_pt.reshape(1, -1))[0]
            
            trial = pts.copy()
            trial[idx] = new_pt
            trial_loss = loss_fn(trial)
            
            grad[idx, dim] = (trial_loss - loss_fn(pts)) / delta
    
    # Update points
    for idx in subset_idx:
        # Project gradient to tangent plane
        g_tangent = grad[idx] - np.dot(grad[idx], pts[idx]) * pts[idx]
        
        # Move in negative gradient direction
        step = step_size * g_tangent
        new_pt = pts[idx] - step
        new_pt = _normalize(new_pt.reshape(1, -1))[0]
        pts[idx] = new_pt
    
    return _normalize(pts)


def _local_search(pts, time_limit=20.0, loss_fn=None, rng=None):
    """Local search with adaptive step size."""
    if loss_fn is None:
        loss_fn = _combined_loss
    if rng is None:
        rng = np.random.RandomState(int(time.time() * 1000) % 2**31)
    
    start_time = time.time()
    pts = _normalize(pts).copy()
    best_loss = loss_fn(pts)
    step_size = 0.05
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try moving each point
        for i in range(N):
            if time.time() - start_time >= time_limit:
                break
            
            # Try several random perturbations
            for _ in range(3):
                # Random tangent direction
                dir_vec = rng.randn(3)
                dir_vec -= np.dot(dir_vec, pts[i]) * pts[i]
                d_norm = np.linalg.norm(dir_vec)
                if d_norm < 1e-15:
                    continue
                dir_vec /= d_norm
                
                theta = step_size * rng.uniform(0.5, 1.5)
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                new_pt = cos_t * pts[i] + sin_t * dir_vec
                new_pt = _normalize(new_pt.reshape(1, -1))[0]
                
                trial = pts.copy()
                trial[i] = new_pt
                trial_loss = loss_fn(trial)
                
                if trial_loss < best_loss - 1e-10:
                    pts = trial
                    best_loss = trial_loss
                    improved = True
                    break
        
        if not improved:
            step_size *= 0.9
            if step_size < 1e-4:
                break
    
    return pts, best_loss


def _simulated_annealing(pts, time_limit=30.0, loss_fn=None, rng=None):
    """Simulated annealing to escape local minima."""
    if loss_fn is None:
        loss_fn = _combined_loss
    if rng is None:
        rng = np.random.RandomState(int(time.time() * 1000) % 2**31)
    
    start_time = time.time()
    pts = _normalize(pts).copy()
    best_pts = pts.copy()
    best_loss = loss_fn(pts)
    
    T = 0.3
    T_min = 0.001
    cooling = 0.998
    
    while time.time() - start_time < time_limit and T > T_min:
        i = rng.randint(N)
        dir_vec = rng.randn(3)
        dir_vec -= np.dot(dir_vec, pts[i]) * pts[i]
        d_norm = np.linalg.norm(dir_vec)
        if d_norm < 1e-15:
            T *= cooling
            continue
        dir_vec /= d_norm
        
        theta = 0.08 * rng.uniform(0.5, 1.5)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        new_pt = cos_t * pts[i] + sin_t * dir_vec
        new_pt = _normalize(new_pt.reshape(1, -1))[0]
        
        trial = pts.copy()
        trial[i] = new_pt
        trial_loss = loss_fn(trial)
        
        current_loss = loss_fn(pts)
        delta = trial_loss - current_loss
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            pts = trial
            if trial_loss < best_loss:
                best_loss = trial_loss
                best_pts = pts.copy()
        
        T *= cooling
    
    return best_pts, best_loss


def construct():
    global DEADLINE
    t_start = time.time()
    
    rng = np.random.RandomState(42)
    
    # Generate diverse starting configurations
    starts = []
    
    # Fibonacci spirals with different phases
    for phase in [0.0, 0.5, 1.0, 1.5, 2.0]:
        pts = _fibonacci_sphere(N, phase=phase, rng=rng)
        starts.append(pts)
    
    # Random starts
    for _ in range(3):
        pts = rng.randn(N, 3)
        starts.append(_normalize(pts))
    
    best_pts = None
    best_score = np.inf
    
    for s_idx, start_pts in enumerate(starts):
        elapsed = time.time() - t_start
        if elapsed > DEADLINE - 25:
            break
        
        # Phase 1: Local search
        try:
            opt_pts, loss1 = _local_search(start_pts, time_limit=15.0, rng=rng)
        except Exception:
            continue
        
        elapsed = time.time() - t_start
        if elapsed > DEADLINE - 15:
            break
        
        # Phase 2: Simulated annealing
        try:
            sa_pts, loss2 = _simulated_annealing(opt_pts, time_limit=15.0, rng=rng)
        except Exception:
            sa_pts = opt_pts
        
        # Phase 3: Final local search polish
        try:
            final_pts, loss3 = _local_search(sa_pts, time_limit=8.0, rng=rng)
        except Exception:
            final_pts = sa_pts
        
        # Evaluate
        try:
            score = _covering_radius_deg(final_pts)
        except Exception:
            score = np.inf
        
        if score < best_score:
            best_score = score
            best_pts = final_pts.copy()
        
        elapsed = time.time() - t_start
        if elapsed > DEADLINE - 10:
            break
    
    # If we didn't find anything, use the first Fibonacci spiral
    if best_pts is None:
        best_pts = starts[0]
    
    # Final normalization and deduplication
    best_pts = _normalize(best_pts)
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-6:
                best_pts[j] += rng.randn(3) * 1e-4
                best_pts[j] = _normalize(best_pts[j].reshape(1, -1))[0]
    
    return best_pts
# EVOLVE-BLOCK-END
