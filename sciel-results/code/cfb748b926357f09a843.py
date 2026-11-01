"""Optimize 50 circle centres in the unit disk via soft-min + L-BFGS-B with high-precision polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 130.0  # Safety margin before 150s limit


def _compute_score(points):
    """Compute the packing radius (higher is better)."""
    if points.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists_to_origin = np.sqrt(np.sum(points ** 2, axis=1))
    boundary_margin = 1.0 - np.max(dists_to_origin)
    if boundary_margin <= 0:
        return 0.0
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    if min_pair <= 0:
        return 0.0
    return min(min_pair / 2.0, boundary_margin)


def _soft_min(x, beta):
    """Soft minimum via log-sum-exp for smooth optimization."""
    # Use stable log-sum-exp
    x_min = np.min(x)
    if x_min < -1e10:
        return x_min
    # Soft min = -soft max = -(1/beta) * log(sum(exp(-beta * x)))
    # For numerical stability:
    exponents = -beta * (x - x_min)
    max_exp = np.max(exponents)
    if max_exp > 500:
        # Overflow protection
        return x_min
    log_sum = max_exp + np.log(np.sum(np.exp(exponents - max_exp)))
    return -(1.0 / beta) * log_sum + x_min


def _soft_max(x, beta):
    """Soft maximum via log-sum-exp."""
    x_max = np.max(x)
    if x_max > 1e10:
        return x_max
    exponents = beta * (x - x_max)
    min_exp = np.min(exponents)
    if min_exp < -500:
        return x_max
    log_sum = min_exp + np.log(np.sum(np.exp(exponents - min_exp)))
    return (1.0 / beta) * log_sum + x_max


def _objective_and_gradient(x, beta=100.0):
    """Smooth objective: minimize negative soft-min packing radius with boundary penalty."""
    pts = x.reshape(N, 2)
    
    # Boundary: soft penalty for points outside disk
    r = np.sqrt(np.sum(pts ** 2, axis=1))
    # We want r[i] + margin <= 1, i.e., margin >= r[i] - 1
    # Penalty: soft_max of (r[i] - 1) which is 0 if all inside
    boundary_violation = np.maximum(0.0, r - 1.0)
    # Use a smooth approximation
    boundary_penalty = np.sum(boundary_violation ** 2) * 10.0
    
    # Pairwise distances
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    
    # Packing radius is min(min_pair/2, 1 - max_r)
    max_r = np.max(r)
    boundary_margin = 1.0 - max_r
    
    # Soft minimum of the two constraints
    # We want to maximize min(min_pair/2, boundary_margin)
    # So minimize -soft_min(min_pair/2, boundary_margin)
    
    # For the pairwise part, use soft min over all pairs
    # Reshape d to get all unique pairs
    iu = np.triu_indices(N, k=1)
    pair_dists = d[iu]
    
    # Soft min over pair distances
    soft_min_pair = _soft_min(pair_dists, beta)
    soft_min_pair_half = soft_min_pair / 2.0
    
    # Combine with boundary margin
    # soft_min of (soft_min_pair_half, boundary_margin)
    combined = np.array([soft_min_pair_half, boundary_margin])
    soft_min_combined = _soft_min(combined, beta)
    
    obj = -soft_min_combined + boundary_penalty
    
    # Gradient computation (numerical or analytical)
    # For L-BFGS-B, we can use numerical gradient if analytical is complex
    # Let's compute gradient analytically for the soft min
    
    # Gradient of soft_min(x, beta) w.r.t. x:
    # d/dx_i [-(1/beta) * log(sum_j exp(-beta*x_j))]
    # = (1/beta) * beta * exp(-beta*x_i) / sum_j exp(-beta*x_j)
    # = exp(-beta*x_i) / sum_j exp(-beta*x_j)
    
    # For pair distances, this is complex. Use numerical gradient fallback.
    # Actually, let's just return the objective and let scipy compute gradient
    # Or use a simpler approach: return just the objective
    
    return obj


def _objective_only(x, beta=100.0):
    """Objective function only, for methods that compute gradient numerically."""
    return _objective_and_gradient(x, beta)


def _get_hex_lattice():
    """Generate hexagonal lattice initialization."""
    best_pts = None
    best_score = -1.0
    
    for trial in range(30):
        spacing = 2.0 / (np.sqrt(N) + 1.0 + 0.01 * trial)
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x ** 2 + y ** 2 <= (1.0 - spacing / 2.0) ** 2:
                    rows.append((x, y))
        
        if len(rows) < N:
            continue
            
        pts = np.array(rows)
        radii = np.sqrt(np.sum(pts ** 2, axis=1))
        idx = np.argsort(radii, kind="stable")[:N]
        candidate = pts[idx]
        
        score = _compute_score(candidate)
        if score > best_score:
            best_score = score
            best_pts = candidate
    
    if best_pts is not None:
        return best_pts
    
    # Fallback: random
    return _get_random_initialization()


def _get_ring_initialization():
    """Initialize points in concentric rings."""
    pts = np.zeros((N, 2))
    
    # Place points in rings
    ring_counts = [1, 6, 12, 18, 13]  # Total = 50
    ring_radii = [0.0, 0.25, 0.5, 0.75, 0.95]
    
    idx = 0
    for ring, (count, radius) in enumerate(zip(ring_counts, ring_radii)):
        if count == 1:
            pts[idx] = [0.0, 0.0]
            idx += 1
        else:
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            for angle in angles:
                pts[idx] = [radius * np.cos(angle), radius * np.sin(angle)]
                idx += 1
    
    return pts


def _get_random_initialization(seed=None):
    """Generate random initial configuration."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((N, 2))
    for i in range(N):
        while True:
            x = rng.uniform(-0.9, 0.9)
            y = rng.uniform(-0.9, 0.9)
            if x ** 2 + y ** 2 <= 0.85 ** 2:
                pts[i] = [x, y]
                break
    return pts


def _get_golden_angle_initialization():
    """Initialize using golden angle spiral for even distribution."""
    pts = np.zeros((N, 2))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))  # ~2.399 radians
    
    for i in range(N):
        r = np.sqrt((i + 0.5) / N) * 0.95
        theta = i * golden_angle
        pts[i] = [r * np.cos(theta), r * np.sin(theta)]
    
    return pts


def _polish(pts, deadline, beta_start=50.0, beta_end=10000.0):
    """High-precision polish pass with increasing beta."""
    best_pts = pts.copy()
    best_score = _compute_score(pts)
    
    # Increase beta gradually for higher precision
    betas = [50.0, 100.0, 200.0, 500.0, 1000.0, 5000.0, 10000.0]
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                _objective_only,
                best_pts.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                options={
                    'maxiter': 2000,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxls': 50
                }
            )
            
            if time.time() > deadline:
                break
            
            x_opt = result.x.reshape(N, 2)
            
            # Clamp to disk
            r = np.sqrt(np.sum(x_opt ** 2, axis=1))
            for i in range(N):
                if r[i] > 0.999:
                    scale = 0.999 / r[i]
                    x_opt[i] *= scale
            
            score = _compute_score(x_opt)
            if score > best_score:
                best_score = score
                best_pts = x_opt.copy()
                
        except Exception:
            continue
    
    return best_pts


def _local_search(pts, deadline):
    """Local search: try small random perturbations and keep improvements."""
    best_pts = pts.copy()
    best_score = _compute_score(pts)
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Try a random perturbation
        scale = 0.01
        perturbed = best_pts + rng.normal(0, scale, size=(N, 2))
        
        # Clamp to disk
        r = np.sqrt(np.sum(perturbed ** 2, axis=1))
        for i in range(N):
            if r[i] > 0.995:
                scale_i = 0.995 / r[i]
                perturbed[i] *= scale_i
        
        score = _compute_score(perturbed)
        if score > best_score:
            best_score = score
            best_pts = perturbed.copy()
        else:
            # Reduce perturbation scale
            scale *= 0.95
            if scale < 1e-6:
                break
    
    return best_pts


def _swap_optimization(pts, deadline):
    """Try swapping pairs of points to escape local minima."""
    best_pts = pts.copy()
    best_score = _compute_score(pts)
    
    rng = np.random.default_rng(123)
    
    # Try several random swaps
    for _ in range(100):
        if time.time() > deadline:
            break
        
        # Pick two random indices
        i, j = rng.choice(N, size=2, replace=False)
        
        # Swap their positions
        swapped = best_pts.copy()
        swapped[i], swapped[j] = best_pts[j].copy(), best_pts[i].copy()
        
        # Also try a small rotation around the origin
        angle = rng.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        swapped[i] = rot @ best_pts[i]
        swapped[j] = rot @ best_pts[j]
        
        # Clamp to disk
        r = np.sqrt(np.sum(swapped ** 2, axis=1))
        for k in range(N):
            if r[k] > 0.995:
                scale_k = 0.995 / r[k]
                swapped[k] *= scale_k
        
        score = _compute_score(swapped)
        if score > best_score:
            best_score = score
            best_pts = swapped.copy()
    
    return best_pts


def construct():
    """Construct centres of N equal circles inside the unit circle."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Generate multiple initializations
    inits = []
    
    # Hexagonal lattice
    hex_pts = _get_hex_lattice()
    inits.append(hex_pts)
    
    # Ring initialization
    ring_pts = _get_ring_initialization()
    inits.append(ring_pts)
    
    # Golden angle spiral
    golden_pts = _get_golden_angle_initialization()
    inits.append(golden_pts)
    
    # Random initializations
    for seed in [1, 2, 3]:
        random_pts = _get_random_initialization(seed=seed)
        inits.append(random_pts)
    
    # Optimize from each initialization
    best_pts = None
    best_score = -1.0
    
    for init in inits:
        if time.time() > deadline - 10:
            break
        
        # Optimize with L-BFGS-B
        try:
            result = minimize(
                _objective_only,
                init.flatten(),
                args=(100.0,),
                method='L-BFGS-B',
                options={
                    'maxiter': 3000,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'maxls': 30
                }
            )
            
            if time.time() > deadline - 5:
                break
            
            x_opt = result.x.reshape(N, 2)
            
            # Clamp to disk
            r = np.sqrt(np.sum(x_opt ** 2, axis=1))
            for i in range(N):
                if r[i] > 0.995:
                    scale = 0.995 / r[i]
                    x_opt[i] *= scale
            
            score = _compute_score(x_opt)
            if score > best_score:
                best_score = score
                best_pts = x_opt.copy()
                
        except Exception:
            continue
    
    # If no optimization succeeded, use best initialization
    if best_pts is None:
        best_pts = hex_pts
        best_score = _compute_score(hex_pts)
    
    # Local search to refine
    if time.time() < deadline - 5:
        best_pts = _local_search(best_pts, deadline - 2)
        best_score = _compute_score(best_pts)
    
    # Swap optimization to escape local minima
    if time.time() < deadline - 5:
        best_pts = _swap_optimization(best_pts, deadline - 2)
        best_score = _compute_score(best_pts)
    
    # Final high-precision polish
    if time.time() < deadline - 2:
        best_pts = _polish(best_pts, deadline - 0.5)
        best_score = _compute_score(best_pts)
    
    # Final validation
    final_score = _compute_score(best_pts)
    if final_score <= 0:
        # Fallback to hex init
        best_pts = _get_hex_lattice()
    
    return best_pts
