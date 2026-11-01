"""Pack 50 equal circles in the unit circle using force-directed + soft-min optimization."""
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.distance import cdist

N = 50

def _compute_score(centers):
    """Compute the packing score (largest common radius)."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    dists = np.sqrt(np.sum(centers**2, axis=1))
    min_boundary = 1.0 - np.max(dists)
    if min_boundary < 0:
        return 0.0
    pairwise = cdist(centers, centers)
    np.fill_diagonal(pairwise, np.inf)
    min_pair = np.min(pairwise)
    score = min(min_boundary, min_pair / 2.0)
    return max(score, 0.0)

def _soft_min(values, beta):
    """Smooth approximation of min using log-sum-exp."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return 0.0
    m = np.min(v)
    return m - np.log(np.sum(np.exp(beta * (v - m)))) / beta

def _objective_and_gradient(x, beta):
    """Objective: negative soft-min of constraints; returns (value, gradient)."""
    n = N
    centers = x.reshape(n, 2)
    # Boundary constraints: 1 - |c_i|
    dists = np.sqrt(np.sum(centers**2, axis=1))
    boundary = 1.0 - dists
    # Pairwise constraints: |c_i - c_j| / 2
    pairwise = cdist(centers, centers)
    np.fill_diagonal(pairwise, np.inf)
    pair_half = pairwise / 2.0
    
    all_constraints = np.concatenate([boundary, pair_half.ravel()])
    # Use soft-min
    m = np.min(all_constraints)
    exp_vals = np.exp(beta * (all_constraints - m))
    soft_min_val = m - np.log(np.sum(exp_vals)) / beta
    obj = -soft_min_val
    
    # Gradient via automatic differentiation approximation (numerical)
    # Actually let's compute analytically
    # d(soft_min)/d(constraint_i) = -exp(beta*(c_i - m)) / sum(exp(beta*(c_j - m)))
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(x)
    # Boundary gradient: d(1-|c_i|)/dc_i = -c_i / |c_i|
    for i in range(n):
        if dists[i] > 1e-12:
            grad[2*i] = -weights[i] * (-centers[i, 0] / dists[i])
            grad[2*i+1] = -weights[i] * (-centers[i, 1] / dists[i])
        else:
            grad[2*i] = -weights[i] * 0.0
            grad[2*i+1] = -weights[i] * 0.0
    
    # Pairwise gradient: d(|c_i-c_j|/2)/dc_i = (c_i - c_j)/|c_i-c_j| * 1/2
    for i in range(n):
        for j in range(i+1, n):
            d = pairwise[i, j]
            if d > 1e-12:
                w = weights[n + i*n + j]  # index into pair_half.ravel()
                # Actually pair_half.ravel() ordering: cdist gives upper triangle? No, full matrix
                # pair_half.ravel() is row-major: index = i*n + j
                w = weights[n + (i * n + j)]
                diff = centers[i] - centers[j]
                grad[2*i] += -w * (diff[0] / d) * 0.5
                grad[2*i+1] += -w * (diff[1] / d) * 0.5
                grad[2*j] += -w * (-diff[0] / d) * 0.5
                grad[2*j+1] += -w * (-diff[1] / d) * 0.5
    
    return obj, grad

def _optimize_from_start(start_centers, beta_start=50.0, beta_end=500.0, time_limit=20.0):
    """Optimize using L-BFGS-B with increasing beta."""
    start = time.time()
    x0 = start_centers.flatten().astype(np.float64)
    
    best_x = x0.copy()
    best_score = _compute_score(start_centers)
    
    n_betas = 8
    betas = np.geomspace(beta_start, beta_end, n_betas)
    
    for beta in betas:
        if time.time() - start > time_limit - 2.0:
            break
        
        def obj_func(x):
            val, _ = _objective_and_gradient(x, beta)
            return val
        
        def grad_func(x):
            _, g = _objective_and_gradient(x, beta)
            return g
        
        try:
            result = minimize(obj_func, best_x, jac=grad_func, method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if result.fun < obj_func(best_x):
                candidate = result.x.reshape(N, 2)
                cand_score = _compute_score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_x = candidate.flatten()
        except Exception:
            break
    
    return best_x.reshape(N, 2), best_score

def _force_directed_relax(centers, time_limit=5.0, target_radius=None):
    """Force-directed relaxation: repel overlapping circles, attract to center."""
    start = time.time()
    centers = centers.copy()
    
    if target_radius is None:
        target_radius = _compute_score(centers)
    
    r = target_radius * 1.01  # Slightly larger to push apart
    dt = 0.005
    damping = 0.9
    
    velocities = np.zeros_like(centers)
    
    while time.time() - start < time_limit - 1.0:
        forces = np.zeros_like(centers)
        
        # Repulsive forces between circles
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                dist = np.sqrt(np.sum(diff**2))
                min_dist = 2 * r
                if dist < min_dist and dist > 1e-12:
                    overlap = min_dist - dist
                    force_mag = 10.0 * overlap
                    force_dir = diff / dist
                    forces[i] += force_mag * force_dir
                    forces[j] -= force_mag * force_dir
        
        # Boundary force: push inward if too close to edge
        for i in range(N):
            dist_from_center = np.sqrt(np.sum(centers[i]**2))
            max_dist = 1.0 - r
            if dist_from_center > max_dist and dist_from_center > 1e-12:
                overlap = dist_from_center - max_dist
                force_mag = 20.0 * overlap
                force_dir = -centers[i] / dist_from_center
                forces[i] += force_mag * force_dir
        
        # Gentle attraction to center
        for i in range(N):
            forces[i] -= 0.1 * centers[i]
        
        # Update velocities and positions
        velocities = damping * velocities + forces * dt
        centers += velocities * dt
        
        # Clamp to unit disk
        for i in range(N):
            d = np.sqrt(np.sum(centers[i]**2))
            if d > 1.0 - r:
                centers[i] = centers[i] * (1.0 - r) / d
    
    return centers

def _get_hex_lattice(n, radius, angle=0.0):
    """Generate hexagonal lattice points within unit circle."""
    spacing = 2 * radius
    sqrt3 = np.sqrt(3)
    points = []
    max_k = int(np.ceil(1.0 / spacing)) + 3
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * sqrt3 / 2.0
            if np.sqrt(x**2 + y**2) <= 1.0 - radius + 1e-9:
                points.append([x, y])
    if abs(angle) > 1e-9:
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        points = [np.dot(rot, p) for p in points]
    points.sort(key=lambda p: p[0]**2 + p[1]**2)
    return np.array(points[:n])

def _get_ring_initialization(n, radii=None):
    """Initialize with concentric rings."""
    if radii is None:
        # Approximate optimal ring radii for 50 circles
        # Ring 0: 1 circle at center
        # Ring 1: 6 circles
        # Ring 2: 12 circles
        # Ring 3: 18 circles
        # Ring 4: 13 circles
        # Total: 50
        rings = [1, 6, 12, 18, 13]
        ring_radii = [0.0, 0.25, 0.50, 0.75, 0.90]
    else:
        rings = [1, 6, 12, 18, 13]
        ring_radii = radii
    
    points = []
    for ring_idx, (count, r) in enumerate(zip(rings, ring_radii)):
        if count == 1:
            points.append([0.0, 0.0])
        else:
            angles = np.linspace(0, 2*np.pi, count, endpoint=False)
            for a in angles:
                points.append([r * np.cos(a), r * np.sin(a)])
    
    return np.array(points[:n])

def _get_fibonacci_initialization(n):
    """Initialize using Fibonacci spiral."""
    points = []
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        r = 0.95 * np.sqrt((i + 0.5) / n)
        theta = i * golden_angle
        points.append([r * np.cos(theta), r * np.sin(theta)])
    return np.array(points)

def _get_random_initialization(n, seed):
    """Random initialization within unit circle."""
    rng = np.random.RandomState(seed)
    points = []
    for i in range(n):
        r = np.sqrt(rng.random()) * 0.95
        theta = rng.random() * 2 * np.pi
        points.append([r * np.cos(theta), r * np.sin(theta)])
    return np.array(points)

def _nelder_mead_polish(centers, time_limit=10.0):
    """Polish with Nelder-Mead on the exact score."""
    start = time.time()
    x0 = centers.flatten()
    
    def neg_score(x):
        c = x.reshape(N, 2)
        return -_compute_score(c)
    
    best_x = x0.copy()
    best_val = neg_score(x0)
    
    for _ in range(3):
        if time.time() - start > time_limit - 1.0:
            break
        try:
            result = minimize(neg_score, best_x, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12,
                                   'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
        except Exception:
            break
    
    return best_x.reshape(N, 2), -best_val

def _local_search(centers, time_limit=10.0):
    """Local search: try moving each point slightly in random directions."""
    start = time.time()
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    rng = np.random.RandomState(123)
    
    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    for step in step_sizes:
        if time.time() - start > time_limit - 1.0:
            break
        for _ in range(200):
            if time.time() - start > time_limit - 1.0:
                break
            idx = rng.randint(N)
            direction = rng.randn(2)
            direction /= np.linalg.norm(direction)
            
            candidate = best_centers.copy()
            candidate[idx] += step * direction
            
            # Keep within bounds
            d = np.linalg.norm(candidate[idx])
            if d > 1.0:
                candidate[idx] = candidate[idx] * (1.0 / d) * 0.999
            
            cand_score = _compute_score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_centers = candidate
    
    return best_centers, best_score

def construct():
    """Construct the best packing found within the time budget."""
    deadline = time.time() + 140.0  # Safety margin from 150s limit
    
    best_centers = None
    best_score = 0.0
    
    # Generate multiple initializations
    initializations = []
    
    # Hexagonal lattice with various angles
    for angle in [0.0, np.pi/6, np.pi/3, np.pi/4, np.pi/2, 2*np.pi/3]:
        for radius in [0.12, 0.125, 0.13]:
            try:
                init = _get_hex_lattice(N, radius, angle)
                if len(init) == N:
                    initializations.append(init)
            except Exception:
                pass
    
    # Ring initialization
    initializations.append(_get_ring_initialization(N))
    
    # Fibonacci spiral
    initializations.append(_get_fibonacci_initialization(N))
    
    # Random initializations
    for seed in [42, 123, 456, 789]:
        initializations.append(_get_random_initialization(N, seed))
    
    # Process each initialization
    for init_idx, init in enumerate(initializations):
        if time.time() > deadline - 30.0:
            break
        
        # Step 1: Force-directed relaxation
        try:
            relaxed = _force_directed_relax(init, time_limit=3.0)
            score = _compute_score(relaxed)
            if score > best_score:
                best_score = score
                best_centers = relaxed
        except Exception:
            relaxed = init
            score = _compute_score(relaxed)
            if score > best_score:
                best_score = score
                best_centers = relaxed
        
        # Step 2: Soft-min optimization with L-BFGS-B
        if time.time() > deadline - 20.0:
            break
        try:
            optimized, opt_score = _optimize_from_start(relaxed, beta_start=50.0, 
                                                        beta_end=500.0, 
                                                        time_limit=8.0)
            if opt_score > best_score:
                best_score = opt_score
                best_centers = optimized
        except Exception:
            pass
        
        # Step 3: Nelder-Mead polish
        if time.time() > deadline - 10.0:
            break
        try:
            polished, pol_score = _nelder_mead_polish(best_centers, time_limit=5.0)
            if pol_score > best_score:
                best_score = pol_score
                best_centers = polished
        except Exception:
            pass
    
    # Final high-precision polish
    if best_centers is not None and time.time() < deadline - 5.0:
        try:
            remaining = min(15.0, deadline - time.time() - 2.0)
            polished, pol_score = _nelder_mead_polish(best_centers, time_limit=remaining)
            if pol_score > best_score:
                best_score = pol_score
                best_centers = polished
        except Exception:
            pass
    
    # Local search as final refinement
    if best_centers is not None and time.time() < deadline - 3.0:
        try:
            remaining = min(10.0, deadline - time.time() - 1.0)
            refined, ref_score = _local_search(best_centers, time_limit=remaining)
            if ref_score > best_score:
                best_score = ref_score
                best_centers = refined
        except Exception:
            pass
    
    # Final Nelder-Mead
    if best_centers is not None and time.time() < deadline - 2.0:
        try:
            remaining = max(2.0, deadline - time.time() - 1.0)
            polished, pol_score = _nelder_mead_polish(best_centers, time_limit=remaining)
            if pol_score > best_score:
                best_score = pol_score
                best_centers = polished
        except Exception:
            pass
    
    if best_centers is None:
        best_centers = _get_hex_lattice(N, 0.12)
    
    return best_centers
