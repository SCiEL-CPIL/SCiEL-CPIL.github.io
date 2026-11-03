"""Packing 50 equal circles in a unit circle - multi-start optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check all centers are inside unit circle
    dists_to_origin = np.linalg.norm(centers, axis=1)
    if np.any(dists_to_origin >= 1.0):
        return 0.0
    # Compute min pairwise distance
    min_pairwise = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pairwise:
                min_pairwise = d
    return min(min_pairwise / 2.0, 1.0 - np.max(dists_to_origin))

def objective(x, t=0.1):
    """Smooth surrogate objective: minimize negative of soft-min radius."""
    centers = x.reshape(N, 2)
    # Distance to boundary
    dists_origin = np.linalg.norm(centers, axis=1)
    boundary_margins = 1.0 - dists_origin
    # Pairwise distances
    pairwise_dists = []
    for i in range(N):
        for j in range(i+1, N):
            pairwise_dists.append(np.linalg.norm(centers[i] - centers[j]))
    pairwise_dists = np.array(pairwise_dists)
    min_pairwise = np.min(pairwise_dists)
    # Soft min using log-sum-exp
    all_margins = np.concatenate([boundary_margins, pairwise_dists / 2.0])
    # Use log-sum-exp for soft min
    m = np.min(all_margins)
    soft_min = m + np.log(np.sum(np.exp((all_margins - m) / t))) * t
    return -soft_min

def gradient_obj(x, t=0.1):
    """Gradient of the soft-min objective."""
    centers = x.reshape(N, 2)
    dists_origin = np.linalg.norm(centers, axis=1)
    boundary_margins = 1.0 - dists_origin
    
    # Compute all margins
    margins = np.empty(N + N*(N-1)//2)
    margins[:N] = boundary_margins
    
    idx = N
    pairwise_indices = []
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            margins[idx] = d / 2.0
            pairwise_indices.append((i, j))
            idx += 1
    
    m = np.min(margins)
    exp_vals = np.exp((margins - m) / t)
    sum_exp = np.sum(exp_vals)
    
    # Soft min value
    soft_min = m + t * np.log(sum_exp)
    
    # Gradient with respect to centers
    grad = np.zeros_like(centers)
    
    # Boundary contributions
    weights_boundary = exp_vals[:N] / sum_exp
    for i in range(N):
        if np.linalg.norm(centers[i]) > 1e-10:
            grad[i] -= weights_boundary[i] * centers[i] / np.linalg.norm(centers[i])
    
    # Pairwise contributions
    weights_pairwise = exp_vals[N:] / sum_exp
    for k, (i, j) in enumerate(pairwise_indices):
        diff = centers[i] - centers[j]
        d = np.linalg.norm(diff)
        if d > 1e-10:
            grad[i] += weights_pairwise[k] * diff / (2.0 * d)
            grad[j] -= weights_pairwise[k] * diff / (2.0 * d)
    
    return -grad.flatten()

def lloyd_init(n_iter=50):
    """Lloyd's algorithm for better initial placement."""
    # Start with random points
    centers = np.random.randn(N, 2)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    centers *= np.sqrt(np.random.uniform(0, 1, (N, 1)))
    
    for _ in range(n_iter):
        # For each point, move to centroid of its Voronoi cell (approximated)
        new_centers = centers.copy()
        for i in range(N):
            # Find nearest neighbors
            dists = np.linalg.norm(centers - centers[i], axis=1)
            # Weight by inverse distance
            weights = 1.0 / (dists + 1e-10)
            weights[i] = 0
            centroid = np.sum(centers * weights[:, np.newaxis], axis=0) / np.sum(weights)
            # Move toward centroid but keep inside circle
            new_centers[i] = centers[i] + 0.5 * (centroid - centers[i])
            norm = np.linalg.norm(new_centers[i])
            if norm > 0.95:
                new_centers[i] *= 0.95 / norm
        centers = new_centers
    return centers

def ring_init():
    """Initialize with ring structure: 1 center + 5 rings."""
    centers = np.zeros((N, 2))
    centers[0] = [0, 0]
    remaining = N - 1
    ring_sizes = []
    for r in range(1, 6):
        size = min(remaining, int(2 * np.pi * r * 0.3))
        if size > 0:
            ring_sizes.append(size)
            remaining -= size
    # Distribute remaining
    if remaining > 0:
        ring_sizes[-1] += remaining
    
    idx = 1
    for ring_idx, size in enumerate(ring_sizes):
        radius = 0.3 + ring_idx * 0.12
        angles = np.linspace(0, 2*np.pi, size, endpoint=False)
        for k in range(size):
            centers[idx] = [radius * np.cos(angles[k]), radius * np.sin(angles[k])]
            idx += 1
    return centers

def hex_init():
    """Hexagonal lattice initialization."""
    spacing = 0.22
    points = []
    for j in range(-8, 9):
        for i in range(-8, 9):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3) / 2
            if x*x + y*y <= 0.85:
                points.append([x, y])
    points = np.array(points)
    if len(points) >= N:
        # Select N closest to origin
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        return points[idx]
    return np.array(points)

def anneal_optimize(centers, max_iters=2000, t0=0.05, t_end=0.001):
    """Simulated annealing to improve packing."""
    current = centers.copy()
    current_r = compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        t = t0 * (t_end / t0) ** (it / max_iters)
        
        # Random perturbation
        noise = np.random.randn(N, 2) * t
        candidate = current + noise
        
        # Keep inside circle
        dists = np.linalg.norm(candidate, axis=1)
        mask = dists > 0.98
        candidate[mask] *= 0.98 / dists[mask, np.newaxis]
        
        cand_r = compute_radius(candidate)
        
        if cand_r >= current_r or np.random.rand() < np.exp((cand_r - current_r) / (t + 1e-10)):
            current = candidate
            current_r = cand_r
            if current_r > best_r:
                best = current.copy()
                best_r = current_r
    
    return best, best_r

def polish(centers):
    """Final polish with scipy optimization."""
    x0 = centers.flatten()
    
    def obj(x):
        return objective(x, t=0.01)
    
    def grad(x):
        return gradient_obj(x, t=0.01)
    
    try:
        res = minimize(obj, x0, jac=grad, method='L-BFGS-B', 
                       options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < obj(x0):
            return res.x.reshape(N, 2)
    except:
        pass
    return centers

def construct():
    start_time = time.time()
    
    # Multiple initialization strategies
    initializations = []
    
    # 1. Hexagonal lattice
    initializations.append(hex_init())
    
    # 2. Ring structure
    initializations.append(ring_init())
    
    # 3. Lloyd's algorithm
    for _ in range(3):
        initializations.append(lloyd_init(30))
    
    # 4. Random with symmetry
    for _ in range(3):
        centers = np.random.randn(N, 2) * 0.5
        dists = np.linalg.norm(centers, axis=1)
        mask = dists > 0.9
        centers[mask] *= 0.9 / dists[mask, np.newaxis]
        initializations.append(centers)
    
    best_centers = None
    best_radius = 0
    
    for init in initializations:
        if time.time() > DEADLINE - 10:
            break
        
        # Anneal
        centers, radius = anneal_optimize(init, max_iters=1500, t0=0.03, t_end=0.0005)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    # Polish the best result
    if best_centers is not None and time.time() < DEADLINE - 5:
        polished = polish(best_centers)
        polished_r = compute_radius(polished)
        if polished_r > best_radius:
            best_centers = polished
            best_radius = polished_r
    
    if best_centers is None:
        # Fallback: simple hex
        best_centers = hex_init()
    
    return best_centers
