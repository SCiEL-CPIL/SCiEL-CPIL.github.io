"""Rewritten program: optimize 50 circle centres in the unit disk via ring-based init + L-BFGS-B + local polish."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _score(positions):
    """Compute the packing radius (minimum of pairwise distances/2 and distance to boundary)."""
    n = len(positions)
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Distance to boundary
    radii = np.hypot(positions[:, 0], positions[:, 1])
    min_boundary = 1.0 - np.max(radii)
    
    return min(min_pair, min_boundary)


def _ring_init(n, center=True):
    """Generate initial positions on concentric rings matching known optimal structure."""
    # Known best: 1 center + rings of 6, 12, 18, 13 (total 50)
    # Ring radii approximately: 0.25, 0.5, 0.75, 0.95
    if center:
        positions = [[0.0, 0.0]]
        remaining = n - 1
    else:
        positions = []
        remaining = n
    
    # Distribute remaining circles among rings
    # Try: 6, 12, 18, 13 for 49 remaining
    ring_sizes = [6, 12, 18, 13]
    ring_radii = [0.25, 0.5, 0.75, 0.95]
    
    idx = 0
    for ring_n, ring_r in zip(ring_sizes, ring_radii):
        if idx >= remaining:
            break
        count = min(ring_n, remaining - idx)
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        for a in angles:
            positions.append([ring_r * np.cos(a), ring_r * np.sin(a)])
        idx += count
    
    # If we still need more, add another outer ring
    while idx < remaining:
        ring_n = 6
        ring_r = 0.98
        count = min(ring_n, remaining - idx)
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        for a in angles:
            positions.append([ring_r * np.cos(a), ring_r * np.sin(a)])
        idx += count
    
    return np.array(positions[:n], dtype=np.float64)


def _random_init(n):
    """Random initialization inside the unit disk."""
    angles = np.random.uniform(0, 2 * np.pi, n)
    radii = np.sqrt(np.random.uniform(0, 1, n))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _hex_init(n):
    """Hexagonal lattice initialization."""
    spacing = 0.22
    points = []
    k = 12
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r < 0.95:
                points.append((x, y))
    
    points = np.array(points)
    dists = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(dists, kind='stable')[:n]
    return points[idx].copy()


def _objective_flat(x, n):
    """Objective function for L-BFGS-B: negative of the packing radius."""
    pos = x.reshape(-1, 2)
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Distance to boundary
    radii = np.hypot(pos[:, 0], pos[:, 1])
    min_boundary = 1.0 - np.max(radii)
    
    r = min(min_pair, min_boundary)
    return -r


def _lbfgs_optimize(positions, max_time=60.0):
    """Optimize using L-BFGS-B with time limit."""
    start_time = time.time()
    n = len(positions)
    x0 = positions.flatten()
    
    # Bounds: each coordinate in [-1, 1]
    bounds = [(-1.0, 1.0)] * (2 * n)
    
    result = minimize(
        _objective_flat, x0, method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
    )
    
    pos = result.x.reshape(-1, 2)
    # Project inside unit disk
    radii = np.hypot(pos[:, 0], pos[:, 1])
    mask = radii > 1.0
    if np.any(mask):
        pos[mask] *= 1.0 / radii[mask]
    
    return pos, -result.fun


def _local_polish(positions, r_init, max_time=30.0):
    """Fine-tune with local force-based optimization."""
    start_time = time.time()
    pos = positions.copy()
    n = len(pos)
    best_r = r_init
    best_pos = pos.copy()
    step = 0.01
    lr = 0.999
    
    while time.time() - start_time < max_time:
        forces = np.zeros_like(pos)
        
        # Pairwise repulsion for close pairs
        for i in range(n):
            for j in range(i + 1, n):
                diff = pos[i] - pos[j]
                dist = np.sqrt(np.sum(diff ** 2))
                if dist < 1e-10:
                    forces[i] += np.random.randn(2) * 0.001
                    forces[j] -= np.random.randn(2) * 0.001
                    continue
                min_dist = 2.0 * best_r
                if dist < min_dist * 1.01:
                    force_mag = (min_dist - dist) / max(dist, 1e-10)
                    forces[i] += force_mag * diff / dist
                    forces[j] -= force_mag * diff / dist
        
        # Boundary
        for i in range(n):
            r = np.hypot(pos[i, 0], pos[i, 1])
            max_r = 1.0 - best_r
            if r > max_r:
                force_mag = (r - max_r) / max(r, 1e-10)
                forces[i] -= force_mag * pos[i] / r
        
        pos += step * forces
        
        # Hard constraint
        radii = np.hypot(pos[:, 0], pos[:, 1])
        max_allowed = 1.0 - best_r * 0.999
        mask = radii > max_allowed
        if np.any(mask):
            pos[mask] *= max_allowed / radii[mask]
        
        new_r = _score(pos)
        if new_r > best_r:
            best_r = new_r
            best_pos = pos.copy()
        
        step *= lr
        if step < 0.0005:
            break
    
    return best_pos, best_r


def construct():
    """Construct positions for N circles in the unit disk."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_positions = None
    best_r = 0.0
    
    # Strategy 1: Ring-based init + L-BFGS-B
    if time.time() < deadline - 70.0:
        pos1 = _ring_init(N, center=True)
        pos1, r1 = _lbfgs_optimize(pos1, max_time=55.0)
        if r1 > best_r:
            best_r = r1
            best_positions = pos1.copy()
    
    # Strategy 2: Hex lattice init + L-BFGS-B
    if time.time() < deadline - 70.0:
        pos2 = _hex_init(N)
        pos2, r2 = _lbfgs_optimize(pos2, max_time=55.0)
        if r2 > best_r:
            best_r = r2
            best_positions = pos2.copy()
    
    # Strategy 3: Random init + L-BFGS-B
    if time.time() < deadline - 70.0:
        np.random.seed(42)
        pos3 = _random_init(N)
        pos3, r3 = _lbfgs_optimize(pos3, max_time=55.0)
        if r3 > best_r:
            best_r = r3
            best_positions = pos3.copy()
    
    # Local polish on best
    if best_positions is not None and time.time() < deadline - 10.0:
        best_positions, best_r = _local_polish(best_positions, best_r, max_time=25.0)
    
    # Final validation
    if best_positions is None:
        best_positions = _ring_init(N, center=True)
        best_r = _score(best_positions)
    
    radii = np.hypot(best_positions[:, 0], best_positions[:, 1])
    max_allowed = 1.0 - best_r * 0.999
    mask = radii > max_allowed
    if np.any(mask):
        best_positions[mask] *= max_allowed / radii[mask]
    
    return best_positions
# EVOLVE-BLOCK-END
