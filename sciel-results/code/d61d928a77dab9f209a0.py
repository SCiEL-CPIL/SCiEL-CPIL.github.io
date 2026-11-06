"""Morse cluster global minima: lattice-based initialization with gradient-guided multi-restart relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 280.0  # 20s safety margin before 300s kill
_instance_call_count = [0]


def _time_left():
    return _HARD_DEADLINE - time.time()


def _fcc_points(n, spacing=1.0):
    """Generate n points from an FCC lattice centered at origin."""
    # For FCC, nearest neighbor distance = a/sqrt(2) where a is cubic lattice constant
    # We want nearest neighbor ~ spacing, so a = spacing * sqrt(2)
    a = spacing * np.sqrt(2.0)
    
    # Estimate radius needed for n atoms
    # FCC density: 4 atoms per cell of volume a^3
    density = 4.0 / (a ** 3)
    vol_needed = n / density
    r_est = (3.0 * vol_needed / (4.0 * np.pi)) ** (1.0 / 3.0)
    r_est *= 1.3  # add margin
    
    points = []
    extent = int(np.ceil(r_est / a)) + 2
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                # Corner
                p = np.array([i, j, k]) * a
                if np.linalg.norm(p) <= r_est:
                    points.append(p)
                # Face centers
                if j < extent and k < extent:
                    p2 = np.array([i, j + 0.5, k + 0.5]) * a
                    if np.linalg.norm(p2) <= r_est:
                        points.append(p2)
                if i < extent and k < extent:
                    p3 = np.array([i + 0.5, j, k + 0.5]) * a
                    if np.linalg.norm(p3) <= r_est:
                        points.append(p3)
                if i < extent and j < extent:
                    p4 = np.array([i + 0.5, j + 0.5, k]) * a
                    if np.linalg.norm(p4) <= r_est:
                        points.append(p4)
    
    if len(points) == 0:
        rng = np.random.default_rng(42)
        return rng.uniform(-r_est, r_est, (n, 3))
    
    points = np.array(points)
    # Sort by distance from origin
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx]
    
    if len(points) >= n:
        return points[:n]
    else:
        rng = np.random.default_rng(42)
        extra = rng.uniform(-r_est, r_est, (n - len(points), 3))
        return np.vstack([points, extra])


def _bcc_points(n, spacing=1.0):
    """Generate n points from a BCC lattice centered at origin."""
    # BCC: nearest neighbor = a*sqrt(3)/2, so a = 2*spacing/sqrt(3)
    a = 2.0 * spacing / np.sqrt(3.0)
    
    r_est = spacing * (n ** (1.0 / 3.0) + 2) * 0.8
    
    points = []
    extent = int(np.ceil(r_est / a)) + 2
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                # Corner
                p = np.array([i, j, k]) * a
                if np.linalg.norm(p) <= r_est:
                    points.append(p)
                # Body center
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5]) * a
                if np.linalg.norm(p2) <= r_est:
                    points.append(p2)
    
    if len(points) == 0:
        rng = np.random.default_rng(42)
        return rng.uniform(-r_est, r_est, (n, 3))
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx]
    
    if len(points) >= n:
        return points[:n]
    else:
        rng = np.random.default_rng(42)
        extra = rng.uniform(-r_est, r_est, (n - len(points), 3))
        return np.vstack([points, extra])


def _sc_points(n, spacing=1.0):
    """Generate n points from a simple cubic lattice centered at origin."""
    a = spacing
    r_est = spacing * (n ** (1.0 / 3.0) + 2) * 0.8
    
    points = []
    extent = int(np.ceil(r_est / a)) + 2
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                p = np.array([i, j, k]) * a
                if np.linalg.norm(p) <= r_est:
                    points.append(p)
    
    if len(points) == 0:
        rng = np.random.default_rng(42)
        return rng.uniform(-r_est, r_est, (n, 3))
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx]
    
    if len(points) >= n:
        return points[:n]
    else:
        rng = np.random.default_rng(42)
        extra = rng.uniform(-r_est, r_est, (n - len(points), 3))
        return np.vstack([points, extra])


def _random_compact(n, spacing=1.0, seed=0):
    """Generate n random points in a compact sphere with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    min_dist = 0.55 * spacing
    
    points = []
    attempts = 0
    max_attempts = n * 300
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    ok = False
                    break
            if ok:
                points.append(p)
        attempts += 1
    
    if len(points) < n:
        # Fill remaining randomly
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    
    return np.array(points)


def _gradient_descent_refine(positions, potential, rho, max_iter=50, lr_init=0.01):
    """Refine positions using gradient descent with momentum."""
    pos = positions.copy()
    lr = lr_init
    momentum = np.zeros_like(pos)
    best_pos = pos.copy()
    best_energy = np.inf
    
    for it in range(max_iter):
        if _time_left() < 2.0:
            break
        try:
            energy, grad = morse_energy_gradient(pos, rho)
        except Exception:
            break
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
        
        # Update with momentum
        momentum = 0.9 * momentum - lr * grad
        pos = pos + momentum
        
        # Decay learning rate
        lr *= 0.95
        
        # Check convergence
        if np.max(np.abs(grad)) < 1e-6:
            break
    
    return best_pos, best_energy


def construct(instance):
    """Find low-energy Morse cluster configurations via multi-restart lattice-based search with gradient refinement."""
    _instance_call_count[0] += 1
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Estimate per-instance time budget
    per_instance_budget = 42.0
    if _instance_call_count[0] <= 2:
        per_instance_budget = 48.0
    elif _instance_call_count[0] >= 5:
        per_instance_budget = 38.0
    
    instance_deadline = time.time() + per_instance_budget
    
    # Generate multiple starting configurations from different lattices
    start_configs = []
    
    # FCC
    try:
        fcc = _fcc_points(size, spacing=1.0)
        start_configs.append(fcc)
    except Exception:
        pass
    
    # BCC
    try:
        bcc = _bcc_points(size, spacing=1.0)
        start_configs.append(bcc)
    except Exception:
        pass
    
    # Simple cubic
    try:
        sc = _sc_points(size, spacing=1.0)
        start_configs.append(sc)
    except Exception:
        pass
    
    # Random compact with different seeds
    for seed in [0, 1, 2, 3, 4, 5]:
        try:
            rc = _random_compact(size, spacing=1.0, seed=seed)
            start_configs.append(rc)
        except Exception:
            pass
    
    # Ensure we have at least some starts
    if len(start_configs) == 0:
        start_configs.append(_random_compact(size, spacing=1.0, seed=0))
    
    # Relax each starting configuration, keeping the best
    best_energy = np.inf
    best_positions = None
    
    for idx, config in enumerate(start_configs):
        # Check time
        if time.time() > instance_deadline - 3.0:
            break
        
        try:
            energy, positions = relax(config, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    # If we have time, do gradient descent refinement on the best result
    if best_positions is not None and time.time() < instance_deadline - 5.0:
        try:
            refined_pos, refined_energy = _gradient_descent_refine(
                best_positions, potential, parameter, max_iter=30, lr_init=0.005
            )
            if refined_energy < best_energy:
                best_energy = refined_energy
                best_positions = refined_pos
        except Exception:
            pass
    
    # Additional perturbations of the best result
    if best_positions is not None and time.time() < instance_deadline - 4.0:
        rng = np.random.default_rng(12345)
        for _ in range(8):
            if time.time() > instance_deadline - 2.0:
                break
            try:
                perturbed = best_positions + rng.uniform(-0.06, 0.06, best_positions.shape)
                energy, positions = relax(perturbed, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                continue
    
    # Final safety check: ensure minimum distance constraint
    if best_positions is None:
        best_positions = _sc_points(size, spacing=1.0)
    
    # Verify minimum distance
    n = len(best_positions)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(best_positions[i] - best_positions[j])
            if d < 0.5:
                mid = 0.5 * (best_positions[i] + best_positions[j])
                direction = best_positions[i] - best_positions[j]
                norm = np.linalg.norm(direction)
                if norm < 1e-10:
                    direction = np.array([1.0, 0.0, 0.0])
                    norm = 1.0
                direction /= norm
                offset = 0.26 * direction
                best_positions[i] = mid + offset
                best_positions[j] = mid - offset
    
    return best_positions
# EVOLVE-BLOCK-END
