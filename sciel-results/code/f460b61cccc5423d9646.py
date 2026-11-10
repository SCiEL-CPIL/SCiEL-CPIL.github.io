"""Morse cluster global minima search with crystalline starting points and gradient-based relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_TIME_LIMIT = 290.0  # 300s limit, keep 10s safety margin


def _morse_energy_and_grad(positions, rho):
    """Compute exact Morse energy and gradient."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-12)
    
    # Mask out self-interactions
    mask = ~np.eye(n, dtype=bool)
    
    exp_term = np.exp(rho * (1.0 - dist))
    energy_pairs = exp_term * (exp_term - 2.0)
    energy = 0.5 * np.sum(energy_pairs * mask)
    
    # Gradient: dE/dr * dr/dx
    # dE/dr = rho * exp_term * (2*exp_term - 2)
    dE_dr = rho * exp_term * (2.0 * exp_term - 2.0)
    dE_dr = dE_dr * mask
    dE_dr = 0.5 * dE_dr  # Symmetrize
    
    # dr/dx_i = (x_i - x_j) / r
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(n):
            if i != j:
                r = dist[i, j]
                if r > 1e-8:
                    grad[i] += dE_dr[i, j] * (positions[i] - positions[j]) / r
    return energy, grad


def _generate_fcc(n, spacing=1.0):
    """Generate FCC lattice points cropped to a sphere."""
    points = []
    # FCC basis vectors
    a = spacing * 2.0
    # Generate lattice points in a cube
    range_val = int(np.ceil(n ** (1.0/3.0))) + 2
    for i in range(-range_val, range_val + 1):
        for j in range(-range_val, range_val + 1):
            for k in range(-range_val, range_val + 1):
                # FCC: add face-centered positions
                for offset in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = np.array([i + offset[0], j + offset[1], k + offset[2]]) * a
                    points.append(p)
    
    points = np.array(points)
    if len(points) == 0:
        return np.zeros((n, 3))
    
    # Center and sort by distance from origin
    center = np.mean(points, axis=0)
    points -= center
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order]
    
    # Take first n points
    if len(points) >= n:
        return points[:n]
    else:
        # Pad with random points if needed
        remaining = n - len(points)
        radius = np.max(dists) + 1.0
        rng = np.random.default_rng(42)
        extra = rng.uniform(-radius, radius, (remaining, 3))
        return np.vstack([points, extra])


def _generate_bcc(n, spacing=1.0):
    """Generate BCC lattice points cropped to a sphere."""
    points = []
    a = spacing * 2.0
    range_val = int(np.ceil(n ** (1.0/3.0))) + 2
    for i in range(-range_val, range_val + 1):
        for j in range(-range_val, range_val + 1):
            for k in range(-range_val, range_val + 1):
                # BCC: corner + body center
                for offset in [(0, 0, 0), (0.5, 0.5, 0.5)]:
                    p = np.array([i + offset[0], j + offset[1], k + offset[2]]) * a
                    points.append(p)
    
    points = np.array(points)
    if len(points) == 0:
        return np.zeros((n, 3))
    
    center = np.mean(points, axis=0)
    points -= center
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order]
    
    if len(points) >= n:
        return points[:n]
    else:
        remaining = n - len(points)
        radius = np.max(dists) + 1.0
        rng = np.random.default_rng(42)
        extra = rng.uniform(-radius, radius, (remaining, 3))
        return np.vstack([points, extra])


def _generate_hcp(n, spacing=1.0):
    """Generate HCP-like lattice points."""
    points = []
    a = spacing * 1.63  # HCP a parameter
    c = a * np.sqrt(8.0/3.0)
    range_val = int(np.ceil(n ** (1.0/3.0))) + 2
    
    for i in range(-range_val, range_val + 1):
        for j in range(-range_val, range_val + 1):
            for k in range(-range_val, range_val + 1):
                # HCP: two atoms per unit cell
                # Atom 1: (0, 0, 0)
                p1 = np.array([i * a, j * a, k * c])
                points.append(p1)
                # Atom 2: (2/3, 1/3, 1/2) in hexagonal coords -> convert
                x = (i + 2.0/3.0) * a
                y = (j + 1.0/3.0) * a
                z = (k + 0.5) * c
                p2 = np.array([x, y, z])
                points.append(p2)
    
    points = np.array(points)
    if len(points) == 0:
        return np.zeros((n, 3))
    
    center = np.mean(points, axis=0)
    points -= center
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order]
    
    if len(points) >= n:
        return points[:n]
    else:
        remaining = n - len(points)
        radius = np.max(dists) + 1.0
        rng = np.random.default_rng(42)
        extra = rng.uniform(-radius, radius, (remaining, 3))
        return np.vstack([points, extra])


def _generate_shell(n, spacing=1.0):
    """Generate points on concentric shells."""
    points = []
    rng = np.random.default_rng(42)
    
    # Estimate shell radii
    for shell in range(1, 10):
        r = shell * spacing * 0.8
        # Number of points on shell ~ 4*pi*r^2 / spacing^2
        n_shell = int(4 * np.pi * r**2 / (spacing * spacing))
        n_shell = min(n_shell, n - len(points))
        if n_shell <= 0:
            break
        # Generate random points on sphere
        for _ in range(n_shell):
            v = rng.normal(size=3)
            v = v / np.linalg.norm(v)
            points.append(v * r)
        if len(points) >= n:
            break
    
    if len(points) < n:
        remaining = n - len(points)
        radius = max(len(points) ** (1.0/3.0) * spacing, 1.0)
        extra = rng.uniform(-radius, radius, (remaining, 3))
        points.extend(extra)
    
    return np.array(points[:n])


def _relax_with_gradient(positions, rho, max_iter=500):
    """Relax positions using L-BFGS-B with exact gradient."""
    def objective(x):
        pos = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(pos, rho)
        return e, g.ravel()
    
    x0 = positions.ravel()
    result = minimize(objective, x0, method='L-BFGS-B', jac=True,
                      options={'maxiter': max_iter, 'gtol': 1e-6, 'maxls': 20})
    return result.fun, result.x.reshape(-1, 3)


def _is_valid(positions, min_dist=0.5):
    """Check if all pairwise distances are >= min_dist."""
    n = len(positions)
    if n < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    return np.min(dist) >= min_dist - 1e-6


def construct(instance):
    """Best of crystalline starting clusters, each relaxed with gradient-based optimization."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    if time.time() - _START_TIME > _TIME_LIMIT - 5:
        # Emergency return
        return np.zeros((size, 3))
    
    generators = [_generate_fcc, _generate_bcc, _generate_hcp, _generate_shell]
    best_energy = np.inf
    best_positions = None
    
    for gen in generators:
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        
        try:
            initial = gen(size)
            # Ensure valid minimum distances
            if not _is_valid(initial):
                # Nudge points apart
                rng = np.random.default_rng(42)
                for i in range(size):
                    for j in range(size):
                        if i != j:
                            d = np.linalg.norm(initial[i] - initial[j])
                            if d < 0.5:
                                initial[i] += rng.normal(scale=0.1, size=3)
            
            energy, positions = _relax_with_gradient(initial, rho, max_iter=300)
            
            if not _is_valid(positions):
                # Fallback to relax()
                energy, positions = relax(initial, potential, rho)
            
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue
    
    if best_positions is None:
        # Final fallback
        rng = np.random.default_rng(size)
        radius = size ** (1.0/3.0)
        points = rng.uniform(-radius, radius, (size, 3))
        best_positions, _ = relax(points, potential, rho)
    
    return best_positions
# EVOLVE-BLOCK-END
