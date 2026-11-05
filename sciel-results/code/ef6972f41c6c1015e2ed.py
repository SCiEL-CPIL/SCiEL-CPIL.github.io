"""Dzugutov cluster global minima search using polytetrahedral seeding and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

def _dzugutov_energy_and_grad(pos):
    """Compute Dzugutov energy and gradient for positions array of shape (n, 3)."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    
    # Pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # Avoid division by zero
    dist_safe = np.where(dist < 1e-10, 1e-10, dist)
    
    # Build energy and gradient
    energy = 0.0
    grad = np.zeros_like(pos)
    
    # Only compute for i < j to avoid double counting
    for i in range(n):
        for j in range(i + 1, n):
            r = dist[i, j]
            if r < 1e-10:
                continue
            if r < a:
                v = A * (r**(-m) - B) * np.exp(c / (r - a))
                # dV/dr
                dv_dr = A * (-m * r**(-m-1)) * np.exp(c / (r - a)) + A * (r**(-m) - B) * np.exp(c / (r - a)) * (-c / (r - a)**2)
            elif r < b:
                v = B * np.exp(d / (r - b))
                dv_dr = B * np.exp(d / (r - a)) * (-d / (r - b)**2)  # Note: should be (r-b)
                dv_dr = B * np.exp(d / (r - b)) * (-d / (r - b)**2)
            else:
                v = 0.0
                dv_dr = 0.0
            
            energy += v
            # Gradient: dV/dr * dr/dx
            # dr/dx_i = (pos[i] - pos[j]) / r
            rij = (pos[i] - pos[j]) / r
            grad[i] += dv_dr * rij
            grad[j] -= dv_dr * rij
    
    return energy, grad


def _dzugutov_energy(pos):
    """Compute just the energy."""
    e, _ = _dzugutov_energy_and_grad(pos)
    return e


def _relax(pos, max_iter=1000, tol=1e-8):
    """Relax positions using L-BFGS-B."""
    def fun(x):
        e, g = _dzugutov_energy_and_grad(x.reshape(-1, 3))
        return e, g
    
    result = minimize(fun, pos.flatten(), method='L-BFGS-B', jac=True, 
                      options={'maxiter': max_iter, 'gtol': tol})
    return result.x.reshape(-1, 3), result.fun


def _generate_polytetrahedral_seed(n, rng):
    """Generate a polytetrahedral-like seed structure."""
    # Start with icosahedral structure as base
    # Icosahedron has 12 vertices + 1 center = 13 atoms
    # Build larger clusters by adding shells
    
    phi = (1 + math.sqrt(5)) / 2
    # Icosahedron vertices (unit edge)
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    # Normalize to unit edge length
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0] - ico_verts[1])
    
    positions = [np.zeros(3)]  # Center atom
    positions.extend(ico_verts)  # 12 shell atoms
    
    # Add more atoms in polytetrahedral fashion
    # Use a simple growth: add atoms at positions that maintain local order
    while len(positions) < n:
        # Pick a random existing atom and add a new one nearby
        idx = rng.integers(len(positions))
        base = positions[idx]
        # Random direction
        theta = rng.uniform(0, 2 * math.pi)
        phi_angle = rng.uniform(0, math.pi)
        direction = np.array([
            math.sin(phi_angle) * math.cos(theta),
            math.sin(phi_angle) * math.sin(theta),
            math.cos(phi_angle)
        ])
        # Distance ~ 1.0 (optimal for Dzugutov)
        new_pos = base + direction * 1.0
        positions.append(new_pos)
    
    return np.array(positions[:n])


def _generate_random_compact_seed(n, rng, spacing=1.1):
    """Generate a random compact seed."""
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check minimum distance
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
    return np.array(points)


def _basin_hopping(pos, rng, n_iterations=50, step_size=0.5):
    """Simple basin hopping."""
    best_pos, best_energy = pos.copy(), _dzugutov_energy(pos)
    
    for _ in range(n_iterations):
        # Perturb
        perturbed = pos + rng.normal(0, step_size, pos.shape)
        # Relax
        relaxed, energy = _relax(perturbed, max_iter=500)
        
        if energy < best_energy:
            best_pos, best_energy = relaxed.copy(), energy
            pos = relaxed  # Continue from new best
        else:
            # Metropolis acceptance
            delta = energy - best_energy
            if rng.random() < math.exp(-delta / 0.5):
                pos = relaxed
            else:
                pos = best_pos  # Revert
    
    return best_pos, best_energy


def construct(instance):
    """Find low-energy Dzugutov cluster structures."""
    start_time = time.time()
    deadline = start_time + 280  # 300s limit with 20s margin
    
    n = instance["n"]
    rng = np.random.default_rng(n * 7919)  # Seed based on n
    
    # Generate multiple initial seeds
    seeds = []
    
    # Polytetrahedral seed
    seeds.append(_generate_polytetrahedral_seed(n, rng))
    
    # Random compact seeds
    for _ in range(3):
        seeds.append(_generate_random_compact_seed(n, rng))
    
    # Relax all seeds and keep the best
    best_energy = float('inf')
    best_pos = None
    
    for seed in seeds:
        if time.time() > deadline - 30:
            break
        try:
            relaxed, energy = _relax(seed, max_iter=1000)
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed.copy()
        except Exception:
            continue
    
    if best_pos is None:
        # Fallback: return the first seed
        best_pos = seeds[0]
        best_energy = _dzugutov_energy(best_pos)
    
    # Basin hopping from the best structure
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        try:
            best_pos, best_energy = _basin_hopping(best_pos, rng, n_iterations=30, step_size=0.4)
        except Exception:
            pass
    
    # Final check: ensure minimum distance constraint
    # If any atoms are too close, perturb them slightly
    for _ in range(100):
        min_dist = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(best_pos[i] - best_pos[j])
                if d < min_dist:
                    min_dist = d
        if min_dist >= 0.5:
            break
        # Nudge the closest pair apart
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(best_pos[i] - best_pos[j])
                if d < 0.5 and d > 1e-10:
                    direction = (best_pos[i] - best_pos[j]) / d
                    shift = (0.5 - d + 0.01) / 2
                    best_pos[i] += direction * shift
                    best_pos[j] -= direction * shift
                    break
    
    return best_pos
# EVOLVE-BLOCK-END
