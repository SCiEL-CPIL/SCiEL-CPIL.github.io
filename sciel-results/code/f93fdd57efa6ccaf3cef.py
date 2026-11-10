"""Morse cluster global minima: Basin Hopping with diverse structural seeds."""
import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global time tracking
_TOTAL_DEADLINE = 300.0
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _remaining():
    return _TOTAL_DEADLINE - _elapsed()


def _morse_energy(positions, rho):
    """Compute Morse potential energy."""
    n = positions.shape[0]
    d = squareform(pdist(positions))
    np.fill_diagonal(d, 1.0)
    r = np.where(d > 0, d, 1.0)
    x = rho * (1.0 - r)
    ex = np.exp(x)
    v = ex * (ex - 2.0)
    energy = 0.5 * np.sum(v)
    return energy


def _morse_energy_grad(positions, rho):
    """Compute Morse potential energy and gradient."""
    n = positions.shape[0]
    d = squareform(pdist(positions))
    np.fill_diagonal(d, 1.0)
    r = np.where(d > 0, d, 1.0)
    x = rho * (1.0 - r)
    ex = np.exp(x)
    v = ex * (ex - 2.0)
    energy = 0.5 * np.sum(v)
    
    # Gradient: dV/dr * dr/dx_i
    dv_dr = -rho * ex * (2.0 * ex - 1.0)
    # For each pair (i,j), gradient contribution to atom i is dv_dr * (r_ij_vec / r_ij)
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i+1, n):
            rij = positions[i] - positions[j]
            dij = np.linalg.norm(rij)
            if dij > 1e-10:
                g = dv_dr[i, j] / dij * rij
                grad[i] += g
                grad[j] -= g
    return energy, grad


def _fcc_cluster(n, spacing=1.0):
    """Generate FCC-like cluster."""
    a = spacing
    points = []
    # Generate FCC lattice points
    for i in range(4):
        for j in range(4):
            for k in range(4):
                p = np.array([i, j, k]) * a
                points.append(p)
                points.append(p + np.array([a/2, a/2, 0]) * a)
                points.append(p + np.array([a/2, 0, a/2]) * a)
                points.append(p + np.array([0, a/2, a/2]) * a)
    points = np.array(points)
    # Center
    points -= points.mean(axis=0)
    # Sort by distance from center, take n closest
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points


def _bcc_cluster(n, spacing=1.0):
    """Generate BCC-like cluster."""
    a = spacing
    points = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                p = np.array([i, j, k]) * a
                points.append(p)
                points.append(p + np.array([a/2, a/2, a/2]) * a)
    points = np.array(points)
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points


def _hcp_cluster(n, spacing=1.0):
    """Generate HCP-like cluster."""
    a = spacing
    c = a * np.sqrt(8.0/3.0)
    points = []
    for layer in range(4):
        z = layer * c
        if layer % 2 == 0:
            for i in range(4):
                for j in range(4):
                    x = i * a + (j % 2) * a/2
                    y = j * a * np.sqrt(3.0)/2.0
                    points.append(np.array([x, y, z]))
        else:
            for i in range(4):
                for j in range(4):
                    x = i * a + a/2
                    y = j * a * np.sqrt(3.0)/2.0 + a * np.sqrt(3.0)/6.0
                    points.append(np.array([x, y, z]))
    points = np.array(points)
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points


def _icosahedral_shell(n, spacing=1.0):
    """Generate icosahedral shell cluster."""
    # Golden ratio
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # 12 vertices of icosahedron
    verts = []
    for i in range(-1, 2, 2):
        for j in range(-1, 2, 2):
            for k in range(-1, 2, 2):
                if i != 0 or j != 0 or k != 0:
                    pass
    # Standard icosahedron vertices
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    verts = np.array(verts, dtype=float)
    verts /= np.linalg.norm(verts[0]) * spacing
    
    points = [np.zeros(3)]  # center
    points.extend(verts)  # first shell
    
    # Second shell (30 points) - cube vertices
    s = 1.0
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                points.append(np.array([i, j, k]) * s * spacing * 1.5)
    
    # Third shell if needed
    while len(points) < n:
        # Add more points on a sphere
        theta = np.random.uniform(0, 2*np.pi)
        z = np.random.uniform(-1, 1)
        r = np.sqrt(1-z*z)
        p = np.array([r*np.cos(theta), r*np.sin(theta), z]) * spacing * 2.5
        points.append(p)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points


def _random_cluster(n, spacing=1.0, seed=None):
    """Generate random cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
        attempts += 1
    # If not enough points, add more randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    points = np.array(points)
    points -= points.mean(axis=0)
    return points


def _lbfgs_refine(positions, rho, maxiter=200):
    """Refine positions using L-BFGS-B."""
    n = positions.shape[0]
    
    def fun_and_grad(x):
        pos = x.reshape(n, 3)
        e, g = _morse_energy_grad(pos, rho)
        return e, g
    
    result = minimize(fun_and_grad, positions.ravel(), method='L-BFGS-B',
                      jac=True, options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8})
    return result.x.reshape(n, 3), result.fun


def _basin_hopping(positions, rho, n_iterations=50, step_size=0.3, seed=None):
    """Basin hopping algorithm."""
    rng = np.random.default_rng(seed)
    n = positions.shape[0]
    current_pos = positions.copy()
    current_energy, _ = _morse_energy_grad(current_pos, rho)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    for i in range(n_iterations):
        if _remaining() < 5.0:
            break
        # Perturb
        perturbed = current_pos + rng.normal(0, step_size, size=current_pos.shape)
        
        # Relax
        try:
            relaxed, e = _lbfgs_refine(perturbed, rho, maxiter=100)
        except:
            continue
        
        if e < current_energy:
            current_pos = relaxed
            current_energy = e
        else:
            # Metropolis accept
            delta = e - current_energy
            if delta < 0 or rng.random() < np.exp(-delta / 0.5):
                current_pos = relaxed
                current_energy = e
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
    
    return best_pos, best_energy


def _check_valid(positions, min_dist=0.5):
    """Check if all pairs are at least min_dist apart."""
    d = pdist(positions)
    return np.min(d) >= min_dist


def construct(instance):
    """Find global minimum for Morse cluster."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    # Generate diverse initial structures
    seeds = []
    seeds.append(('fcc', _fcc_cluster(n, spacing=1.0)))
    seeds.append(('bcc', _bcc_cluster(n, spacing=1.0)))
    seeds.append(('hcp', _hcp_cluster(n, spacing=1.0)))
    seeds.append(('ico', _icosahedral_shell(n, spacing=1.0)))
    for i in range(3):
        seeds.append((f'rand{i}', _random_cluster(n, spacing=1.0, seed=42+i)))
    
    best_energy = np.inf
    best_pos = None
    
    # First pass: relax all seeds
    for name, pos in seeds:
        if _remaining() < 10.0:
            break
        try:
            relaxed, e = _lbfgs_refine(pos, rho, maxiter=200)
            if _check_valid(relaxed, 0.5) and e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
        except:
            pass
    
    # Second pass: basin hopping on best candidates
    if _remaining() > 20.0:
        # Sort seeds by energy after relaxation
        relaxed_seeds = []
        for name, pos in seeds:
            if _remaining() < 15.0:
                break
            try:
                relaxed, e = _lbfgs_refine(pos, rho, maxiter=100)
                relaxed_seeds.append((e, relaxed))
            except:
                pass
        
        relaxed_seeds.sort(key=lambda x: x[0])
        
        # Basin hop on top 3
        for e, pos in relaxed_seeds[:3]:
            if _remaining() < 10.0:
                break
            try:
                bh_pos, bh_e = _basin_hopping(pos, rho, n_iterations=30, step_size=0.2, seed=0)
                if _check_valid(bh_pos, 0.5) and bh_e < best_energy:
                    best_energy = bh_e
                    best_pos = bh_pos.copy()
            except:
                pass
    
    # Final refinement
    if best_pos is not None and _remaining() > 5.0:
        try:
            final_pos, final_e = _lbfgs_refine(best_pos, rho, maxiter=300)
            if _check_valid(final_pos, 0.5):
                best_pos = final_pos
                best_energy = final_e
        except:
            pass
    
    if best_pos is None:
        # Fallback: random
        best_pos = _random_cluster(n, spacing=1.0, seed=123)
    
    return best_pos
