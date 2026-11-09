"""
Optimized Morse Cluster Minimizer using Structured Initializers and Basin Hopping.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global time management
TOTAL_TIME_BUDGET = 295.0  # Leave 5s margin for overhead
start_time_global = time.time()

def _morse_energy(x, rho):
    """Compute Morse energy for a cluster."""
    n = x.shape[0]
    if n < 2:
        return 0.0
    x = x.reshape(-1, 3)
    # Compute all pair distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dist, np.inf)
    # Compute energy
    exp_term = np.exp(rho * (1.0 - dist))
    e = np.sum(exp_term * (exp_term - 2.0))
    return e

def _morse_gradient(x, rho):
    """Compute Morse gradient for a cluster."""
    n = x.shape[0]
    if n < 2:
        return np.zeros_like(x)
    x = x.reshape(-1, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    # Avoid division by zero
    mask = dist > 1e-10
    dist[mask] = dist[mask]
    dist[~mask] = 1.0
    
    exp_term = np.exp(rho * (1.0 - dist))
    # Gradient of energy wrt positions
    # dE/dr = 2 * rho * exp(rho*(1-r)) * (exp(rho*(1-r)) - 1)
    grad_r = 2.0 * rho * exp_term * (exp_term - 1.0)
    # Avoid issues with diagonal
    np.fill_diagonal(grad_r, 0.0)
    
    # Chain rule: dE/dx_i = sum_j grad_r_ij * (x_i - x_j) / r_ij
    # diff[i,j] = x_i - x_j
    # dist[i,j] is the distance
    # We need to divide diff by dist
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vec = diff / dist[:, :, np.newaxis]
    
    # Sum over j
    grad = np.sum(grad_r[:, :, np.newaxis] * unit_vec, axis=1)
    return grad

def generate_fcc_shells(n, spacing=1.0):
    """Generate n atoms in FCC shell structure."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Generate points in an FCC lattice, sorted by distance from origin
    max_coord = int(np.ceil(n ** (1/3.0)) * 2) + 5
    coords = []
    
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * spacing
                    coords.append(p)
    
    coords = np.array(coords)
    dists = np.linalg.norm(coords, axis=1)
    order = np.argsort(dists)
    return coords[order[:n]]

def generate_hcp_shells(n, spacing=1.0):
    """Generate n atoms in HCP shell structure."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    max_coord = int(np.ceil(n ** (1/3.0)) * 2) + 5
    coords = []
    a = spacing
    c = a * np.sqrt(8.0/3.0)
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # HCP basis
                for dx, dy, dz in [(0,0,0), (1,0,0), (0.5, np.sqrt(3)/2, 0), 
                                   (0.5, np.sqrt(3)/6, c/2), (0.5, 2*np.sqrt(3)/3, c/2)]:
                    p = np.array([dx, dy, dz]) * a
                    p[0] += i * a
                    p[1] += j * a
                    p[2] += k * c
                    coords.append(p)
    
    coords = np.array(coords)
    dists = np.linalg.norm(coords, axis=1)
    order = np.argsort(dists)
    return coords[order[:n]]

def generate_icosahedral(n, spacing=1.0):
    """Generate n atoms in icosahedral shell structure."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Icosahedral vertices (12)
    phi = (1 + np.sqrt(5)) / 2
    icos_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]) * spacing / np.sqrt(1 + phi**2)
    
    coords = [np.zeros(3)]  # Center
    if n > 1:
        coords.extend(list(icos_verts[:min(12, n-1)]))
    
    # If we need more, add a second shell (scaled icosahedron)
    if n > 13:
        second_shell = icos_verts * 2.0
        remaining = n - len(coords)
        coords.extend(list(second_shell[:remaining]))
    
    return np.array(coords[:n])

def generate_spherical_shells(n, spacing=1.0):
    """Generate n atoms in spherical shell structure."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    coords = [np.zeros(3)]  # Center
    shell_radii = [1.0, 1.8, 2.6, 3.4, 4.2]
    shell_caps = [12, 42, 92, 162, 252]  # Approximate max atoms per shell
    
    for radius, cap in zip(shell_radii, shell_caps):
        if len(coords) >= n:
            break
        # Generate points on sphere using Fibonacci spiral
        num_in_shell = min(cap, n - len(coords))
        for i in range(num_in_shell):
            z = 1 - 2 * (i + 0.5) / num_in_shell
            r = np.sqrt(1 - z**2)
            theta = 2 * np.pi * i / 1.618033988749895
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * radius * spacing
            coords.append(p)
    
    return np.array(coords[:n])

def generate_random_cluster(n, spacing=1.0, seed=None):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
            points.append(p)
        attempts += 1
    
    if len(points) < n:
        # Fallback: just add more random points
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    
    return np.array(points)

def local_minimize(x, rho, max_iter=1000, time_limit=5.0):
    """Local minimization using L-BFGS-B."""
    start = time.time()
    x = x.flatten()
    
    def fun_and_grad(x_flat):
        x_3d = x_flat.reshape(-1, 3)
        e = _morse_energy(x_3d, rho)
        g = _morse_gradient(x_3d, rho).flatten()
        return e, g
    
    try:
        from scipy.optimize import minimize
        result = minimize(fun_and_grad, x, method='L-BFGS-B', 
                         jac=True, options={'maxiter': max_iter, 'maxfun': 5000})
        if time.time() - start > time_limit:
            return x_3d, e  # Return current best
        return result.x.reshape(-1, 3), result.fun
    except Exception:
        return x_3d, _morse_energy(x_3d, rho)

def basin_hopping_step(x, rho, step_size=0.1, rng=None):
    """Perform one basin hopping step."""
    if rng is None:
        rng = np.random.default_rng()
    
    # Perturb
    x_perturbed = x + rng.normal(0, step_size, x.shape)
    
    # Local minimize
    x_new, e_new = local_minimize(x_perturbed, rho, max_iter=500, time_limit=2.0)
    
    return x_new, e_new

def construct(instance):
    """Find the lowest-energy arrangement of n atoms bound by Morse potentials."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    # Time budget for this instance
    instance_start = time.time()
    instance_time_budget = TOTAL_TIME_BUDGET / 6.0
    
    # Generate diverse structural prototypes
    prototypes = []
    prototypes.append(generate_fcc_shells(size, 1.0))
    prototypes.append(generate_hcp_shells(size, 1.0))
    prototypes.append(generate_icosahedral(size, 1.0))
    prototypes.append(generate_spherical_shells(size, 1.0))
    
    # Add random prototypes
    for seed in [42, 123, 456, 789]:
        prototypes.append(generate_random_cluster(size, 1.0, seed))
    
    # Local minimize each prototype
    best_x = None
    best_e = np.inf
    
    for i, proto in enumerate(prototypes):
        if time.time() - instance_start > instance_time_budget * 0.4:
            break
        
        x_min, e_min = local_minimize(proto, rho, max_iter=1000, time_limit=3.0)
        
        if e_min < best_e:
            best_e = e_min
            best_x = x_min
        
        # Basin hopping from this local minimum
        if time.time() - instance_start > instance_time_budget * 0.7:
            break
        
        rng = np.random.default_rng(i)
        for bh_iter in range(10):
            if time.time() - instance_start > instance_time_budget * 0.9:
                break
            
            x_new, e_new = basin_hopping_step(best_x, rho, step_size=0.15, rng=rng)
            
            if e_new < best_e:
                best_e = e_new
                best_x = x_new
            else:
                # Metropolis acceptance
                if np.log(rng.uniform()) < (best_e - e_new) / 0.05:
                    best_x = x_new
    
    # Final local minimization
    if best_x is not None:
        best_x, best_e = local_minimize(best_x, rho, max_iter=2000, time_limit=5.0)
    
    # Use the provided relax function as a final check
    if best_x is not None:
        try:
            e_relaxed, x_relaxed = relax(best_x, potential, rho)
            if e_relaxed < best_e:
                best_e = e_relaxed
                best_x = x_relaxed
        except Exception:
            pass
    
    return best_x if best_x is not None else np.zeros((size, 3))
# EVOLVE-BLOCK-END
