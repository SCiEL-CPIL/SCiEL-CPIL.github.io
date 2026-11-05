"""
Dzugutov cluster global minima search using continuous annealing of softness parameters.
Strategy: Anneal c and d from large values (smooth landscape) to target values,
using gradient descent with momentum. This allows a single trajectory to flow
into the global polytetrahedral basin.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov potential parameters
A_P = 5.82
B_P = 1.28
a_P = 1.87
b_P = 1.94
c_P = 1.10
d_P = 0.27
m_P = 16.0


def _dzugutov_energy(pos, c_val, d_val):
    """Compute Dzugutov energy with annealed c and d parameters."""
    n = len(pos)
    if n < 2:
        return 0.0
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    dist = dist[mask]
    
    energy = 0.0
    for r in dist:
        if r < a_P:
            energy += A_P * (r ** (-m_P) - B_P) * np.exp(c_val / (r - a_P))
        elif r < b_P:
            energy += B_P * np.exp(d_val / (r - b_P))
    return energy


def _dzugutov_energy_grad(pos, c_val, d_val):
    """Compute Dzugutov energy and gradient with annealed parameters."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    
    energy = 0.0
    grad = np.zeros_like(pos)
    
    for i in range(n):
        for j in range(i + 1, n):
            rij = pos[i] - pos[j]
            r = np.sqrt(np.sum(rij ** 2))
            if r < 1e-10:
                continue
            
            if r < a_P:
                # V(r) = A * (r^-m - B) * exp(c/(r-a))
                # dV/dr = A * [-m*r^(-m-1) * exp(c/(r-a)) + (r^-m - B) * exp(c/(r-a)) * (-c)/(r-a)^2]
                exp_term = np.exp(c_val / (r - a_P))
                dV_dr = A_P * (-m_P * r ** (-m_P - 1) * exp_term +
                              (r ** (-m_P) - B_P) * exp_term * (-c_val) / (r - a_P) ** 2)
                energy += A_P * (r ** (-m_P) - B_P) * exp_term
            elif r < b_P:
                # V(r) = B * exp(d/(r-b))
                # dV/dr = B * exp(d/(r-b)) * (-d)/(r-b)^2
                exp_term = np.exp(d_val / (r - b_P))
                dV_dr = B_P * exp_term * (-d_val) / (r - b_P) ** 2
                energy += B_P * exp_term
            else:
                dV_dr = 0.0
            
            # Gradient contribution: dE/dri = dV/dr * (ri - rj)/r
            factor = dV_dr / r
            grad[i] += factor * rij
            grad[j] -= factor * rij
    
    return energy, grad


def _generate_icosahedral_seed(n, rng):
    """Generate an initial structure based on icosahedral/polytetrahedral geometry."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
        [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
        [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0]
    ], dtype=float)
    vertices /= np.sqrt(1 + phi ** 2)  # Normalize to unit sphere
    
    # Scale to appropriate radius
    radius = 1.1 * n ** (1.0 / 3.0) * 0.6
    
    points = []
    # Start with icosahedral core
    for v in vertices:
        if len(points) >= n:
            break
        points.append(v * radius * 0.5)
    
    # Add more points on icosahedral shells
    shell_radii = [radius * 0.3, radius * 0.6, radius * 0.9]
    for shell_r in shell_radii:
        while len(points) < n:
            # Random point on sphere
            theta = rng.uniform(0, 2 * np.pi)
            z = rng.uniform(-1, 1)
            r_xy = np.sqrt(1 - z ** 2)
            p = np.array([r_xy * np.cos(theta), r_xy * np.sin(theta), z]) * shell_r
            # Check minimum distance
            if all(np.linalg.norm(p - q) >= 0.8 for q in points):
                points.append(p)
    
    # Fill remaining with random points
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 for q in points):
            points.append(p)
    
    return np.array(points[:n])


def _anneal_and_refine(pos, n, deadline):
    """Anneal c and d parameters, then refine with gradient descent."""
    # Start with large c, d (smooth landscape) and anneal down
    c_start = 5.0
    d_start = 2.0
    n_anneal_steps = 50
    
    current_pos = pos.copy()
    
    for step in range(n_anneal_steps):
        if time.time() > deadline - 30:
            break
        
        # Linear annealing schedule
        t = step / max(n_anneal_steps - 1, 1)
        c_val = c_start + (c_P - c_start) * t
        d_val = d_start + (d_P - d_start) * t
        
        # Gradient descent with momentum
        lr = 0.01 * (1 - t * 0.5)  # Decrease learning rate
        momentum = 0.9
        vel = np.zeros_like(current_pos)
        
        n_inner = 20
        for _ in range(n_inner):
            if time.time() > deadline - 30:
                break
            
            energy, grad = _dzugutov_energy_grad(current_pos, c_val, d_val)
            vel = momentum * vel - lr * grad
            current_pos = current_pos + vel
            
            # Ensure minimum distance constraint
            for i in range(n):
                for j in range(i + 1, n):
                    rij = current_pos[i] - current_pos[j]
                    r = np.linalg.norm(rij)
                    if r < 0.5:
                        # Push apart
                        if r < 1e-10:
                            rij = np.random.randn(3) * 0.01
                            r = np.linalg.norm(rij)
                        push = (0.5 - r) * 0.5
                        current_pos[i] += push * rij / r
                        current_pos[j] -= push * rij / r
    
    # Final refinement with exact potential
    energy, final_pos = relax(current_pos, 'dzugutov')
    return energy, final_pos


def construct(instance):
    """Find low-energy structure for Dzugutov cluster."""
    n = instance["n"]
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s
    
    rng = np.random.default_rng(n)
    
    best_energy = np.inf
    best_pos = None
    
    # Generate multiple diverse seeds
    n_seeds = 6
    for seed_idx in range(n_seeds):
        if time.time() > deadline - 60:
            break
        
        # Different initial structures
        if seed_idx == 0:
            pos = _generate_icosahedral_seed(n, rng)
        else:
            # Random compact cluster with icosahedral bias
            radius = 1.1 * n ** (1.0 / 3.0) * 0.7
            points = []
            # Icosahedral vertices as guide
            phi = (1 + np.sqrt(5)) / 2
            ico_verts = np.array([
                [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
                [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
                [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
                [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0]
            ], dtype=float)
            ico_verts /= np.sqrt(1 + phi ** 2)
            
            while len(points) < n:
                if len(ico_verts) > 0:
                    idx = rng.integers(len(ico_verts))
                    base = ico_verts[idx] * radius * 0.4
                    ico_verts = np.delete(ico_verts, idx, axis=0)
                else:
                    base = rng.uniform(-radius, radius, 3)
                
                p = base + rng.normal(0, 0.3, 3)
                if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 for q in points):
                    points.append(p)
            pos = np.array(points[:n])
        
        # Anneal and refine
        try:
            energy, final_pos = _anneal_and_refine(pos, n, deadline)
            if energy < best_energy:
                best_energy = energy
                best_pos = final_pos
        except Exception:
            pass
    
    # If no valid result, fallback to simple relax
    if best_pos is None:
        radius = 1.1 * n ** (1.0 / 3.0) * 0.7
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 for q in points):
                points.append(p)
        _, best_pos = relax(np.array(points), 'dzugutov')
    
    return best_pos
# EVOLVE-BLOCK-END
