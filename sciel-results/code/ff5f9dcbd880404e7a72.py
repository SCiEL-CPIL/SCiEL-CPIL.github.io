"""Improved program: construct the 55-atom Mackay icosahedron, pre-optimize shell radii,
then apply L-BFGS relaxation with a deadline guard."""
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar
from itertools import combinations

N = 55

# Deadline guard: evaluator kills at 150s wall-clock; use 110s as safety margin
DEADLINE_SECONDS = 110.0

def _lj_energy(pos):
    """Compute total Lennard-Jones energy for N atoms."""
    n = len(pos)
    rij = np.zeros((n, n, 3))
    for i in range(n):
        rij[i, i+1:] = pos[i] - pos[i+1:]
        rij[i+1:, i] = -rij[i, i+1:]
    r2 = np.sum(rij**2, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero on diagonal
    r6 = r2**(-3)
    r12 = r6**2
    energy = 4.0 * np.sum(r12 - r6)
    return energy

def _lj_energy_grad(pos):
    """Compute gradient of Lennard-Jones energy."""
    n = len(pos)
    rij = np.zeros((n, n, 3))
    for i in range(n):
        rij[i, i+1:] = pos[i] - pos[i+1:]
        rij[i+1:, i] = -rij[i, i+1:]
    r2 = np.sum(rij**2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2**(-3)
    r12 = r6**2
    # dE/dr2 = 4*(-12*r2^(-7) + 6*r2^(-4)) for each pair
    # grad_i = sum_j 4*(-12*r2^(-7) + 6*r2^(-4)) * 2 * (pos_i - pos_j)
    coeff = 4.0 * (-12.0 * r2**(-7) + 6.0 * r2**(-4))
    np.fill_diagonal(coeff, 0.0)
    grad = 2.0 * np.sum(coeff[:, :, np.newaxis] * rij, axis=1)
    return grad

def _mackay_icosahedron_55(r_inner, r_outer):
    """
    Construct 55-atom Mackay icosahedron.
    - 1 atom at center
    - 12 atoms at inner shell radius r_inner (vertices of regular icosahedron)
    - 30 atoms at outer shell radius r_outer (edge midpoints of the icosahedron, scaled)
    - 12 atoms at outer shell radius r_outer * (vertex/edge-midpoint ratio)
    
    Actually the standard Mackay icosahedron M(1,2) has:
    - 1 center
    - 12 at radius ~1.054 (inner icosahedron vertices)
    - 30 at radius ~1.755 (outer shell, edge-midpoint directions)
    - 12 at radius ~2.000 (outer shell, vertex directions)
    
    Let me use the standard construction:
    Inner shell: 12 vertices of regular icosahedron at radius r1
    Outer shell: 30 edge-midpoint directions at radius r2, and 12 vertex directions at radius r3
    
    For a regular icosahedron with vertices at distance 1 from center:
    - Edge length = 2/sqrt(5) * phi where phi = (1+sqrt(5))/2... 
    Let me just use the standard coordinates.
    
    Regular icosahedron vertices (unit radius):
    The 12 vertices can be given as:
    (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1) where phi = (1+sqrt(5))/2
    Normalized to unit radius.
    
    Edge midpoints: 30 midpoints of edges.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 vertices of regular icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit radius
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find edges: pairs of vertices at minimum distance
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            dists[i, j] = d
            dists[j, i] = d
    
    # Minimum edge length
    min_dist = np.min(dists[dists > 0])
    edge_mask = dists < min_dist * 1.001
    
    # Get edge midpoints (30 edges)
    edge_midpoints = []
    for i in range(12):
        for j in range(i+1, 12):
            if edge_mask[i, j]:
                mid = (verts_unit[i] + verts_unit[j]) / 2.0
                mid_unit = mid / np.linalg.norm(mid)
                edge_midpoints.append(mid_unit)
    
    edge_midpoints = np.array(edge_midpoints)  # shape (30, 3)
    
    # Build the cluster
    points = [np.zeros(3)]  # center atom
    
    # Inner shell: 12 vertices at radius r_inner
    for v in verts_unit:
        points.append(r_inner * v)
    
    # Outer shell: 30 edge-midpoint directions at radius r_outer
    for v in edge_midpoints:
        points.append(r_outer * v)
    
    # Outer shell: 12 vertex directions at radius r_outer * (vertex_radius_ratio)
    # In Mackay icosahedron, the outer vertex atoms are at a larger radius than edge-midpoint atoms
    # The ratio is such that vertex atoms are at distance ~2.0 while edge-midpoint atoms are at ~1.755
    # Let's parameterize: outer vertex radius = r_outer * ratio, where ratio ~ 2.0/1.755 ≈ 1.1397
    # Actually, let's just use a third parameter or fix the ratio.
    # Standard Mackay: inner radius ~1.054, edge-mid outer ~1.755, vertex outer ~2.0
    # So vertex_outer / edge_mid_outer = 2.0/1.755 ≈ 1.1397
    vertex_ratio = 2.0 / 1.755
    for v in verts_unit:
        points.append(r_outer * vertex_ratio * v)
    
    return np.array(points, dtype=float)

def _optimize_radii():
    """Optimize the two shell radii to minimize LJ energy."""
    def energy_from_radii(params):
        r_inner, r_outer = params
        if r_inner < 0.3 or r_outer < 0.5 or r_outer < r_inner:
            return 1e10
        pos = _mackay_icosahedron_55(r_inner, r_outer)
        return _lj_energy(pos)
    
    # Initial guess based on known Mackay geometry
    r0 = [1.054, 1.755]
    result = minimize(energy_from_radii, r0, method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
    return result.x, result.fun

def construct():
    start_time = time.time()
    
    # Step 1: Optimize shell radii
    try:
        radii, radius_energy = _optimize_radii()
    except Exception:
        radii = [1.054, 1.755]
        radius_energy = _lj_energy(_mackay_icosahedron_55(1.054, 1.755))
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 10:
        # Not enough time for relaxation; return the radius-optimized structure
        pos = _mackay_icosahedron_55(radii[0], radii[1])
        return pos
    
    # Step 2: Apply L-BFGS relaxation
    pos0 = _mackay_icosahedron_55(radii[0], radii[1])
    
    def energy_func(x):
        return _lj_energy(x.reshape(-1, 3))
    
    def grad_func(x):
        return _lj_energy_grad(x.reshape(-1, 3)).flatten()
    
    try:
        result = minimize(
            energy_func, 
            pos0.flatten(), 
            jac=grad_func,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'gtol': 1e-10, 'ftol': 1e-14}
        )
        pos_relaxed = result.x.reshape(-1, 3)
        final_energy = result.fun
    except Exception:
        pos_relaxed = pos0
        final_energy = radius_energy
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 5:
        return pos_relaxed
    
    # Step 3: Multi-start refinement - try a few small perturbations
    best_pos = pos_relaxed
    best_energy = final_energy
    
    np.random.seed(42)
    remaining = DEADLINE_SECONDS - elapsed - 5
    if remaining > 10:
        n_attempts = min(5, int(remaining / 15))
        for attempt in range(n_attempts):
            if time.time() - start_time > DEADLINE_SECONDS - 5:
                break
            # Small perturbation
            noise_scale = 0.05
            pos_perturbed = pos_relaxed + np.random.normal(0, noise_scale, pos_relaxed.shape)
            
            try:
                res = minimize(
                    energy_func,
                    pos_perturbed.flatten(),
                    jac=grad_func,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'gtol': 1e-10, 'ftol': 1e-14}
                )
                e = res.fun
                if e < best_energy:
                    best_energy = e
                    best_pos = res.x.reshape(-1, 3)
            except Exception:
                pass
    
    return best_pos
