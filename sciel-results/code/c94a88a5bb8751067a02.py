"""Dzugutov cluster global minima search using icosahedral seeds and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

# Dzugutov potential parameters
A0 = 5.82
B0 = 1.28
a0 = 1.87
b0 = 1.94
c0 = 1.10
d0 = 0.27
m0 = 16.0
MIN_DIST = 0.5

def _energy_gradient(pos):
    """Compute Dzugutov energy and gradient. Returns (energy, gradient)."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    # Avoid division by zero
    dist[np.diag_indices_from(dist)] = 1e10
    
    # Pairwise energy
    E_pair = np.zeros((n, n))
    F_pair = np.zeros((n, n, 3))
    
    mask_a = dist < a0
    mask_b = dist < b0
    mask_ab = mask_a | mask_b
    
    # For r < a: V = A(r^-m - B) * exp(c/(r-a))
    # For a <= r < b: V = B * exp(d/(r-b))
    # For r >= b: V = 0
    
    r_safe = np.where(mask_ab, dist, 1.0)  # avoid issues
    
    # Region 1: r < a
    r1 = np.where(mask_a, dist, 1.0)
    rm1 = r1 ** (-m0)
    exp1 = np.exp(c0 / (r1 - a0))
    V1 = A0 * (rm1 - B0) * exp1
    # dV1/dr = A * [(-m*r^(-m-1) * exp1) + (r^-m - B) * exp1 * (-c/(r-a)^2)]
    dV1dr = A0 * (-m0 * r1**(-m0-1) * exp1 + (rm1 - B0) * exp1 * (-c0 / (r1 - a0)**2))
    
    # Region 2: a <= r < b
    r2 = np.where(mask_b & ~mask_a, dist, 1.0)
    exp2 = np.exp(d0 / (r2 - b0))
    V2 = B0 * exp2
    dV2dr = B0 * exp2 * (-d0 / (r2 - b0)**2)
    
    E_pair = np.where(mask_a, V1, 0.0) + np.where(mask_b & ~mask_a, V2, 0.0)
    dVdr = np.where(mask_a, dV1dr, 0.0) + np.where(mask_b & ~mask_a, dV2dr, 0.0)
    
    # Total energy (each pair counted once)
    total_E = 0.5 * np.sum(E_pair)
    
    # Gradient: dE/dpos_i = sum_{j != i} dV/dr_ij * (pos_i - pos_j) / r_ij
    # Force on i from j: -dV/dr_ij * (pos_i - pos_j)/r_ij ... let me be careful
    # E = sum_{i<j} V(r_ij), r_ij = |pos_i - pos_j|
    # dE/dpos_i = sum_{j != i} V'(r_ij) * (pos_i - pos_j) / r_ij
    # where V'(r) = dV/dr
    
    # For pair (i,j): contribution to grad_i = V'(r_ij) * (pos_i - pos_j) / r_ij
    # contribution to grad_j = V'(r_ij) * (pos_j - pos_i) / r_ij = -contribution to grad_i
    
    unit = diff / dist[:, :, np.newaxis]  # (pos_i - pos_j) / r_ij
    # grad_i += V'(r_ij) * unit_ij for each j
    grad = np.sum(dVdr[:, :, np.newaxis] * unit, axis=1)
    
    # Check minimum distance constraint
    min_d = np.min(dist[~np.eye(n, dtype=bool)])
    if min_d < MIN_DIST:
        # Add penalty to push apart
        penalty = 1e6 * (MIN_DIST - min_d)
        # Gradient penalty: push atoms apart
        for i in range(n):
            for j in range(i+1, n):
                if dist[i,j] < MIN_DIST:
                    direction = (pos[i] - pos[j]) / max(dist[i,j], 1e-10)
                    grad[i] += 1e6 * direction
                    grad[j] -= 1e6 * direction
    
    return total_E, grad


def _relax(pos, max_iter=500, tol=1e-8):
    """Relax a cluster using L-BFGS-B."""
    def obj_and_grad(x):
        pos2 = x.reshape(-1, 3)
        e, g = _energy_gradient(pos2)
        return e, g.flatten()
    
    result = minimize(obj_and_grad, pos.flatten(), jac=True, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'gtol': tol, 'maxls': 50})
    return result.fun, result.x.reshape(-1, 3)


def _make_icosahedron(center, radius, n_shell):
    """Generate icosahedral shell vertices."""
    # Golden ratio
    phi = (1 + math.sqrt(5)) / 2.0
    # Icosahedron vertices (12)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0]) * radius
    return verts + center


def _generate_icosahedral_cluster(n, seed=0):
    """Generate an icosahedral cluster with n atoms."""
    rng = np.random.default_rng(seed)
    
    # Core: 1 atom at center
    # First shell: 12 atoms (icosahedron)
    # Second shell: 30 atoms (cuboctahedron-like)
    # Third shell: 56 atoms
    
    # Build up shells
    atoms = [np.zeros(3)]  # center
    
    # Shell 1: 12 atoms
    r1 = 1.2
    shell1 = _make_icosahedron(np.zeros(3), r1, 1)
    atoms.extend(shell1.tolist())
    
    # Shell 2: 30 atoms - use positions from icosahedral symmetry
    # The 30-atom shell can be generated from icosahedron edge midpoints and face centers
    # More practically: use a scaled icosahedron with additional vertices
    # For a 30-atom shell, we can use the vertices of a cuboctahedron scaled
    # or generate from icosahedral symmetry operations
    
    # Let me use a simpler approach: place atoms on concentric shells with icosahedral symmetry
    # Shell 2 has 30 positions. These correspond to the 30 edges of the icosahedron (midpoints)
    # Actually, the 30-atom shell in icosahedral clusters is at the vertices of a cuboctahedron
    # Let me just use random placement with minimum distance constraint for outer shells
    
    current_n = len(atoms)
    
    if n <= current_n:
        return np.array(atoms[:n])
    
    # For remaining atoms, place on outer shells with icosahedral symmetry
    # Use icosahedral symmetry group to generate positions
    # 60 rotation operations of icosahedral group
    
    # Generate more positions by adding shells
    r2 = 2.0
    # 30-atom shell: vertices of a cuboctahedron
    # Cuboctahedron has 12 vertices, not 30. Let me think differently.
    
    # Actually for icosahedral clusters:
    # n=13: 1+12
    # n=55: 1+12+42 (or 1+12+30+12)
    # n=147: 1+12+42+92
    
    # Let me use a different approach: start with icosahedral core and add atoms
    # on a grid constrained by icosahedral symmetry
    
    # Simpler: use the icosahedral shell structure
    # For n up to 60, we need shells of 1, 12, and then up to 47 more
    
    # Let me generate positions using icosahedral symmetry
    # The icosahedral group has 60 elements. We can pick an orbit of size 60/6=10
    # (stabilizer of a generic point in I_h is trivial, so orbit size 60)
    # Or orbits of size 30 (points on 2-fold axes), 20 (3-fold), 12 (5-fold)
    
    # For a 30-atom shell: place 30 atoms at the midpoints of icosahedron edges
    # Icosahedron has 30 edges
    
    # Edge midpoints of icosahedron
    # The 12 vertices are known. Edges connect vertices at the minimum distance.
    # Let me compute edges
    v = verts  # 12 vertices of icosahedron at radius r1
    v = v / np.linalg.norm(v[0])  # unit icosahedron
    
    # Find edges (minimum distance pairs)
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            dists[i,j] = np.linalg.norm(v[i] - v[j])
            dists[j,i] = dists[i,j]
    
    edge_dist = np.min(dists[dists > 0])
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i,j] - edge_dist) < 1e-10:
                edges.append((i, j))
    
    # 30 edge midpoints
    edge_mids = []
    for i, j in edges:
        mid = (v[i] + v[j]) / 2.0
        edge_mids.append(mid)
    edge_mids = np.array(edge_mids)
    edge_mids = edge_mids / np.linalg.norm(edge_mids[0])  # normalize to unit sphere
    
    shell2 = edge_mids * r2
    atoms.extend(shell2.tolist())
    
    current_n = len(atoms)
    
    if n <= current_n:
        return np.array(atoms[:n])
    
    # Need more atoms: add a third shell
    # 56-atom shell or partial
    # For the 56-atom shell, use face centers of the icosahedron (20 faces) 
    # plus additional positions
    
    # Face centers of icosahedron
    # Each face is a triangle of 3 vertices
    # Find all faces (triples of vertices that form a face)
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i,j] - edge_dist) < 1e-10:
                for k in range(j+1, 12):
                    if abs(dists[i,k] - edge_dist) < 1e-10 and abs(dists[j,k] - edge_dist) < 1e-10:
                        faces.append((i, j, k))
    
    face_centers = []
    for i, j, k in faces:
        fc = (v[i] + v[j] + v[k]) / 3.0
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    face_centers = face_centers / np.linalg.norm(face_centers[0])
    
    r3 = 2.6
    shell3 = face_centers * r3
    atoms.extend(shell3.tolist())
    
    current_n = len(atoms)
    
    if n <= current_n:
        return np.array(atoms[:n])
    
    # Add more atoms randomly on outer shell
    # 1+12+30+20 = 63, so for n up to 60 we should be fine
    # But if we need more, add random atoms
    while len(atoms) < n:
        angle1 = rng.uniform(0, 2*math.pi)
        angle2 = rng.uniform(0, math.pi)
        r = r3 + rng.uniform(0, 0.5)
        p = np.array([
            r * math.sin(angle2) * math.cos(angle1),
            r * math.sin(angle2) * math.sin(angle1),
            r * math.cos(angle2)
        ])
        # Check minimum distance
        if all(np.linalg.norm(p - np.array(a)) >= MIN_DIST for a in atoms):
            atoms.append(p)
        else:
            # Try a few more times
            for _ in range(10):
                angle1 = rng.uniform(0, 2*math.pi)
                angle2 = rng.uniform(0, math.pi)
                r = r3 + rng.uniform(0, 0.5)
                p = np.array([
                    r * math.sin(angle2) * math.cos(angle1),
                    r * math.sin(angle2) * math.sin(angle1),
                    r * math.cos(angle2)
                ])
                if all(np.linalg.norm(p - np.array(a)) >= MIN_DIST for a in atoms):
                    atoms.append(p)
                    break
    
    return np.array(atoms[:n])


def _generate_random_cluster(n, seed=0):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = 1.2 * n ** (1.0/3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.9 for q in points):
            points.append(p)
        attempts += 1
    
    if len(points) < n:
        # Fallback: add more points with smaller minimum distance
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
    
    return np.array(points[:n])


def _basin_hop(pos, n_perturb=5, perturb_scale=0.1, n_inner=50, seed=0):
    """Simple basin hopping."""
    rng = np.random.default_rng(seed)
    best_energy, best_pos = _relax(pos, max_iter=n_inner)
    
    for i in range(n_perturb):
        # Perturb
        mask = rng.random(len(pos)) < 0.3
        perturbation = rng.normal(0, perturb_scale, size=(len(pos), 3))
        perturbed = pos.copy()
        perturbed[mask] += perturbation[mask]
        
        # Relax
        e, p = _relax(perturbed, max_iter=n_inner)
        if e < best_energy:
            best_energy, best_pos = e, p
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy Dzugutov cluster."""
    n = instance["n"]
    start_time = time.time()
    deadline = start_time + 290  # Leave 40s margin
    
    best_energy = float('inf')
    best_pos = None
    
    # Strategy 1: Icosahedral seeds with different seeds
    for seed in range(10):
        if time.time() > deadline:
            break
        pos = _generate_icosahedral_cluster(n, seed=seed)
        if len(pos) != n:
            continue
        e, relaxed = _relax(pos, max_iter=300)
        if e < best_energy:
            best_energy, best_pos = e, relaxed
    
    # Strategy 2: Basin hopping from icosahedral seeds
    for seed in range(5):
        if time.time() > deadline:
            break
        pos = _generate_icosahedral_cluster(n, seed=seed)
        if len(pos) != n:
            continue
        e, p = _basin_hop(pos, n_perturb=3, perturb_scale=0.15, n_inner=100, seed=seed)
        if e < best_energy:
            best_energy, best_pos = e, p
    
    # Strategy 3: Random starts
    for seed in range(15):
        if time.time() > deadline:
            break
        pos = _generate_random_cluster(n, seed=seed)
        if len(pos) != n:
            continue
        e, relaxed = _relax(pos, max_iter=200)
        if e < best_energy:
            best_energy, best_pos = e, relaxed
    
    # Strategy 4: Basin hopping from random starts
    for seed in range(5):
        if time.time() > deadline:
            break
        pos = _generate_random_cluster(n, seed=seed)
        if len(pos) != n:
            continue
        e, p = _basin_hop(pos, n_perturb=3, perturb_scale=0.2, n_inner=80, seed=seed)
        if e < best_energy:
            best_energy, best_pos = e, p
    
    # Final: if we still have time, do more basin hopping from the best
    if time.time() < deadline - 30:
        e, p = _basin_hop(best_pos, n_perturb=5, perturb_scale=0.1, n_inner=100, seed=99)
        if e < best_energy:
            best_energy, best_pos = e, p
    
    # Verify minimum distance constraint
    if best_pos is not None:
        for i in range(len(best_pos)):
            for j in range(i+1, len(best_pos)):
                d = np.linalg.norm(best_pos[i] - best_pos[j])
                if d < MIN_DIST:
                    # Push apart
                    direction = (best_pos[i] - best_pos[j]) / max(d, 1e-10)
                    shift = (MIN_DIST - d) / 2.0
                    best_pos[i] += direction * shift
                    best_pos[j] -= direction * shift
    
    return best_pos
# EVOLVE-BLOCK-END
