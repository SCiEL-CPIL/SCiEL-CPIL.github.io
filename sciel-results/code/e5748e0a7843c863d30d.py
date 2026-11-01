"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START

def _lj_energy_and_grad(x):
    """Compute LJ energy and gradient for flat array x of shape (N,3)."""
    x = x.reshape(N, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** 3
    r12 = r6 ** 2
    # Energy: 4 * sum_{i<j} (r^-12 - r^-6)
    # Using full matrix, each pair counted twice, so divide by 2
    energy = 2.0 * np.sum(1.0 / r12 - 1.0 / r6)
    # Subtract the diagonal contribution (we set r2=1 on diagonal, so term is 4*(1-1)=0, but let's be safe)
    # Actually with r2=1, r6=1, r12=1, term = 1-1=0, so diagonal contributes 0. Good.
    
    # Gradient: dE/dx_i = sum_{j!=i} 4*(-12*r^-13*(x_i-x_j)_k + 6*r^-7*(x_i-x_j)_k)
    # = sum_{j!=i} 4*(x_i-x_j)_k * (6*r^-7 - 12*r^-13)
    # Let f(r2) = 6/r6 - 12/r12 = 6*r^-6 - 12*r^-12
    # grad_i = 4 * sum_{j!=i} f(r2_ij) * (x_i - x_j)
    f = 6.0 / r6 - 12.0 / r12
    np.fill_diagonal(f, 0.0)
    grad = 4.0 * np.sum(f[:, :, np.newaxis] * diff, axis=1)
    return energy, grad.reshape(-1)


def _lj_energy(x):
    """Compute LJ energy only."""
    x = x.reshape(N, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** 3
    r12 = r6 ** 2
    energy = 2.0 * np.sum(1.0 / r12 - 1.0 / r6)
    return energy


def _generate_mackay_icosahedron():
    """Generate the two-shell Mackay icosahedron for 55 atoms.
    
    Shell 1: 12 vertices of icosahedron at radius r1
    Shell 2: 42 points at radius r2 (12 face centers + 30 edge midpoints)
    
    Using standard Mackay icosahedron geometry.
    """
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # For a Mackay icosahedron with 55 atoms:
    # Inner shell: 12 atoms at radius r1
    # Outer shell: 42 atoms at radius r2
    # The ratio r2/r1 is determined by the geometry
    # For the standard Mackay icosahedron, the second shell atoms are at:
    # - 12 face centers (one per face of the icosahedron)
    # - 30 edge midpoints (one per edge)
    
    # Let's use the known optimal radii for the 55-atom Mackay icosahedron
    # From literature, the optimal configuration has:
    # r1 (inner shell) ≈ 1.0 (in LJ units where minimum is at r=2^(1/6)≈1.1225)
    # Actually, let's construct it properly.
    
    # The icosahedron has 20 faces and 30 edges.
    # Face centers: average of 3 vertices of each face
    # Edge midpoints: average of 2 vertices of each edge
    
    # First, find all faces and edges of the icosahedron
    # The icosahedron adjacency can be determined from the vertex coordinates
    # Two vertices are connected if their distance is the edge length
    
    # Edge length of unit icosahedron (vertices on unit sphere)
    # The 12 vertices are at distance sqrt(1+phi^2) from origin before normalization
    # After normalization to unit sphere, edge length = 2*sin(pi/5) = 2*sin(36°)
    # Let's just compute pairwise distances and find edges
    
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            dists[i, j] = np.linalg.norm(verts[i] - verts[j])
            dists[j, i] = dists[i, j]
    
    # Find the minimum non-zero distance (edge length)
    min_dist = np.min(dists[dists > 0])
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_dist) < 1e-10:
                edges.append((i, j))
    
    # Find faces: triples of vertices that form a triangle (all three pairs are edges)
    edge_set = set()
    for i, j in edges:
        edge_set.add((min(i,j), max(i,j)))
    
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                if (min(i,j), max(i,j)) in edge_set and \
                   (min(j,k), max(j,k)) in edge_set and \
                   (min(i,k), max(i,k)) in edge_set:
                    faces.append((i, j, k))
    
    # Should have 20 faces and 30 edges
    # Now construct the two-shell structure
    
    # For Mackay icosahedron, the outer shell radius relative to inner:
    # The face centers are at distance r_face = r1 * (some factor)
    # Actually, let's use the standard construction:
    # Inner shell: 12 atoms at the icosahedron vertices, scaled to radius R1
    # Outer shell: 42 atoms, scaled to radius R2
    
    # The optimal R1 and R2 for minimum energy can be found by optimization
    # For now, use a reasonable starting guess and let relaxation fix it
    
    # Standard Mackay icosahedron: R2/R1 ≈ 1.618 (golden ratio related)
    # Actually, for the 55-atom cluster, the ratio is specific.
    
    # Let's use R1 = 1.0 and R2 = 1.618 as initial guess
    # Then relax to find optimal
    
    R1 = 1.0
    R2 = 1.618  # golden ratio
    
    # Inner shell: 12 vertices
    inner = verts * R1
    
    # Outer shell: 42 points
    outer_points = []
    
    # 12 face centers
    for face in faces:
        fc = (verts[face[0]] + verts[face[1]] + verts[face[2]]) / 3.0
        fc = fc / np.linalg.norm(fc) * R2
        outer_points.append(fc)
    
    # 30 edge midpoints
    for edge in edges:
        em = (verts[edge[0]] + verts[edge[1]]) / 2.0
        em = em / np.linalg.norm(em) * R2
        outer_points.append(em)
    
    outer = np.array(outer_points)
    
    # Combine
    all_points = np.vstack([inner, outer])
    
    # Center the structure
    all_points -= np.mean(all_points, axis=0)
    
    return all_points


def construct():
    """Construct initial positions for 55-atom LJ cluster optimization."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Generate Mackay icosahedron
    mackay = _generate_mackay_icosahedron()
    
    # Compute initial energy
    initial_energy = _lj_energy(mackay.reshape(-1))
    
    # Try to relax the Mackay icosahedron directly first
    best_positions = mackay.copy()
    best_energy = initial_energy
    
    try:
        result = minimize(
            _lj_energy_and_grad,
            mackay.reshape(-1),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-15}
        )
        if result.success and result.fun < best_energy:
            best_positions = result.x.reshape(N, 3)
            best_energy = result.fun
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 30:
        return best_positions
    
    # Multi-start: perturb Mackay icosahedron with Gaussian noise and relax
    rng = np.random.default_rng(42)
    
    for trial in range(20):
        if time.time() > deadline - 5:
            break
        
        # Add Gaussian noise
        noise_scale = 0.1 + 0.05 * trial
        perturbed = mackay + rng.normal(0, noise_scale, size=mackay.shape)
        
        try:
            result = minimize(
                _lj_energy_and_grad,
                perturbed.reshape(-1),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'gtol': 1e-8, 'ftol': 1e-14}
            )
            if result.success and result.fun < best_energy:
                best_positions = result.x.reshape(N, 3)
                best_energy = result.fun
        except Exception:
            continue
    
    # Final high-precision polish
    if time.time() < deadline - 10:
        try:
            result = minimize(
                _lj_energy_and_grad,
                best_positions.reshape(-1),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'gtol': 1e-12, 'ftol': 1e-16}
            )
            if result.success and result.fun < best_energy:
                best_positions = result.x.reshape(N, 3)
                best_energy = result.fun
        except Exception:
            pass
    
    return best_positions

# EVOLVE-BLOCK-END
