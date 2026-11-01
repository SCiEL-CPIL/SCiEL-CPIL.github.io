"""Lennard-Jones cluster global minimum, 55 atoms - Mackay icosahedron based search."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

def _lj_energy(positions):
    """Compute total Lennard-Jones energy."""
    pos = np.asarray(positions, dtype=float)
    n = len(pos)
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    energy = 4.0 * np.sum(r12 - r6)
    # Subtract self-interaction (we set diagonal to 1, so r6=1, r12=1, contribution=0)
    return energy

def _lj_energy_and_grad(positions):
    """Compute total Lennard-Jones energy and gradient."""
    pos = np.asarray(positions, dtype=float)
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    energy = 4.0 * np.sum(r12 - r6)
    
    # Gradient: dE/dr_i = sum_j 4*(-12*r^-14 + 6*r^-8) * (r_i - r_j)
    # = sum_j 4*(-12/r^14 + 6/r^8) * (r_i - r_j)
    # In terms of r2: r^-14 = r2^-7, r^-8 = r2^-4
    # grad = 4 * sum_j (-12*r2^-7 + 6*r2^-4) * (r_i - r_j)
    # Note: (r_i - r_j) = diff[i,j]
    
    # Coefficient for each pair
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return energy, grad

def _valid(positions):
    """Check if all atoms are at least 0.5 apart."""
    pos = np.asarray(positions, dtype=float)
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    return min_r2 >= 0.25  # 0.5^2

def _relax(positions, maxiter=500, tol=1e-12):
    """Local relaxation using L-BFGS with energy and gradient."""
    pos = np.asarray(positions, dtype=float).copy()
    
    def fun_and_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.reshape(-1)
    
    res = minimize(fun_and_grad, pos.flatten(), method='L-BFGS-B', 
                   jac=True, options={'maxiter': maxiter, 'gtol': tol, 'ftol': 1e-15})
    return res.x.reshape(-1, 3), res.fun

def _mckay_icosahedron():
    """Generate Mackay icosahedron with 55 atoms (1+12+42)."""
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices (12)
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize icosahedron vertices to unit length
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # Center atom
    center = np.array([[0.0, 0.0, 0.0]])
    
    # First shell: 12 atoms at icosahedron vertices, scaled
    # The optimal spacing for LJ is r=2^(1/6) ≈ 1.1225
    # For Mackay icosahedron, first shell radius is typically around 1.0-1.2
    shell1_radius = 1.0
    shell1 = ico_verts * shell1_radius
    
    # Second shell: 42 atoms
    # These are at positions corresponding to edges, face centers, and other positions
    # In a Mackay icosahedron, the second shell consists of:
    # - 30 atoms along edges (midpoints of icosahedron edges, scaled)
    # - 12 atoms at face centers (scaled)
    # Total: 30 + 12 = 42
    
    # Generate icosahedron edges
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            dist = np.linalg.norm(ico_verts[i] - ico_verts[j])
            # Icosahedron edge length in normalized form
            if dist < 1.5:  # threshold for adjacent vertices
                edges.append((i, j))
    
    # Edge midpoints (30 edges)
    edge_midpoints = []
    for i, j in edges:
        mid = (ico_verts[i] + ico_verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        edge_midpoints.append(mid)
    edge_midpoints = np.array(edge_midpoints)
    
    # Face centers (12 faces)
    # Find triangular faces of icosahedron
    faces = []
    for i in range(12):
        for j in range(12):
            if i == j:
                continue
            for k in range(12):
                if i == k or j == k:
                    continue
                if i < j < k:
                    # Check if i,j,k form a face (all three are mutually adjacent)
                    d_ij = np.linalg.norm(ico_verts[i] - ico_verts[j])
                    d_jk = np.linalg.norm(ico_verts[j] - ico_verts[k])
                    d_ik = np.linalg.norm(ico_verts[i] - ico_verts[k])
                    if d_ij < 1.5 and d_jk < 1.5 and d_ik < 1.5:
                        faces.append((i, j, k))
    
    # Remove duplicate faces
    unique_faces = []
    face_set = set()
    for face in faces:
        # Sort vertices to create canonical form
        sorted_face = tuple(sorted(face))
        if sorted_face not in face_set:
            face_set.add(sorted_face)
            unique_faces.append(face)
    
    # Compute face centers
    face_centers = []
    for i, j, k in unique_faces[:12]:  # Take first 12
        center = (ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3.0
        center = center / np.linalg.norm(center)
        face_centers.append(center)
    face_centers = np.array(face_centers)
    
    # Second shell radius
    shell2_radius = 1.8
    
    # Combine second shell atoms
    shell2 = np.vstack([edge_midpoints * shell2_radius, face_centers * shell2_radius])
    
    # Combine all atoms
    all_atoms = np.vstack([center, shell1, shell2])
    
    return all_atoms

def construct():
    """Construct 55-atom LJ cluster configuration."""
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Generate Mackay icosahedron
    pos = _mckay_icosahedron()
    
    # Ensure exactly 55 atoms
    if len(pos) < N:
        # Add more atoms if needed
        extra = np.random.randn(N - len(pos), 3) * 0.1
        pos = np.vstack([pos, extra])
    else:
        pos = pos[:N]
    
    best_pos = pos.copy()
    best_energy = _lj_energy(pos)
    
    # Local relaxation
    try:
        relaxed_pos, relaxed_energy = _relax(pos, maxiter=1000, tol=1e-14)
        if _valid(relaxed_pos):
            if relaxed_energy < best_energy:
                best_pos = relaxed_pos
                best_energy = relaxed_energy
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_pos
    
    # Try multiple relaxation runs with different initial perturbations
    rng = np.random.RandomState(42)
    
    for trial in range(20):
        if time.time() - start_time > deadline - 10:
            break
        
        # Create perturbed version
        perturbation = rng.randn(N, 3) * 0.05
        trial_pos = best_pos + perturbation
        
        try:
            relaxed_pos, relaxed_energy = _relax(trial_pos, maxiter=500, tol=1e-12)
            if _valid(relaxed_pos) and relaxed_energy < best_energy:
                best_pos = relaxed_pos
                best_energy = relaxed_energy
        except Exception:
            continue
    
    # Final polish with tighter tolerance
    try:
        if time.time() - start_time < deadline - 5:
            polished_pos, polished_energy = _relax(best_pos, maxiter=2000, tol=1e-15)
            if _valid(polished_pos) and polished_energy < best_energy:
                best_pos = polished_pos
                best_energy = polished_energy
    except Exception:
        pass
    
    return best_pos
