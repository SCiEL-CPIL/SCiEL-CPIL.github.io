"""Lennard-Jones cluster global minimum, 55 atoms.
Strategy:
  1. Build a high-quality initial guess: Mackay icosahedron (two shells) with
     slight random perturbation to break symmetry and avoid saddle points.
  2. Use scipy.optimize.minimize (L-BFGS-B) for local relaxation.
  3. Multiple restarts with different random seeds and perturbation magnitudes.
  4. Deadline guard: track wall-clock time and return best result before 150s.
"""
import numpy as np
import time
import math

N = 55
DEADLINE = 145.0  # safety margin before 150s kill

def lj_energy(pos):
    """Compute total Lennard-Jones energy: 4*sum(r^-12 - r^-6) over all pairs."""
    n = pos.shape[0]
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, np.inf)  # avoid division by zero
    r2 = r2[np.triu_indices(n, k=1)]
    r6 = r2 ** (-3)
    r12 = r6 * r6
    return 4.0 * np.sum(r12 - r6)

def lj_grad(pos):
    """Compute gradient of Lennard-Jones energy."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    # dE/dr = 4*(-12*r^-13 + 6*r^-7) = 24*(r^-7 - 2*r^-13)
    # dE/dpos_i = sum_j dE/dr_ij * (pos_i - pos_j)/r_ij
    # Using r2: dE/dpos_i = sum_j 24*(r2^-3 - 2*r2^-6) * (pos_i - pos_j) * r2^-0.5 * r2 / r2
    # Actually: dE/dr = 24*(r^-7 - 2*r^-13)
    # dE/dpos_i_j = dE/dr * (pos_i_j - pos_j_j)/r = dE/dr * diff[i,j]/r
    # = 24*(r^-7 - 2*r^-13) * diff[i,j] / r = 24*(r^-8 - 2*r^-14) * diff[i,j]
    # In terms of r2: r^-8 = r2^-4, r^-14 = r2^-7
    r2_inv = 1.0 / r2
    r2_inv4 = r2_inv ** 4
    r2_inv7 = r2_inv ** 7
    factor = 24.0 * (r2_inv4 - 2.0 * r2_inv7)
    grad = np.sum(diff * factor[:, :, np.newaxis], axis=1)
    return grad

def build_mackay_icosahedron(perturb=0.0):
    """
    Build a Mackay icosahedron cluster of 55 atoms.
    Shell 1: 12 atoms at vertices of regular icosahedron (radius r1)
    Shell 2: 42 atoms (30 at edge midpoints + 12 at face centers) at radius r2
    Center: 1 atom at origin
    Total: 1 + 12 + 42 = 55
    """
    # Golden ratio
    phi = (1.0 + math.sqrt(5.0)) / 2.0

    # Icosahedron vertices (12)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]

    # For LJ, optimal inner shell radius ~1.0, outer ~1.62 (approximate)
    # Use known Mackay geometry: inner shell at r=1, outer at r=sqrt(1+phi^2)/sqrt(1+phi^2)... 
    # Standard Mackay: inner icosahedron radius R1, outer at R2 = R1 * sqrt(1 + phi^2) / 1? 
    # Actually for Mackay icosahedron, the ratio is such that nearest neighbors are ~1.
    # Let's use R1 = 1.0, and place outer shell at the icosahedron face centers and edge midpoints scaled.
    
    # Edge midpoints: for each edge of icosahedron, midpoint direction
    # Face centers: for each triangular face, centroid direction
    
    # Icosahedron edges: 30 edges. We need the adjacency.
    # Let's compute edges by distance between vertices
    vdist = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            vdist[i,j] = np.linalg.norm(verts[i] - verts[j])
    # Find edges (shortest distances)
    min_dist = vdist[np.triu_indices(12, 1)].min()
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(vdist[i,j] - min_dist) < 1e-10:
                edges.append((i,j))
    
    # Edge midpoints (30)
    edge_midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        edge_midpoints.append(mid)
    edge_midpoints = np.array(edge_midpoints)
    
    # Face centers: 20 faces. Find faces (triangles of vertices that form a face)
    # A face is a set of 3 vertices that are all mutually connected by edges
    edge_set = set()
    for i, j in edges:
        edge_set.add((min(i,j), max(i,j)))
    
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            if (min(i,j), max(i,j)) not in edge_set:
                continue
            for k in range(j+1, 12):
                if (min(j,k), max(j,k)) not in edge_set:
                    continue
                if (min(i,k), max(i,k)) not in edge_set:
                    continue
                faces.append((i,j,k))
    
    face_centers = []
    for i, j, k in faces:
        center = (verts[i] + verts[j] + verts[k]) / 3.0
        center = center / np.linalg.norm(center)
        face_centers.append(center)
    face_centers = np.array(face_centers)
    
    # Radii: for Mackay icosahedron with inner shell at R1, outer shell at R2
    # Nearest neighbor distance should be ~1.0 for LJ minimum
    # Inner shell neighbors: distance between adjacent inner vertices = 2*R1*sin(pi/5)
    # For icosahedron, edge length = 2*R1*sin(pi/5) where R1 is circumradius
    # We want edge length ~1.0, so R1 = 1/(2*sin(pi/5))
    R1 = 1.0 / (2.0 * math.sin(math.pi / 5.0))
    
    # Outer shell: edge midpoints are at distance from center = R1 * cos(pi/10) ... 
    # Actually, the midpoint of an edge of a unit-circumradius icosahedron is at distance cos(pi/10) from center? 
    # Let's just scale: outer shell radius R2 such that distance from inner vertex to outer edge midpoint ~1.0
    # Distance from vertex v to midpoint m of adjacent edge: |v - m| where m = (v + v')/2
    # |v - (v+v')/2| = |v - v'|/2 = edge_length/2 = 0.5 (if edge_length=1)
    # But the outer atom is at R2 * m_direction, not at the actual midpoint
    # Let's use R2 = R1 * 1.62 (typical Mackay ratio)
    R2 = R1 * 1.62
    
    atoms = []
    # Center
    atoms.append(np.array([0.0, 0.0, 0.0]))
    # Inner shell
    for v in verts:
        atoms.append(v * R1)
    # Outer shell: edge midpoints
    for m in edge_midpoints:
        atoms.append(m * R2)
    # Outer shell: face centers
    for f in face_centers:
        atoms.append(f * R2)
    
    pos = np.array(atoms[:N], dtype=float)
    
    if perturb > 0:
        pos += np.random.normal(0, perturb, pos.shape)
    
    return pos

def construct():
    t_start = time.time()
    best_pos = None
    best_energy = np.inf
    
    # Try multiple restarts with different perturbations and seeds
    rng = np.random.RandomState(42)
    
    for restart in range(20):
        if time.time() - t_start > DEADLINE - 5:
            break
        
        # Vary perturbation size
        perturb = 0.05 * (1 + restart * 0.1)
        
        # Create initial guess
        pos = build_mackay_icosahedron(perturb=perturb * 0.1)  # small perturb
        # Add some random noise
        pos += rng.normal(0, perturb, pos.shape)
        
        # Check minimum distance
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    min_dist = d
        if min_dist < 0.5:
            continue
        
        # Local optimization
        try:
            from scipy.optimize import minimize
            result = minimize(
                lj_energy,
                pos,
                jac=lj_grad,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'gtol': 1e-8, 'ftol': 1e-12}
            )
            if result.success or result.fun < best_energy:
                if result.fun < best_energy:
                    best_energy = result.fun
                    best_pos = result.x.copy()
        except Exception:
            pass
        
        # Also try with larger perturbation
        if time.time() - t_start > DEADLINE - 5:
            break
        pos2 = build_mackay_icosahedron(perturb=0.0)
        pos2 += rng.normal(0, 0.1 + restart * 0.02, pos2.shape)
        
        min_dist2 = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos2[i] - pos2[j])
                if d < min_dist2:
                    min_dist2 = d
        if min_dist2 < 0.5:
            continue
        
        try:
            from scipy.optimize import minimize
            result2 = minimize(
                lj_energy,
                pos2,
                jac=lj_grad,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'gtol': 1e-8, 'ftol': 1e-12}
            )
            if result2.fun < best_energy:
                best_energy = result2.fun
                best_pos = result2.x.copy()
        except Exception:
            pass
    
    if best_pos is None:
        # Fallback: return the seed program's output
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(points[:N], dtype=float) * 1.12
    
    return best_pos
