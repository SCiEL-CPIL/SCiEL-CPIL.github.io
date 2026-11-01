"""Lennard-Jones 55-atom cluster: Mackay icosahedron + L-BFGS relaxation."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 55

# EVOLVE-BLOCK-START

def _lj_energy(pos):
    """Compute total LJ energy for N atoms. pos shape (N, 3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    return e

def _lj_energy_grad(pos):
    """Compute LJ energy and gradient. pos shape (N, 3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    # grad_i = sum_{j != i} 4 * (12*r^-12 - 6*r^-6) * (pos_i - pos_j) / r
    # = sum_j 4 * (12*r12 - 6*r6) * (pos_i - pos_j) / r
    # Using r2: r = sqrt(r2), r12 = r2^-6, r6 = r2^-3
    # force on i from j: 4*(12*r2^-6 - 6*r2^-3) * (pos_i - pos_j) / r
    # = 4*(12*r2^-6 - 6*r2^-3) * (pos_i - pos_j) / sqrt(r2)
    coeff = 4.0 * (12.0 * r12 - 6.0 * r6) / np.sqrt(r2)
    np.fill_diagonal(coeff, 0.0)
    grad = coeff @ diff  # shape (n, 3)
    return e, grad

def _make_mackay_icosahedron():
    """
    Construct a two-shell Mackay icosahedron with 55 atoms.
    Shell 1: 12 vertices of an icosahedron at distance d1 from center.
    Shell 2: 42 vertices at distance d2 from center.
    
    For LJ, the optimal radii are approximately:
    d1 = 1.122 (first shell, ~2^0.5 * 1.122? No, let me think more carefully)
    
    Actually, for a Mackay icosahedron with shells at radii r1 and r2:
    - 12 atoms on shell 1 (icosahedron vertices)
    - 42 atoms on shell 2 (triangular numbers on icosahedron faces)
    
    The standard approach: use icosahedron geometry.
    Icosahedron vertices can be defined using the golden ratio phi = (1+sqrt(5))/2.
    Vertices: (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1)
    These have distance sqrt(1+phi^2) from center.
    
    For the two-shell Mackay icosahedron:
    Shell 1: 12 icosahedron vertices at radius R1
    Shell 2: 42 vertices - these are at the centers of edges and faces of the icosahedron,
             but for a Mackay icosahedron, shell 2 consists of:
             - 30 edge midpoints (but scaled)
             - 12 face centers (but scaled)
             Wait, 30 + 12 = 42. Yes!
    
    So shell 2 has 30 atoms at edge-midpoint directions and 12 at face-center directions,
    all at radius R2.
    
    For LJ optimization, R1 ≈ 1.12 and R2 ≈ 2.0 or so. Let me use reasonable starting values.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # Icosahedron vertices (12)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit vectors
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Shell 1: 12 vertices at radius R1
    R1 = 1.12
    shell1 = verts_unit * R1
    
    # Find all edges of the icosahedron (pairs of vertices that are closest neighbors)
    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform
    dists = squareform(pdist(verts_unit))
    np.fill_diagonal(dists, np.inf)
    
    # The icosahedron has 30 edges. Find the minimum distance
    min_dist = np.min(dists)
    tol = 1e-6
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_dist) < tol:
                edges.append((i, j))
    
    # Edge midpoints (30)
    edge_midpoints = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        mid_norm = np.linalg.norm(mid)
        if mid_norm > 1e-10:
            edge_midpoints.append(mid / mid_norm)
    edge_midpoints = np.array(edge_midpoints)
    
    # Face centers (12)
    # An icosahedron has 20 faces. For a Mackay icosahedron, we need 12 face centers?
    # Wait, I need to reconsider. A two-shell Mackay icosahedron with 55 atoms:
    # 1 + 12 + 42 = 55? No, that's 55 without center.
    # Actually: 12 (shell 1) + 42 (shell 2) = 54. Plus 1 center = 55.
    # OR: 12 + 42 = 54... hmm.
    
    # Let me reconsider. The Mackay icosahedron with 55 atoms:
    # - 1 atom at center
    # - 12 atoms at icosahedron vertices (shell 1)
    # - 42 atoms on shell 2
    # Total: 1 + 12 + 42 = 55. Yes!
    
    # For shell 2 with 42 atoms: these are arranged on the icosahedron's faces.
    # Each face of the icosahedron is a triangle. If we place atoms at the 
    # "second ring" positions on each face... 
    # Actually, for a Mackay icosahedron, shell n has n^2 atoms per face? No.
    # The number of atoms in shell k of a Mackay icosahedron is 10*k^2 + 2.
    # Shell 1: 10*1 + 2 = 12. Shell 2: 10*4 + 2 = 42. Yes!
    
    # So shell 2 has 42 atoms. These are located at positions that correspond to
    # a subdivided icosahedron. The 42 positions can be generated by taking
    # the icosahedron and placing atoms at the "midpoints" of a finer grid.
    
    # A simpler approach: the 42 shell-2 positions are:
    # - 30 at edge midpoints (as computed above)
    # - 12 at... wait, 30 + 12 = 42. But an icosahedron has 20 faces, not 12.
    
    # Let me look at this differently. The 42 positions on shell 2 of a Mackay 
    # icosahedron correspond to the vertices of a truncated icosahedron? No.
    
    # Actually, I think the correct decomposition is:
    # The icosahedron has 12 vertices, 30 edges, 20 faces.
    # For the Mackay icosahedron shell 2:
    # - 12 positions above face centers? No, 20 faces.
    # 
    # Let me just use a known construction. The 42 points are:
    # For each of the 20 faces, place 2 atoms? 20*2 = 40, not 42.
    # For each of the 30 edges, 1 atom (midpoint) = 30
    # For each of the 12 vertices, 1 atom (slightly outside) = 12
    # Total: 30 + 12 = 42. Yes!
    
    # So shell 2 consists of:
    # - 30 atoms at edge midpoint directions
    # - 12 atoms at vertex directions (same as shell 1 directions but at larger radius)
    
    # Wait, that doesn't sound right either. Let me just use a different approach.
    
    # Alternative: use the fact that the 55-atom Mackay icosahedron is well-known.
    # I'll generate it using a standard algorithm.
    
    # Actually, let me just use a reasonable starting configuration and let L-BFGS relax it.
    # I'll create:
    # - 1 center atom
    # - 12 atoms at icosahedron vertices at radius ~1.12
    # - 42 atoms distributed on a sphere of radius ~2.0
    
    # For the 42 outer atoms, I'll use a Fibonacci sphere or similar distribution
    # and let relaxation fix it up.
    
    # Actually, let me try a more careful construction.
    # The 42 shell-2 atoms in a Mackay icosahedron are at:
    # - 30 edge-midpoint positions
    # - 12 positions that are "opposite" to the 12 vertices (but this is the same as vertices)
    
    # I think the cleanest approach is:
    # Shell 1: 12 icosahedron vertices at radius R1
    # Shell 2: For each edge midpoint (30) and each face center (20)? 30+20=50, too many.
    
    # Let me just use a numerical approach: start with a reasonable guess and relax.
    
    # Center atom
    center = np.array([[0.0, 0.0, 0.0]])
    
    # Shell 1: 12 icosahedron vertices
    shell1 = verts_unit * R1
    
    # Shell 2: 42 atoms. I'll use a combination of edge midpoints and additional points.
    # For now, let me place 42 points using a Fibonacci sphere at radius R2,
    # then relax.
    
    R2 = 2.0
    # Fibonacci sphere for 42 points
    n_outer = 42
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    outer_points = []
    for i in range(n_outer):
        y = 1.0 - 2.0 * i / (n_outer - 1)
        radius_yz = np.sqrt(max(0, 1.0 - y*y))
        theta = golden_angle * i
        x = np.cos(theta) * radius_yz
        z = np.sin(theta) * radius_yz
        outer_points.append([x, y, z])
    outer_points = np.array(outer_points)
    shell2 = outer_points * R2
    
    pos = np.vstack([center, shell1, shell2])
    return pos

def _relax(pos, max_iter=1000, timeout=30.0):
    """Relax positions using L-BFGS. Returns relaxed positions."""
    t0 = time.time()
    
    def objective(x):
        if time.time() - t0 > timeout:
            return 1e10
        p = x.reshape(-1, 3)
        e, g = _lj_energy_grad(p)
        return e, g.flatten()
    
    x0 = pos.flatten()
    res = minimize(objective, x0, method='L-BFGS-B', jac=True,
                   options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-12})
    if time.time() - t0 > timeout:
        return pos
    return res.x.reshape(-1, 3)

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # 10s safety margin before 150s kill
    
    # Start with Mackay icosahedron
    pos = _make_mackay_icosahedron()
    best_pos = pos.copy()
    best_energy = _lj_energy(pos)
    
    # Relax
    if time.time() < deadline - 20.0:
        pos_relaxed = _relax(pos, max_iter=2000, timeout=30.0)
        e_relaxed = _lj_energy(pos_relaxed)
        if e_relaxed < best_energy:
            best_pos = pos_relaxed
            best_energy = e_relaxed
    
    # Try a few more restarts if time allows
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Perturb the best configuration
        noise_scale = 0.05 * (restart + 1)
        pos_perturbed = best_pos + np.random.randn(N, 3) * noise_scale
        
        # Ensure minimum distance constraint
        # Simple check: if any pair is too close, increase noise
        for attempt in range(10):
            d = cdist(pos_perturbed, pos_perturbed)
            np.fill_diagonal(d, np.inf)
            if np.min(d) >= 0.5:
                break
            pos_perturbed = best_pos + np.random.randn(N, 3) * noise_scale * (attempt + 1)
        
        if time.time() > deadline - 10.0:
            break
        
        pos_relaxed = _relax(pos_perturbed, max_iter=1000, timeout=15.0)
        e_relaxed = _lj_energy(pos_relaxed)
        if e_relaxed < best_energy:
            best_pos = pos_relaxed
            best_energy = e_relaxed
    
    # Final check: ensure minimum distance
    d = cdist(best_pos, best_pos)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 0.5:
        # Fallback to a valid configuration
        side = int(np.ceil(N ** (1/3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0]-(side-1)/2)**2 + (p[1]-(side-1)/2)**2 + (p[2]-(side-1)/2)**2)
        best_pos = np.array(points[:N], dtype=float) * 1.12
    
    return best_pos

# EVOLVE-BLOCK-END
