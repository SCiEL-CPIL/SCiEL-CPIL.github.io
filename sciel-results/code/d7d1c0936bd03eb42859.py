"""LJ38 cluster global minimum search using basin hopping with multi-start."""
import numpy as np
import time
import sys

N = 38

# Deadline guard: evaluator kills at 150s wall-clock; we aim to finish well before.
DEADLINE = 145.0  # seconds, safety margin


def _lj_energy_and_grad(pos):
    """Compute total LJ energy and gradient for N atoms in reduced units.
    
    Energy: 4 * sum_{i<j} (r_ij^-12 - r_ij^-6)
    """
    pos = np.asarray(pos, dtype=np.float64)
    n = len(pos)
    # Pairwise distances
    # Use vectorized computation
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (n, n, 3)
    r2 = np.sum(diff * diff, axis=2)  # (n, n)
    # Avoid division by zero on diagonal
    np.fill_diagonal(r2, 1.0)
    r = np.sqrt(r2)
    r6 = r2 * r2 * r2  # r^6
    r12 = r6 * r6
    # Energy per pair (upper triangle only)
    # E_pair = 4 * (r^-12 - r^-6)
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    # Zero out diagonal
    np.fill_diagonal(pair_energy, 0.0)
    # Total energy is half the sum (each pair counted twice)
    energy = 0.5 * np.sum(pair_energy)
    
    # Gradient: dE/dx_i = sum_{j != i} 4 * (-12 * r^-13 + 6 * r^-7) * (x_i - x_j) / r
    # = sum_{j != i} 4 * (6 * r^-7 - 12 * r^-13) * (x_i - x_j) / r
    # = sum_{j != i} 4 * (6/r^8 - 12/r^14) * (x_i - x_j)
    # Let's compute coefficient: c_ij = 4 * (6/r^8 - 12/r^14)
    r8 = r6 * r2
    r14 = r12 * r2
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)  # (n, n)
    np.fill_diagonal(coeff, 0.0)
    # grad_i = sum_j coeff_ij * (pos_i - pos_j)
    # diff[i,j] = pos_i - pos_j
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad


def _lj_energy_only(pos):
    """Compute only the LJ energy (faster, no gradient)."""
    pos = np.asarray(pos, dtype=np.float64)
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 * r2 * r2
    r12 = r6 * r6
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    return 0.5 * np.sum(pair_energy)


def _local_minimize(pos, maxiter=200, tol=1e-10):
    """Local minimization using L-BFGS-B. Returns (minimized_pos, energy)."""
    from scipy.optimize import minimize
    
    def fun(x):
        e, _ = _lj_energy_and_grad(x)
        return e
    
    def jac(x):
        _, g = _lj_energy_and_grad(x)
        return g
    
    try:
        result = minimize(fun, pos, jac=jac, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'ftol': tol, 'gtol': 1e-8, 'maxls': 50})
        return result.x, result.fun
    except Exception:
        return pos, _lj_energy_only(pos)


def _make_icosahedral_start():
    """Create an icosahedral-like starting configuration for 38 atoms.
    
    Icosahedron has 12 vertices, 30 edges, 20 faces.
    For 38 atoms: 1 center + 12 icosahedral vertices + 15 face centers + 10 extra
    Or: 1 + 12 + 15 + 10 = 38
    """
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    
    # 12 vertices of icosahedron (radius 1)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit radius
    verts /= np.linalg.norm(verts[0])
    
    # 1 center
    center = np.array([[0.0, 0.0, 0.0]])
    
    # 15 face centers: each face is an equilateral triangle
    # Find faces by checking which triples of vertices form faces
    # For an icosahedron, each face is a triangle where all three vertices are at distance 2 from each other
    # In our normalized icosahedron, edge length = 2 / |verts[0]|
    edge_len = np.linalg.norm(verts[0] - verts[1])
    
    face_centers = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if (abs(d1 - edge_len) < 1e-6 and 
                    abs(d2 - edge_len) < 1e-6 and 
                    abs(d3 - edge_len) < 1e-6):
                    fc = (verts[i] + verts[j] + verts[k]) / 3.0
                    face_centers.append(fc)
    
    face_centers = np.array(face_centers)
    
    # We need 38 atoms: 1 + 12 + 15 + 10 = 38
    # 10 remaining: place them along edges or in other symmetric positions
    # Let's add 10 more atoms at edge midpoints (there are 30 edges, pick 10)
    edge_midpoints = []
    edge_set = set()
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if abs(d - edge_len) < 1e-6:
                mid = (verts[i] + verts[j]) / 2.0
                edge_midpoints.append(mid)
                edge_set.add((i, j))
    
    edge_midpoints = np.array(edge_midpoints[:10])
    
    # Combine all
    pos = np.vstack([center, verts, face_centers, edge_midpoints])
    
    # Scale to reasonable LJ distances (minimum at r=2^(1/6) ~ 1.122)
    # The icosahedral shell radius should be around 1.122 for nearest neighbors
    # Center to vertex distance = 1.0 in our normalized coords
    # Scale so that center-to-vertex distance is ~1.122
    scale = 1.122
    pos *= scale
    
    return pos


def _make_fcc_truncated_octahedron():
    """Create FCC truncated octahedron configuration for 38 atoms.
    
    FCC truncated octahedron for N=38:
    This is the known global minimum structure.
    Layers: 
    - Center layer
    - Surrounding layers in FCC arrangement
    """
    # FCC truncated octahedron has 38 atoms arranged as:
    # A central atom, surrounded by 12 in a cuboctahedral shell,
    # then 14 in another layer, then 11 in the outer layer
    # Or more precisely, it's a truncated octahedron with 8 hexagonal faces and 6 square faces
    
    # Let's use a known construction:
    # The 38-atom FCC cluster can be described by coordinates
    # Using the Cambridge Cluster Database description
    
    # For now, create a reasonable FCC-like cluster
    # FCC unit cell with lattice parameter a
    # Atoms at FCC positions within a truncated octahedron
    
    # Simpler approach: use the known energy and create a close-packed structure
    # Let's build it layer by layer
    
    a = 1.122  # nearest neighbor distance
    
    # FCC truncated octahedron for 38 atoms:
    # Layer structure (viewed along [111]):
    # Actually, let's use a different approach.
    # The 38-atom FCC cluster is a truncated octahedron.
    
    # Coordinates for FCC truncated octahedron (38 atoms):
    # Using the description from the literature:
    # 8 hexagonal faces, 6 square faces
    # Vertices: 24, Edge centers: 12, Face centers: 2, plus interior
    
    # Let me use a parametric construction
    # Truncated octahedron with inradius r
    # The 38-atom cluster has specific coordinates
    
    # Alternative: build from FCC lattice
    # FCC positions in a cubic cell: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) times a
    # For a truncated octahedron, we take a subset of FCC sites
    
    # Let's use a simpler heuristic: generate FCC points and select closest to center
    # FCC lattice points
    points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                # FCC basis
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    x = (i + dx) * a
                    y = (j + dy) * a
                    z = (k + dz) * a
                    points.append([x, y, z])
    
    points = np.array(points)
    # Sort by distance from origin
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    pos = points[idx[:N]]
    
    return pos


def _random_start(radius_range=(1.0, 3.0)):
    """Generate a random configuration with atoms in a spherical shell."""
    pos = np.random.randn(N, 3)
    # Scale to reasonable radius
    norms = np.linalg.norm(pos, axis=1)
    target_radii = np.random.uniform(radius_range[0], radius_range[1], N)
    pos = pos / norms[:, np.newaxis] * target_radii[:, np.newaxis]
    return pos


def _basin_hopping(pos0, n_iter=50, step_size=0.3, maxiter_local=100):
    """Basin hopping algorithm to escape local minima."""
    current_pos, current_energy = _local_minimize(pos0, maxiter=maxiter_local)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    rng = np.random.default_rng(42)
    
    for i in range(n_iter):
        # Random perturbation
        noise = rng.normal(0, step_size, size=(N, 3))
        trial_pos = current_pos + noise
        
        # Local minimize
        trial_pos, trial_energy = _local_minimize(trial_pos, maxiter=maxiter_local)
        
        # Metropolis acceptance
        if trial_energy < current_energy:
            current_pos = trial_pos
            current_energy = trial_energy
        else:
            delta_e = trial_energy - current_energy
            # Simple acceptance with decreasing temperature
            T = max(0.01, 0.5 * np.exp(-i / 10.0))
            if rng.random() < np.exp(-delta_e / T):
                current_pos = trial_pos
                current_energy = trial_energy
        
        # Update best
        if current_energy < best_energy:
            best_pos = current_pos.copy()
            best_energy = current_energy
    
    return best_pos, best_energy


def construct():
    """Construct 38-atom LJ cluster configuration minimizing total energy."""
    start_time = time.time()
    
    # Strategy: multi-start with basin hopping
    # Try multiple initial configurations:
    # 1. Icosahedral-like
    # 2. FCC-like  
    # 3. Random configurations
    
    best_pos = None
    best_energy = np.inf
    
    # Generate diverse starting configurations
    starts = []
    
    # 1. Icosahedral start
    try:
        starts.append(_make_icosahedral_start())
    except Exception:
        pass
    
    # 2. FCC truncated octahedron start
    try:
        starts.append(_make_fcc_truncated_octahedron())
    except Exception:
        pass
    
    # 3. Random starts
    rng = np.random.default_rng(123)
    for _ in range(5):
        pos = rng.randn(N, 3)
        norms = np.linalg.norm(pos, axis=1)
        target_radii = rng.uniform(1.0, 2.5, N)
        pos = pos / norms[:, np.newaxis] * target_radii[:, np.newaxis]
        starts.append(pos)
    
    # 4. More structured random starts
    for _ in range(3):
        # Shell-based random
        pos = np.zeros((N, 3))
        # Center atom
        pos[0] = [0, 0, 0]
        # 12 atoms on first shell
        phi = (1 + np.sqrt(5)) / 2
        icos_verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        icos_verts /= np.linalg.norm(icos_verts[0])
        r1 = 1.122 * (1 + rng.uniform(-0.1, 0.1))
        pos[1:13] = icos_verts * r1
        # Remaining 25 atoms on outer shell
        for i in range(13, N):
            direction = rng.randn(3)
            direction /= np.linalg.norm(direction)
            r2 = 1.122 * (1.5 + rng.uniform(-0.2, 0.2))
            pos[i] = direction * r2
        starts.append(pos)
    
    # Process each start with basin hopping
    for idx, start in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 30:  # Leave 30s buffer
            break
        
        try:
            # Quick local minimize first
            pos_min, e_min = _local_minimize(start, maxiter=50)
            
            if e_min < best_energy:
                best_pos = pos_min.copy()
                best_energy = e_min
            
            # If we have time, do basin hopping
            elapsed = time.time() - start_time
            if elapsed < DEADLINE - 60 and idx < 3:  # Only do BH for first few starts
                pos_bh, e_bh = _basin_hopping(start, n_iter=20, step_size=0.25, maxiter_local=50)
                if e_bh < best_energy:
                    best_pos = pos_bh.copy()
                    best_energy = e_bh
        except Exception as e:
            continue
    
    # Final polish: tight local minimization on best
    if best_pos is not None:
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 10:
            try:
                best_pos, best_energy = _local_minimize(best_pos, maxiter=500, tol=1e-12)
            except Exception:
                pass
    
    if best_pos is None:
        # Fallback: simple cubic lattice
        side = int(np.ceil(N ** (1/3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side-1)/2)**2 + (p[1] - (side-1)/2)**2 + (p[2] - (side-1)/2)**2)
        best_pos = np.array(points[:N], dtype=float) * 1.12
    
    # Ensure minimum distance constraint (atoms at least 0.5 apart)
    # Check and fix if needed
    for _ in range(100):
        diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        if min_dist >= 0.5:
            break
        # Nudge atoms apart
        for i in range(N):
            for j in range(i+1, N):
                if dists[i, j] < 0.5:
                    d = best_pos[i] - best_pos[j]
                    dn = np.linalg.norm(d)
                    if dn > 1e-10:
                        push = (0.5 - dn) / dn * 0.5
                        best_pos[i] += d * push
                        best_pos[j] -= d * push
    
    return best_pos
