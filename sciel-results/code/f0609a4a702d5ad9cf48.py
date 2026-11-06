import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
from scipy.optimize import minimize
import time

def _get_icosahedron_vertices(radius=1.0):
    """Generate the 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi],
        [0, -1, phi],
        [0, 1, -phi],
        [0, -1, -phi],
        [1, phi, 0],
        [-1, phi, 0],
        [1, -phi, 0],
        [-1, -phi, 0],
        [phi, 0, 1],
        [phi, 0, -1],
        [-phi, 0, 1],
        [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    return verts * radius

def _get_icosahedron_edges():
    """Get the 30 edges of the icosahedron (pairs of vertex indices)."""
    verts = _get_icosahedron_vertices(1.0)
    min_dist = np.inf
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < min_dist:
                min_dist = d
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, min_dist, rtol=1e-6):
                edges.append((i, j))
    return edges

def _build_mackay_55(r1=1.0, r2=1.9):
    """
    Build the 55-atom Mackay icosahedron:
    - 1 center
    - 12 vertices (shell 1)
    - 30 edge midpoints (shell 2)
    Total: 1 + 12 + 30 = 43... wait, that's 43 not 55.
    
    Actually the Mackay icosahedron for N=55 has:
    - 1 center
    - 12 vertices of icosahedron
    - 30 face centers? No...
    
    Let me reconsider. The Mackay icosahedron shells:
    Shell 0: 1 (center)
    Shell 1: 12 (icosahedron vertices)
    Shell 2: 42 (12 vertices + 30 edge midpoints)
    Total for shell 2 complete: 1+12+42 = 55
    
    So shell 2 has 42 atoms: the 12 icosahedron vertices projected to shell 2 radius,
    plus 30 edge midpoints projected to shell 2 radius.
    """
    base_verts = _get_icosahedron_vertices(1.0)
    edges = _get_icosahedron_edges()
    
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 1: 12 icosahedron vertices at radius r1
    for v in base_verts:
        points.append(v * r1)
    
    # Shell 2: 42 atoms at radius r2
    # 12 vertices direction + 30 edge midpoint directions
    shell2_dirs = []
    for v in base_verts:
        shell2_dirs.append(v)
    for i, j in edges:
        mid = (base_verts[i] + base_verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        shell2_dirs.append(mid)
    
    for d in shell2_dirs:
        points.append(d * r2)
    
    return np.array(points)

def _build_mackay_n(n, r1=1.0, r2=1.9):
    """
    Build a Mackay-like icosahedral cluster of n atoms.
    For n <= 13: center + up to 12 icosahedron vertices
    For 14 <= n <= 55: center + 12 shell1 + up to 42 shell2
    For n > 55: add more shells
    """
    base_verts = _get_icosahedron_vertices(1.0)
    edges = _get_icosahedron_edges()
    
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 1: 12 icosahedron vertices at radius r1
    if n > 1:
        for v in base_verts:
            if len(points) < n:
                points.append(v * r1)
    
    # Shell 2: 42 atoms at radius r2
    if n > 13:
        shell2_dirs = []
        for v in base_verts:
            shell2_dirs.append(v)
        for i, j in edges:
            mid = (base_verts[i] + base_verts[j]) / 2.0
            mid = mid / np.linalg.norm(mid)
            shell2_dirs.append(mid)
        
        for d in shell2_dirs:
            if len(points) < n:
                points.append(d * r2)
    
    # Shell 3 for n > 55: use face centers and other directions
    if n > 55:
        # Get face centers of the icosahedron (20 faces)
        # Each face is a triangle of 3 vertices
        # We need to find the 20 triangular faces
        verts = base_verts
        edges_set = set()
        for i, j in edges:
            edges_set.add((min(i,j), max(i,j)))
        
        # Find triangles (faces)
        faces = []
        for i in range(12):
            for j in range(i+1, 12):
                if (min(i,j), max(i,j)) not in edges_set:
                    continue
                for k in range(j+1, 12):
                    if (min(j,k), max(j,k)) not in edges_set:
                        continue
                    if (min(i,k), max(i,k)) not in edges_set:
                        continue
                    faces.append((i, j, k))
        
        # Remove duplicate faces (each face appears 3 times from 3 edges)
        unique_faces = []
        seen = set()
        for f in faces:
            key = tuple(sorted(f))
            if key not in seen:
                seen.add(key)
                unique_faces.append(f)
        
        shell3_dirs = []
        # 20 face centers
        for f in unique_faces:
            fc = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
            fc = fc / np.linalg.norm(fc)
            shell3_dirs.append(fc)
        
        # Add more directions: edge midpoints at different level
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2.0
            mid = mid / np.linalg.norm(mid)
            shell3_dirs.append(mid)
        
        r3 = r2 + 0.8
        for d in shell3_dirs:
            if len(points) < n:
                points.append(d * r3)
        
        # If still not enough, add more
        while len(points) < n:
            r_extra = r3 + 0.5
            for v in base_verts:
                if len(points) < n:
                    points.append(v * r_extra)
            r_extra += 0.5
    
    return np.array(points[:n])

def _defect_seed_from_55(n, seed=42):
    """
    Generate an n-atom configuration by modifying the 55-atom Mackay icosahedron.
    For n < 55: remove atoms from the outer shell (shell 2) in a symmetry-respecting way.
    For n > 55: add atoms.
    """
    if n == 55:
        return _build_mackay_55()
    
    mackay55 = _build_mackay_55()
    rng = np.random.default_rng(seed)
    
    if n < 55:
        # Atoms: index 0 = center, 1-12 = shell1, 13-54 = shell2
        # Preferentially remove from shell 2 (outer)
        # Shell 2 has 42 atoms (indices 13-54)
        # Shell 1 has 12 atoms (indices 1-12)
        
        num_to_remove = 55 - n
        
        # Build a priority list: remove shell 2 first, then shell 1
        shell2_indices = list(range(13, 55))  # 42 atoms
        shell1_indices = list(range(1, 13))    # 12 atoms
        
        # Shuffle within each shell for diversity
        rng.shuffle(shell2_indices)
        rng.shuffle(shell1_indices)
        
        remove_list = []
        for idx in shell2_indices:
            if len(remove_list) < num_to_remove:
                remove_list.append(idx)
        for idx in shell1_indices:
            if len(remove_list) < num_to_remove:
                remove_list.append(idx)
        
        remove_set = set(remove_list[:num_to_remove])
        keep_indices = [i for i in range(55) if i not in remove_set]
        
        return mackay55[keep_indices][:n]
    else:
        # n > 55: start with Mackay 55 and add atoms
        points = list(mackay55)
        base_verts = _get_icosahedron_vertices(1.0)
        r3 = 2.7
        for v in base_verts:
            if len(points) < n:
                points.append(v * r3)
        while len(points) < n:
            r_extra = r3 + 0.5
            for v in base_verts:
                if len(points) < n:
                    points.append(v * r_extra)
            r_extra += 0.5
        return np.array(points[:n])

def _dzugutov_energy_and_gradient(positions):
    """Compute Dzugutov energy and gradient using the provided function."""
    try:
        grad = dzugutov_energy_gradient(positions)
        # The function returns gradient; we need energy too
        # Let's compute energy manually
        A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
        n = len(positions)
        energy = 0.0
        for i in range(n):
            for j in range(i+1, n):
                r = np.linalg.norm(positions[i] - positions[j])
                if r < 1e-10:
                    continue
                if r < a:
                    energy += A * (r**(-m) - B) * np.exp(c / (r - a))
                elif r < b:
                    energy += B * np.exp(d / (r - b))
                else:
                    energy += 0.0
        return energy, grad
    except Exception:
        return None, None

def _l_bfgs_optimize(positions, potential, max_iter=200, time_limit=None, start_time=None):
    """
    Use L-BFGS-B with the Dzugutov gradient for optimization.
    """
    def objective_and_grad(flat_pos):
        pos = flat_pos.reshape(-1, 3)
        try:
            grad = dzugutov_energy_gradient(pos)
            # Compute energy
            A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
            n = len(pos)
            energy = 0.0
            for i in range(n):
                for j in range(i+1, n):
                    r = np.linalg.norm(pos[i] - pos[j])
                    if r < 1e-10:
                        continue
                    if r < a:
                        energy += A * (r**(-m) - B) * np.exp(c / (r - a))
                    elif r < b:
                        energy += B * np.exp(d / (r - b))
            return energy, grad.ravel()
        except Exception:
            return 1e10, np.zeros_like(flat_pos)
    
    try:
        result = minimize(
            objective_and_grad,
            positions.ravel(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'maxfun': max_iter * 3, 'gtol': 1e-6}
        )
        if result.success or result.fun < 1e10:
            opt_pos = result.x.reshape(-1, 3)
            opt_energy = result.fun
            return opt_energy, opt_pos
    except Exception:
        pass
    
    return None, None

def _check_min_distance(positions, min_dist=0.5):
    """Check if all pairs are at least min_dist apart."""
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(positions[i] - positions[j]) < min_dist:
                return False
    return True

def construct(instance):
    """
    Find low-energy Dzugutov cluster structures using defect-seeded icosahedral
    initialization and L-BFGS-B gradient-based optimization with multi-start.
    """
    size = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + 285  # 15s margin before 300s limit
    
    best_energy = float('inf')
    best_positions = None
    
    def try_optimize(init_positions, label=""):
        nonlocal best_energy, best_positions
        if time.time() > deadline:
            return False
        
        # First try the fast relax
        try:
            energy, positions = relax(init_positions, potential, None)
            if energy < best_energy and _check_min_distance(positions):
                best_energy = energy
                best_positions = positions
        except Exception:
            pass
        
        # Then try L-BFGS for more thorough optimization
        if time.time() < deadline - 5:
            try:
                # Use the relaxed position as starting point for L-BFGS
                lpos = positions if 'positions' in dir() else init_positions
                opt_energy, opt_pos = _l_bfgs_optimize(lpos, potential, max_iter=100)
                if opt_energy is not None and opt_energy < best_energy:
                    if _check_min_distance(opt_pos):
                        best_energy = opt_energy
                        best_positions = opt_pos
            except Exception:
                pass
        
        return True
    
    # Strategy: multiple defect-seeded starts from the 55-atom Mackay icosahedron
    # plus some geometric initializations
    
    # 1. Direct Mackay icosahedral construction
    try:
        init_mackay = _build_mackay_n(size)
        try_optimize(init_mackay, "mackay_direct")
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_positions if best_positions is not None else _build_mackay_n(size)
    
    # 2. Defect-seeded from 55-atom (multiple seeds for diversity)
    for seed in [42, 123, 456, 789, 1000]:
        if time.time() > deadline - 10:
            break
        try:
            init_defect = _defect_seed_from_55(size, seed=seed)
            if len(init_defect) == size:
                try_optimize(init_defect, f"defect_{seed}")
        except Exception:
            pass
    
    if time.time() > deadline:
        return best_positions if best_positions is not None else _build_mackay_n(size)
    
    # 3. Perturbed Mackay starts
    if best_positions is not None:
        for scale in [0.15, 0.25, 0.35]:
            if time.time() > deadline - 10:
                break
            try:
                rng = np.random.default_rng(size + int(scale * 1000))
                perturbed = best_positions + rng.normal(0, scale, best_positions.shape)
                try_optimize(perturbed, f"perturbed_{scale}")
            except Exception:
                pass
    
    if time.time() > deadline:
        return best_positions if best_positions is not None else _build_mackay_n(size)
    
    # 4. Additional diverse starts if time allows
    if time.time() < deadline - 30:
        # Random compact start
        try:
            rng = np.random.default_rng(size + 999)
            radius = 1.5 * size ** (1.0/3.0)
            points = []
            attempts = 0
            while len(points) < size and attempts < size * 300:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) > radius:
                    attempts += 1
                    continue
                if points:
                    dists = np.linalg.norm(np.array(points) - p, axis=1)
                    if np.min(dists) < 0.7:
                        attempts += 1
                        continue
                points.append(p)
                attempts += 1
            if len(points) >= size:
                init_rand = np.array(points[:size])
                try_optimize(init_rand, "random")
        except Exception:
            pass
    
    if time.time() > deadline:
        return best_positions if best_positions is not None else _build_mackay_n(size)
    
    # 5. Final perturbation rounds if time remains
    if best_positions is not None and time.time() < deadline - 15:
        rng = np.random.default_rng(size + 2024)
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            try:
                perturbed = best_positions + rng.normal(0, 0.1, best_positions.shape)
                try_optimize(perturbed, "final_perturb")
            except Exception:
                pass
    
    if best_positions is None:
        return _build_mackay_n(size)
        
    return best_positions
