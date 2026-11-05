# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 325.0  # 325s wall clock limit (evaluator kills at 330s)

def _get_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return _DEADLINE - (time.time() - _GLOBAL_START_TIME)

def _morse_energy(x, rho):
    """Compute Morse energy for a set of positions (vectorized)."""
    n = x.shape[0]
    if n < 2:
        return 0.0
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    er = rho * (1.0 - dist)
    er = np.clip(er, -500, 500)
    exp_er = np.exp(er)
    energy = np.sum(exp_er * (exp_er - 2.0)) / 2.0
    return energy

def _min_distance(x):
    """Compute minimum distance between any pair of atoms (vectorized)."""
    n = x.shape[0]
    if n < 2:
        return np.inf
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    return np.min(dist)

def _is_valid(x):
    """Check if the configuration is valid (atoms at least 0.5 apart)."""
    return _min_distance(x) >= 0.5

def _rotation_matrix(axis, angle):
    """Rodrigues rotation matrix for unit axis and angle."""
    axis = np.asarray(axis, dtype=float)
    nrm = np.linalg.norm(axis)
    if nrm < 1e-12:
        return np.eye(3)
    axis = axis / nrm
    c = math.cos(angle)
    s = math.sin(angle)
    x, y, z = axis
    return np.array([
        [c + x*x*(1-c),   x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s, c + y*y*(1-c),   y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)]
    ])

def _generate_icosahedron(n_pts, radius=1.0):
    """Return up to n_pts points on/near an icosahedron of given radius."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    ico = [
        ( 0.0,  1.0,  phi), ( 0.0,  1.0, -phi), ( 0.0, -1.0,  phi), ( 0.0, -1.0, -phi),
        ( 1.0,  phi,  0.0), ( 1.0, -phi,  0.0), (-1.0,  phi,  0.0), (-1.0, -phi,  0.0),
        ( phi,  0.0,  1.0), ( phi,  0.0, -1.0), (-phi,  0.0,  1.0), (-phi,  0.0, -1.0),
    ]
    norm = math.sqrt(1.0 + phi*phi)
    pts = [np.array(v) / norm for v in ico]
    if n_pts <= 12:
        return np.array(pts[:n_pts]) * radius
    # Add face centers
    faces = [
        (0, 1, 4), (0, 4, 2), (0, 2, 6), (0, 6, 8), (0, 8, 1),
        (1, 8, 3), (1, 3, 5), (1, 5, 4), (2, 4, 5), (2, 5, 7),
        (2, 7, 6), (3, 8, 9), (3, 9, 10), (3, 10, 5), (4, 5, 10),
        (4, 10, 9), (6, 7, 11), (6, 11, 9), (6, 9, 8), (7, 5, 10),
    ]
    for f in faces:
        v = (pts[f[0]] + pts[f[1]] + pts[f[2]]) / 3.0
        nv = np.linalg.norm(v)
        if nv > 1e-9:
            pts.append(v / nv)
    if n_pts <= len(pts):
        return np.array(pts[:n_pts]) * radius
    # Fibonacci sphere for rest
    remaining = n_pts - len(pts)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for k in range(remaining):
        y = 1.0 - 2.0 * (k + 0.5) / remaining
        r = math.sqrt(max(0.0, 1.0 - y*y))
        th = golden * k
        pts.append(np.array([r * math.cos(th), y, r * math.sin(th)]))
    return np.array(pts[:n_pts]) * radius

def _generate_fcc_chunk(n_pts, radius=1.0):
    """FCC chunk of up to n_pts atoms, centered, scaled to radius."""
    basis = [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]
    mc = int(math.ceil(radius)) + 2
    pts = []
    for i in range(-mc, mc+1):
        for j in range(-mc, mc+1):
            for k in range(-mc, mc+1):
                for dx, dy, dz in basis:
                    p = np.array([i+dx, j+dy, k+dz], dtype=float)
                    if np.linalg.norm(p) <= radius + 0.5:
                        pts.append(p)
    pts.sort(key=lambda p: np.linalg.norm(p))
    if len(pts) >= n_pts:
        arr = np.array(pts[:n_pts])
    else:
        arr = np.array(pts)
        while len(arr) < n_pts:
            arr = np.vstack([arr, np.array([0.0, 0.0, 0.0])])
    # Center
    arr = arr - arr.mean(axis=0)
    # Scale to target radius
    max_r = np.max(np.linalg.norm(arr, axis=1)) if len(arr) > 0 else 1.0
    if max_r > 1e-9:
        arr = arr * (radius / max_r)
    return arr[:n_pts]

def _generate_bcc_chunk(n_pts, radius=1.0):
    """BCC chunk of up to n_pts atoms, centered, scaled to radius."""
    basis = [(0,0,0), (0.5,0.5,0.5)]
    mc = int(math.ceil(radius)) + 2
    pts = []
    for i in range(-mc, mc+1):
        for j in range(-mc, mc+1):
            for k in range(-mc, mc+1):
                for dx, dy, dz in basis:
                    p = np.array([i+dx, j+dy, k+dz], dtype=float)
                    if np.linalg.norm(p) <= radius + 0.5:
                        pts.append(p)
    pts.sort(key=lambda p: np.linalg.norm(p))
    if len(pts) >= n_pts:
        arr = np.array(pts[:n_pts])
    else:
        arr = np.array(pts)
        while len(arr) < n_pts:
            arr = np.vstack([arr, np.array([0.0, 0.0, 0.0])])
    arr = arr - arr.mean(axis=0)
    max_r = np.max(np.linalg.norm(arr, axis=1)) if len(arr) > 0 else 1.0
    if max_r > 1e-9:
        arr = arr * (radius / max_r)
    return arr[:n_pts]

def _generate_fibonacci_sphere(n_pts, radius=1.0):
    """Fibonacci sphere with n_pts points at given radius."""
    golden = math.pi * (3.0 - math.sqrt(5.0))
    pts = []
    for k in range(n_pts):
        y = 1.0 - 2.0 * (k + 0.5) / n_pts
        r = math.sqrt(max(0.0, 1.0 - y*y))
        th = golden * k
        pts.append(np.array([r * math.cos(th), y, r * math.sin(th)]))
    return np.array(pts) * radius

def _generate_icospiral(n, seed=42):
    """Legacy icosahedral spiral for fallback."""
    rng = np.random.default_rng(seed)
    pts = [np.array([0.0, 0.0, 0.0])]
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    ico = [
        ( 0.0,  1.0,  phi), ( 0.0,  1.0, -phi), ( 0.0, -1.0,  phi), ( 0.0, -1.0, -phi),
        ( 1.0,  phi,  0.0), ( 1.0, -phi,  0.0), (-1.0,  phi,  0.0), (-1.0, -phi,  0.0),
        ( phi,  0.0,  1.0), ( phi,  0.0, -1.0), (-phi,  0.0,  1.0), (-phi,  0.0, -1.0),
    ]
    norm = math.sqrt(1.0 + phi*phi)
    for v in ico:
        pts.append(np.array(v) / norm)
    faces = [
        (0, 1, 4), (0, 4, 2), (0, 2, 6), (0, 6, 8), (0, 8, 1),
        (1, 8, 3), (1, 3, 5), (1, 5, 4), (2, 4, 5), (2, 5, 7),
        (2, 7, 6), (3, 8, 9), (3, 9, 10), (3, 10, 5), (4, 5, 10),
        (4, 10, 9), (6, 7, 11), (6, 11, 9), (6, 9, 8), (7, 5, 10),
    ]
    ico_arr = [np.array(v) / norm for v in ico]
    for f in faces:
        v = (ico_arr[f[0]] + ico_arr[f[1]] + ico_arr[f[2]]) / 3.0
        nv = np.linalg.norm(v)
        if nv > 1e-9:
            pts.append(v / nv)
    remaining = n - len(pts)
    if remaining > 0:
        golden = math.pi * (3.0 - math.sqrt(5.0))
        for k in range(remaining):
            y = 1.0 - 2.0 * (k + 0.5) / remaining
            r = math.sqrt(max(0.0, 1.0 - y*y))
            th = golden * k
            pts.append(np.array([r * math.cos(th), y, r * math.sin(th)]))
    pts = pts[:n]
    jit = rng.uniform(-0.02, 0.02, (len(pts), 3))
    return np.array(pts) + jit

def _generate_fibonacci_shell(n, seed=42):
    """Pure Fibonacci-sphere shell (no center) - good for larger n."""
    rng = np.random.default_rng(seed)
    arr = _generate_fibonacci_sphere(n, radius=1.2)
    jit = rng.uniform(-0.03, 0.03, (n, 3))
    return arr + jit

def _generate_fcc_chunk_full(n, seed=42):
    """Full FCC chunk for direct initialization."""
    rng = np.random.default_rng(seed)
    radius = 1.6 * (n ** (1.0/3.0)) + 1.0
    arr = _generate_fcc_chunk(n, radius=radius)
    jit = rng.uniform(-0.04, 0.04, (n, 3))
    return arr + jit

def _generate_bcc_chunk_full(n, seed=42):
    """Full BCC chunk for direct initialization."""
    rng = np.random.default_rng(seed)
    radius = 1.6 * (n ** (1.0/3.0)) + 1.0
    arr = _generate_bcc_chunk(n, radius=radius)
    jit = rng.uniform(-0.04, 0.04, (n, 3))
    return arr + jit

def _generate_hcp_chunk(n, seed=42):
    """HCP-like chunk: two interpenetrating hexagonal layers."""
    rng = np.random.default_rng(seed)
    pts = []
    a = 1.0
    c = 1.633
    n_layers = max(2, int(2 * (n / 12.0) ** 0.5))
    for layer in range(n_layers):
        z = layer * c
        for ix in range(-6, 7):
            for iy in range(-6, 7):
                x = a * (ix + 0.5 * (iy & 1))
                y = a * math.sqrt(3.0) / 2.0 * iy
                if layer % 2 == 1:
                    x += 0.5 * a
                r = math.sqrt(x*x + y*y)
                if r <= 2.5 * (n ** (1.0/3.0)):
                    pts.append(np.array([x, y, z]))
    pts.sort(key=lambda p: np.linalg.norm(p))
    if len(pts) >= n:
        arr = np.array(pts[:n])
    else:
        while len(pts) < n:
            pts.append(rng.uniform(-3, 3, 3))
        arr = np.array(pts)
    jit = rng.uniform(-0.03, 0.03, (n, 3))
    return arr + jit

def _relax_and_validate(positions, potential, rho):
    """Relax positions and check validity."""
    try:
        energy, relaxed = relax(positions, potential, rho)
        if _min_distance(relaxed) >= 0.5:
            return energy, relaxed
        else:
            return float('inf'), None
    except Exception:
        return float('inf'), None

def _polish(positions, rho, max_iter=300):
    """High-precision L-BFGS-B polish with exact gradient."""
    x_flat = positions.flatten()

    def objective(x):
        pos = x.reshape(-1, 3)
        if _min_distance(pos) < 0.5:
            return 1e10
        return _morse_energy(pos, rho)

    def gradient(x):
        pos = x.reshape(-1, 3)
        if _min_distance(pos) < 0.5:
            return np.zeros_like(x)
        return morse_energy_gradient(pos, rho).flatten()

    try:
        res = minimize(objective, x_flat, jac=gradient, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10,
                                'maxls': 40, 'maxcor': 20})
        if res.fun < 1e10:
            pos_res = res.x.reshape(-1, 3)
            if _min_distance(pos_res) >= 0.5:
                return float(res.fun), pos_res
    except Exception:
        pass

    return _morse_energy(positions, rho), positions

# --- Super-atom (rigid sub-cluster) representation ---

def _superatom_energy_and_grad(params, sub_clusters, rho):
    """
    params: flat array of [pos1(3), axis1(3), angle1, pos2(3), axis2(3), angle2, ...]
    Each sub-cluster has 7 DOF: 3 position, 3 axis, 1 angle.
    Returns (energy, gradient) where gradient is w.r.t. params.
    """
    n_clusters = len(sub_clusters)
    full_pos = np.zeros((sum(sc.shape[0] for sc in sub_clusters), 3))
    offset = 0
    for i, sc in enumerate(sub_clusters):
        pos = params[i*7:i*7+3]
        axis = params[i*7+3:i*7+6]
        angle = params[i*7+6]
        R = _rotation_matrix(axis, angle)
        full_pos[offset:offset+len(sc)] = (sc @ R.T) + pos
        offset += len(sc)
    energy = _morse_energy(full_pos, rho)
    # Numerical gradient (central difference)
    grad = np.zeros_like(params)
    eps = 1e-6
    for j in range(len(params)):
        params[j] += eps
        e_plus, _ = _superatom_energy_only(params, sub_clusters, rho)
        params[j] -= 2*eps
        e_minus, _ = _superatom_energy_only(params, sub_clusters, rho)
        params[j] += eps
        grad[j] = (e_plus - e_minus) / (2*eps)
    return energy, grad

def _superatom_energy_only(params, sub_clusters, rho):
    """Compute energy only (no gradient)."""
    full_pos = np.zeros((sum(sc.shape[0] for sc in sub_clusters), 3))
    offset = 0
    for i, sc in enumerate(sub_clusters):
        pos = params[i*7:i*7+3]
        axis = params[i*7+3:i*7+6]
        angle = params[i*7+6]
        R = _rotation_matrix(axis, angle)
        full_pos[offset:offset+len(sc)] = (sc @ R.T) + pos
        offset += len(sc)
    return _morse_energy(full_pos, rho), full_pos

def _superatom_to_positions(params, sub_clusters):
    """Convert super-atom params to full position array."""
    full_pos = np.zeros((sum(sc.shape[0] for sc in sub_clusters), 3))
    offset = 0
    for i, sc in enumerate(sub_clusters):
        pos = params[i*7:i*7+3]
        axis = params[i*7+3:i*7+6]
        angle = params[i*7+6]
        R = _rotation_matrix(axis, angle)
        full_pos[offset:offset+len(sc)] = (sc @ R.T) + pos
        offset += len(sc)
    return full_pos

def _split_into_subclusters(positions, n_clusters, rho, seed=0):
    """
    Split a relaxed configuration into n_clusters rigid sub-clusters.
    Uses simple spatial partitioning: sort by x, y, z and split into groups.
    Returns list of sub-cluster position arrays (centered at origin).
    """
    n = positions.shape[0]
    if n_clusters <= 1:
        return [positions - positions.mean(axis=0)]
    
    # Try multiple partition strategies and pick the one with lowest internal energy
    best_subclusters = None
    best_internal_energy = float('inf')
    
    rng = np.random.default_rng(seed)
    
    for strategy in range(3):
        if strategy == 0:
            # Sort by x
            order = np.argsort(positions[:, 0])
        elif strategy == 1:
            # Sort by y
            order = np.argsort(positions[:, 1])
        else:
            # Sort by projection onto random axis
            axis = rng.standard_normal(3)
            axis /= np.linalg.norm(axis)
            order = np.argsort(positions @ axis)
        
        # Split into consecutive groups
        group_sizes = [n // n_clusters] * n_clusters
        remainder = n % n_clusters
        for i in range(remainder):
            group_sizes[i] += 1
        
        subclusters = []
        idx = 0
        for size in group_sizes:
            group_idx = order[idx:idx+size]
            group_pos = positions[group_idx]
            # Center the group
            group_pos = group_pos - group_pos.mean(axis=0)
            subclusters.append(group_pos)
            idx += size
        
        # Compute internal energy of this partitioning
        full_check = np.zeros((n, 3))
        offset = 0
        for sc in subclusters:
            full_check[offset:offset+len(sc)] = sc
            offset += len(sc)
        internal_e = _morse_energy(full_check, rho)
        
        if internal_e < best_internal_energy:
            best_internal_energy = internal_e
            best_subclusters = subclusters
    
    return best_subclusters

def _superatom_basin_hopping(positions, n_clusters, rho, potential, max_iterations=50, seed=0):
    """
    Basin hopping in super-atom space.
    Returns (best_energy, best_positions).
    """
    rng = np.random.default_rng(seed)
    
    # Initial split
    sub_clusters = _split_into_subclusters(positions, n_clusters, rho, seed=seed)
    n_clusters_actual = len(sub_clusters)
    
    # Initial params: identity rotation, positions at cluster centroids
    params = np.zeros(n_clusters_actual * 7)
    for i, sc in enumerate(sub_clusters):
        params[i*7:i*7+3] = [0.0, 0.0, 0.0]  # Will be set below
        params[i*7+3:i*7+6] = [0.0, 0.0, 1.0]  # z-axis
        params[i*7+6] = 0.0
    
    # Set initial positions to match original configuration
    # We need to find the centroids of each group in the original positions
    n = positions.shape[0]
    group_sizes = [len(sc) for sc in sub_clusters]
    # Re-derive the same ordering
    order = np.argsort(positions[:, 0])
    idx = 0
    for i, sc in enumerate(sub_clusters):
        group_idx = order[idx:idx+len(sc)]
        centroid = positions[group_idx].mean(axis=0)
        params[i*7:i*7+3] = centroid
        idx += len(sc)
    
    # Relax in super-atom space
    def sa_objective(p):
        e, full_pos = _superatom_energy_only(p, sub_clusters, rho)
        if _min_distance(full_pos) < 0.5:
            return e + 1e6 * (0.5 - _min_distance(full_pos))**2
        return e
    
    try:
        res = minimize(sa_objective, params, method='L-BFGS-B',
                       options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < 1e10:
            full_pos = _superatom_to_positions(res.x, sub_clusters)
            if _min_distance(full_pos) >= 0.5:
                energy, relaxed = _relax_and_validate(full_pos, potential, rho)
                if relaxed is not None:
                    return energy, relaxed
    except Exception:
        pass
    
    best_energy = float('inf')
    best_positions = None
    
    current_params = params.copy()
    current_energy, current_full = _superatom_energy_only(current_params, sub_clusters, rho)
    if _min_distance(current_full) >= 0.5:
        best_energy = current_energy
        best_positions = current_full.copy()
    
    temp = 2.0
    
    for it in range(max_iterations):
        if _get_remaining_time() < 5.0:
            break
        
        # Perturb: random small rotation and translation of a random sub-cluster
        i = rng.integers(0, n_clusters_actual)
        angle_delta = rng.uniform(-0.3, 0.3)
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        
        new_params = current_params.copy()
        # Apply incremental rotation
        old_axis = new_params[i*7+3:i*7+6]
        old_angle = new_params[i*7+6]
        # Compose rotations: new = R(axis, angle_delta) @ R(old_axis, old_angle)
        # For simplicity, just add to angle and use same axis (approximate)
        new_params[i*7+6] = old_angle + angle_delta
        # Small random translation
        new_params[i*7:i*7+3] += rng.standard_normal(3) * 0.05
        
        new_energy, new_full = _superatom_energy_only(new_params, sub_clusters, rho)
        if _min_distance(new_full) < 0.5:
            # Penalty
            new_energy += 1e6 * (0.5 - _min_distance(new_full))**2
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or (temp > 1e-6 and rng.random() < math.exp(-delta_e / temp)):
            current_params = new_params
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = new_full.copy()
        
        temp *= 0.95
    
    if best_positions is None:
        return float('inf'), None
    
    # Final full relaxation
    energy, relaxed = _relax_and_validate(best_positions, potential, rho)
    if relaxed is not None:
        return energy, relaxed
    return best_energy, best_positions

def _perturb_rotate(positions, angle, rng):
    """
    Structured perturbation: rotate a random subset of atoms around a random axis
    through the centroid.
    """
    n = positions.shape[0]
    new_pos = positions.copy()
    centroid = positions.mean(axis=0)
    axis = rng.standard_normal(3)
    R = _rotation_matrix(axis, angle)
    frac = 0.5 + 0.3 * rng.random()
    k = max(1, int(frac * n))
    idx = rng.choice(n, size=k, replace=False)
    new_pos[idx] = (positions[idx] - centroid) @ R.T + centroid
    return new_pos

def _basin_hop_rotation(start_positions, potential, rho, max_iterations=80,
                        angle_range=(0.1, 0.6), seed=0):
    """
    Basin hopping with rotation-based perturbations.
    Returns (best_energy, best_positions).
    """
    rng = np.random.default_rng(seed)
    current_energy, current_positions = _relax_and_validate(start_positions, potential, rho)

    if current_positions is None:
        perturbed = start_positions + rng.standard_normal(start_positions.shape) * 0.1
        current_energy, current_positions = _relax_and_validate(perturbed, potential, rho)
        if current_positions is None:
            return float('inf'), None

    best_energy = current_energy
    best_positions = current_positions.copy()
    temp = 3.0
    a_lo, a_hi = angle_range

    for it in range(max_iterations):
        if _get_remaining_time() < 5.0:
            break

        lo = a_lo * (0.5 ** (it / max(1, max_iterations)))
        hi = a_hi * (0.5 ** (it / max(1, max_iterations)))
        angle = rng.uniform(lo, hi)
        new_positions = _perturb_rotate(current_positions, angle, rng)

        new_energy, new_relaxed = _relax_and_validate(new_positions, potential, rho)

        if new_relaxed is None:
            angle2 = angle * 0.3
            new_positions = _perturb_rotate(current_positions, angle2, rng)
            new_energy, new_relaxed = _relax_and_validate(new_positions, potential, rho)

        if new_relaxed is None:
            continue

        delta_e = new_energy - current_energy
        if delta_e < 0 or (temp > 1e-6 and rng.random() < math.exp(-delta_e / temp)):
            current_energy = new_energy
            current_positions = new_relaxed
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_positions.copy()

        temp *= 0.95

    return best_energy, best_positions

def construct(instance):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()

    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]

    if rho is None:
        rho = 6

    best_energy = float('inf')
    best_positions = None

    # Structured initial guesses
    initial_guesses = []
    generators = [
        lambda: _generate_icospiral(n, seed=42),
        lambda: _generate_icospiral(n, seed=7),
        lambda: _generate_fibonacci_shell(n, seed=42),
        lambda: _generate_fcc_chunk_full(n, seed=42),
        lambda: _generate_bcc_chunk_full(n, seed=42),
        lambda: _generate_hcp_chunk(n, seed=42),
    ]
    for gen in generators:
        try:
            g = gen()
            if g.shape[0] == n:
                initial_guesses.append(g)
        except Exception:
            pass

    # Relax all initial guesses
    relaxed_results = []
    for guess in initial_guesses:
        if _get_remaining_time() < 10.0:
            break
        energy, positions = _relax_and_validate(guess, potential, rho)
        if positions is not None and energy < float('inf'):
            relaxed_results.append((energy, positions))

    relaxed_results.sort(key=lambda x: x[0])

    if relaxed_results:
        best_energy = relaxed_results[0][0]
        best_positions = relaxed_results[0][1].copy()

    # Super-atom basin hopping from top starts
    # Determine number of sub-clusters based on n
    if n <= 40:
        n_subclusters = 3
    elif n <= 60:
        n_subclusters = 4
    else:
        n_subclusters = 5

    if _get_remaining_time() > 30.0:
        num_sa = min(len(relaxed_results), 3)
        for idx in range(num_sa):
            if _get_remaining_time() < 15.0:
                break
            start_pos = relaxed_results[idx][1]
            sa_seed = 200 + idx * 23
            sa_energy, sa_positions = _superatom_basin_hopping(
                start_pos, n_subclusters, rho, potential,
                max_iterations=40, seed=sa_seed
            )
            if sa_positions is not None and sa_energy < best_energy:
                best_energy = sa_energy
                best_positions = sa_positions.copy()

    # Rotation-based basin hopping from top starts
    num_bh = min(len(relaxed_results), 4)
    if rho <= 3:
        angle_range = (0.15, 0.8)
    elif rho <= 6:
        angle_range = (0.10, 0.5)
    else:
        angle_range = (0.05, 0.3)

    if n <= 40:
        max_iter = 100
    elif n <= 60:
        max_iter = 70
    else:
        max_iter = 45

    for idx in range(num_bh):
        if _get_remaining_time() < 15.0:
            break
        start_pos = relaxed_results[idx][1]
        bh_seed = 100 + idx * 17
        bh_energy, bh_positions = _basin_hop_rotation(
            start_pos, potential, rho,
            max_iterations=max_iter,
            angle_range=angle_range,
            seed=bh_seed,
        )
        if bh_positions is not None and bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_positions.copy()

        if _get_remaining_time() > 20.0 and idx < 2:
            bh_energy2, bh_positions2 = _basin_hop_rotation(
                start_pos, potential, rho,
                max_iterations=max_iter // 2,
                angle_range=(angle_range[0] * 1.5, angle_range[1] * 1.5),
                seed=bh_seed + 7,
            )
            if bh_positions2 is not None and bh_energy2 < best_energy:
                best_energy = bh_energy2
                best_positions = bh_positions2.copy()

    # Final high-precision polish
    if best_positions is not None and _get_remaining_time() > 5.0:
        polished_energy, polished_pos = _polish(best_positions, rho, max_iter=300)
        if polished_pos is not None and polished_energy < best_energy:
            best_energy = polished_energy
            best_positions = polished_pos

    # Fallback
    if best_positions is None:
        best_positions = _generate_icospiral(n, seed=0)
        if not _is_valid(best_positions):
            best_positions = best_positions * 1.5
            if not _is_valid(best_positions):
                best_positions = _generate_fibonacci_shell(n, seed=99)

    return best_positions
# EVOLVE-BLOCK-END
