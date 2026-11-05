"""Dzugutov cluster global minima search using polytetrahedral initialization and parameter annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Physical Dzugutov potential parameters
A_phys = 5.82
B_phys = 1.28
a_phys = 1.87
b_phys = 1.94
c_phys = 1.10
d_phys = 0.27
m_phys = 16.0

# Time management
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0


def _dzugutov_energy(p, c_val=None, d_val=None):
    """Compute Dzugutov energy with optional annealed c, d parameters."""
    if c_val is None:
        c_val = c_phys
    if d_val is None:
        d_val = d_phys
    n = p.shape[0]
    E = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(p[i] - p[j])
            if r < a_phys:
                E += A_phys * (r ** (-m_phys) - B_phys) * np.exp(c_val / (r - a_phys))
            elif r < b_phys:
                E += B_phys * np.exp(d_val / (r - b_phys))
    return E


def _dzugutov_energy_flat(x, c_val=None, d_val=None):
    """Flat energy function for scipy.optimize."""
    n = len(x) // 3
    p = x.reshape(n, 3)
    return _dzugutov_energy(p, c_val, d_val)


def _dzugutov_gradient_flat(x, c_val=None, d_val=None):
    """Flat gradient function for scipy.optimize."""
    if c_val is None:
        c_val = c_phys
    if d_val is None:
        d_val = d_phys
    n = len(x) // 3
    p = x.reshape(n, 3)
    grad = np.zeros_like(x)
    for i in range(n):
        for j in range(i + 1, n):
            diff = p[i] - p[j]
            r = np.linalg.norm(diff)
            if r < 1e-10:
                continue
            if r < a_phys:
                # V(r) = A(r^-m - B) * exp(c/(r-a))
                # dV/dr = A * (-m*r^(-m-1) - B*c/(r-a)^2 * exp(c/(r-a))) ... 
                # Actually: d/dr [A(r^-m - B)exp(c/(r-a))]
                # = A * [(-m*r^(-m-1)) * exp(c/(r-a)) + (r^-m - B) * exp(c/(r-a)) * (-c/(r-a)^2)]
                exp_term = np.exp(c_val / (r - a_phys))
                dV_dr = A_phys * (-m_phys * r ** (-m_phys - 1) - c_val * (r ** (-m_phys) - B_phys) / (r - a_phys) ** 2) * exp_term
            elif r < b_phys:
                # V(r) = B * exp(d/(r-b))
                # dV/dr = B * exp(d/(r-b)) * (-d/(r-b)^2)
                exp_term = np.exp(d_val / (r - b_phys))
                dV_dr = -B_phys * d_val / (r - b_phys) ** 2 * exp_term
            else:
                continue
            
            # Force on atom i: -dV/dr * (p[i]-p[j])/r
            f = -dV_dr / r * diff
            grad[3*i:3*i+3] += f
            grad[3*j:3*j+3] -= f
    return grad


def _icosahedron_vertices(radius=1.0):
    """Generate 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis] * radius
    return verts


def _get_icosahedron_faces_edges():
    """Get faces and edges of unit icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    verts_unit = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts_unit, axis=1)
    verts_unit = verts_unit / norms[:, np.newaxis]
    
    edge_len = 2.0 / np.sqrt(phi * phi + 1)
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            dist = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if abs(dist - edge_len) < 1e-6:
                edges.append((i, j))
    
    edge_set = set(tuple(sorted(e)) for e in edges)
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                if (tuple(sorted((i, j))) in edge_set and
                    tuple(sorted((j, k))) in edge_set and
                    tuple(sorted((i, k))) in edge_set):
                    faces.append((i, j, k))
    
    return verts_unit, faces, edges





def _build_mackay_icosahedron(n, rng):
    """Build a Mackay icosahedral cluster with n atoms."""
    r1 = 1.05
    r2 = 1.85
    
    points = [np.zeros(3)]
    ico = _icosahedron_vertices(r1)
    points.extend(list(ico))
    
    if n <= len(points):
        if n < len(points):
            idx = rng.choice(len(points), n, replace=False)
            points = [points[i] for i in idx]
        return np.array(points)
    
    verts_unit, faces, edges = _get_icosahedron_faces_edges()
    
    face_centers = []
    for face in faces:
        fc = np.mean(verts_unit[list(face)], axis=0)
        fc = fc / np.linalg.norm(fc)
        face_centers.append(fc)
    
    edge_mids = []
    for i, j in edges:
        em = (verts_unit[i] + verts_unit[j]) / 2
        em = em / np.linalg.norm(em)
        edge_mids.append(em)
    
    second_shell = []
    for fc in face_centers:
        second_shell.append(fc * r2)
    for em in edge_mids:
        second_shell.append(em * r2)
    for v in verts_unit:
        second_shell.append(v * (r1 + r2) / 2)
    
    second_shell = np.array(second_shell)
    
    needed = n - len(points)
    if needed <= 0:
        return np.array(points[:n])
    
    n_select = min(needed, len(second_shell))
    idx = rng.permutation(len(second_shell))
    for i in range(n_select):
        points.append(second_shell[idx[i]])
    
    while len(points) < n:
        p = rng.uniform(-r2, r2, 3)
        if np.linalg.norm(p) <= r2:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.55:
                    valid = False
                    break
            if valid:
                points.append(p)
            else:
                points.append(p + 0.15 * rng.standard_normal(3))
        else:
            points.append(p + 0.1 * rng.standard_normal(3))
    
    return np.array(points[:n])


def _build_random_compact(n, rng):
    """Build a random compact cluster with polytetrahedral bias."""
    spacing = 1.15
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    points = []
    attempts = 0
    while len(points) < n and attempts < 15000:
        if rng.random() < 0.3 and len(points) >= 13:
            base_idx = rng.integers(0, len(points))
            base = points[base_idx]
            theta = rng.uniform(0, 2 * np.pi)
            phi_angle = np.arccos(2 * rng.random() - 1)
            offset_dir = np.array([
                np.sin(phi_angle) * np.cos(theta),
                np.sin(phi_angle) * np.sin(theta),
                np.cos(phi_angle)
            ])
            p = base + spacing * offset_dir
        else:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) > radius:
                continue
        
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.55:
                valid = False
                break
        if valid:
            points.append(p)
        attempts += 1
    
    while len(points) < n:
        if points:
            base = points[len(points) % len(points)]
            p = base + 0.35 * rng.standard_normal(3)
        else:
            p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])


def _build_tetrahedral_cluster(n, rng):
    """Build a cluster based on tetrahedral packing (polytetrahedral model)."""
    r = 1.05
    points = []
    
    tet = np.array([
        [1, 1, 1], [1, -1, -1],
        [-1, 1, -1], [-1, -1, 1]
    ], dtype=float)
    norms = np.linalg.norm(tet, axis=1)
    tet = tet / norms[:, np.newaxis] * r
    points.extend(list(tet))
    
    octa = np.array([
        [r * 1.3, 0, 0], [-r * 1.3, 0, 0],
        [0, r * 1.3, 0], [0, -r * 1.3, 0],
        [0, 0, r * 1.3], [0, 0, -r * 1.3]
    ], dtype=float)
    for p in octa:
        if len(points) >= n:
            break
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.55:
                valid = False
                break
        if valid:
            points.append(p)
    
    shell_radius = r * 1.6
    while len(points) < n:
        if rng.random() < 0.4 and len(points) > 4:
            base_idx = rng.integers(0, len(points))
            base = points[base_idx]
            theta = rng.uniform(0, 2 * np.pi)
            phi_angle = np.arccos(2 * rng.random() - 1)
            offset_dir = np.array([
                np.sin(phi_angle) * np.cos(theta),
                np.sin(phi_angle) * np.sin(theta),
                np.cos(phi_angle)
            ])
            p = base + r * 0.9 * offset_dir
        else:
            p = rng.uniform(-shell_radius, shell_radius, 3)
            if np.linalg.norm(p) > shell_radius:
                p = p * shell_radius / np.linalg.norm(p)
        
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.55:
                valid = False
                break
        if valid:
            points.append(p)
        else:
            points.append(p + 0.2 * rng.standard_normal(3))
    
    return np.array(points[:n])


def _anneal_optimize(pos0, c_start, d_start, n_steps=50, max_time=60.0, t0=None):
    """
    Continuous annealing of c and d parameters with gradient descent.
    Starts with large c, d (smooth landscape) and anneals to physical values.
    """
    if t0 is None:
        t0 = time.time()
    
    n = pos0.shape[0]
    x = pos0.flatten().copy()
    best_x = x.copy()
    best_energy = _dzugutov_energy(pos0, c_phys, d_phys)
    
    for step in range(n_steps):
        # Check time
        if time.time() - t0 > max_time:
            break
        
        # Annealing schedule: exponential decay from start to physical
        frac = step / max(n_steps - 1, 1)
        c_val = c_start * np.exp(-3.0 * frac) + c_phys * (1 - np.exp(-3.0 * frac))
        d_val = d_start * np.exp(-3.0 * frac) + d_phys * (1 - np.exp(-3.0 * frac))
        
        # Use L-BFGS-B for local optimization at current c, d
        try:
            res = minimize(
                _dzugutov_energy_flat,
                x,
                jac=_dzugutov_gradient_flat,
                args=(c_val, d_val),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-6,
                         'maxfun': 5000}
            )
            x = res.x
            e_annealed = res.fun
        except Exception:
            continue
        
        # Evaluate at physical parameters
        p = x.reshape(n, 3)
        e_phys = _dzugutov_energy(p, c_phys, d_phys)
        if e_phys < best_energy:
            best_energy = e_phys
            best_x = x.copy()
    
    return best_x.reshape(n, 3), best_energy


def construct(instance):
    """Find low-energy Dzugutov clusters using parameter annealing and polytetrahedral initialization."""
    start_time = time.time()
    
    size = instance["n"]
    
    rng = np.random.default_rng(size * 7 + 13)
    
    best_energy = float('inf')
    best_positions = None
    
    def check_time():
        return (time.time() - start_time) < (TIME_LIMIT - SAFETY_MARGIN)
    
    def try_anneal(pos, c_start=5.0, d_start=2.0, n_steps=40, max_time=50.0):
        nonlocal best_energy, best_positions
        if not check_time():
            return
        try:
            positions, energy = _anneal_optimize(pos, c_start, d_start, n_steps, max_time, start_time)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            pass
    
    # Generate diverse initial structures
    initial_structures = []
    
    # Mackay icosahedral variants
    for i in range(4):
        pos = _build_mackay_icosahedron(size, rng)
        scale = 0.05 + 0.05 * i
        pos = pos + scale * rng.standard_normal((size, 3))
        initial_structures.append(pos)
    
    # Random compact
    for i in range(3):
        pos = _build_random_compact(size, rng)
        pos = pos + 0.05 * rng.standard_normal((size, 3))
        initial_structures.append(pos)
    
    # Tetrahedral
    for i in range(2):
        pos = _build_tetrahedral_cluster(size, rng)
        pos = pos + 0.05 * rng.standard_normal((size, 3))
        initial_structures.append(pos)
    
    # Run annealing on each initial structure
    for idx, pos in enumerate(initial_structures):
        if not check_time():
            break
        # Vary annealing parameters for diversity
        c_start = 4.0 + 2.0 * (idx % 3)
        d_start = 1.5 + 0.5 * (idx % 2)
        n_steps = 35 + 10 * (idx % 2)
        try_anneal(pos, c_start, d_start, n_steps, 45.0)
    
    # If we have a good solution, do refined annealing with tighter parameters
    if best_positions is not None and check_time():
        # Fine annealing: start closer to physical parameters
        for i in range(3):
            if not check_time():
                break
            pos = best_positions + 0.1 * rng.standard_normal((size, 3))
            c_start = 2.0 + 0.5 * i
            d_start = 0.8 + 0.2 * i
            try_anneal(pos, c_start, d_start, 50, 35.0)
    
    # Additional diverse restarts if time permits
    if check_time():
        for i in range(4):
            if not check_time():
                break
            if i % 2 == 0:
                pos = _build_mackay_icosahedron(size, rng)
            else:
                pos = _build_random_compact(size, rng)
            pos = pos + 0.15 * rng.standard_normal((size, 3))
            c_start = 6.0 + i
            d_start = 2.5 + 0.5 * i
            try_anneal(pos, c_start, d_start, 45, 40.0)
    
    # Final refinement around best
    if best_positions is not None and check_time():
        for i in range(2):
            if not check_time():
                break
            pos = best_positions + 0.05 * rng.standard_normal((size, 3))
            try_anneal(pos, 1.5, 0.5, 60, 30.0)
    
    if best_positions is None:
        best_positions = _build_random_compact(size, rng)
    
    return best_positions
# EVOLVE-BLOCK-END
