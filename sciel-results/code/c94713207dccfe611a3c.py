"""Structure prediction strategy: embed known small-cluster minima as cores, grow by adding atoms."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
from scipy.optimize import minimize

def _icosahedron_13(spacing=1.0):
    """Return 13-atom icosahedral cluster (1 center + 12 vertices)."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit radius
    verts = verts / np.linalg.norm(verts[0])
    # Scale to desired spacing
    r = spacing * 0.85
    shell = verts * r
    center = np.array([[0.0, 0.0, 0.0]])
    return np.vstack([center, shell])

def _icosahedron_14(spacing=1.0):
    """14-atom: 13-atom icosahedron + 1 atom at a face center."""
    base = _icosahedron_13(spacing)
    # Find a triangular face and place atom at its center
    # Face between vertices 0, 1, 8 (indices in the 12-vertex shell)
    v0, v1, v8 = base[1], base[2], base[9]
    face_center = (v0 + v1 + v8) / 3.0
    # Push outward slightly
    norm = np.linalg.norm(face_center)
    if norm > 1e-10:
        face_center = face_center / norm * (norm + spacing * 0.15)
    return np.vstack([base, face_center.reshape(1, 3)])

def _icosahedron_15(spacing=1.0):
    """15-atom: 13-atom icosahedron + 2 atoms at opposite face centers."""
    base = _icosahedron_13(spacing)
    v0, v1, v8 = base[1], base[2], base[9]
    face_center1 = (v0 + v1 + v8) / 3.0
    norm1 = np.linalg.norm(face_center1)
    if norm1 > 1e-10:
        face_center1 = face_center1 / norm1 * (norm1 + spacing * 0.15)
    
    v5, v6, v10 = base[6], base[7], base[11]
    face_center2 = (v5 + v6 + v10) / 3.0
    norm2 = np.linalg.norm(face_center2)
    if norm2 > 1e-10:
        face_center2 = face_center2 / norm2 * (norm2 + spacing * 0.15)
    
    return np.vstack([base, face_center1.reshape(1, 3), face_center2.reshape(1, 3)])

def _mackay_icosahedron(n, spacing=1.0):
    """
    Generate a Mackay icosahedral cluster with n atoms.
    Uses nested shells with icosahedral symmetry.
    Shell 1: 1 center + 12 vertices = 13
    Shell 2: 30 face centers = 43 total
    Shell 3: 20 edge midpoints + 120 ... = more complex
    For simplicity, use 13, 14, 15, then grow by adding atoms on a sphere.
    """
    if n <= 13:
        base = _icosahedron_13(spacing)
        return base[:n]
    elif n == 14:
        return _icosahedron_14(spacing)
    elif n == 15:
        return _icosahedron_15(spacing)
    else:
        # Start with 13-atom core and add atoms on a spherical shell
        core = _icosahedron_13(spacing)
        n_add = n - 13
        # Place additional atoms on a sphere of radius ~1.5 * core radius
        core_radius = spacing * 0.85
        shell_radius = core_radius * 1.6
        
        # Use Fibonacci sphere for uniform distribution
        shell_pts = []
        for i in range(n_add):
            z = 1 - 2 * (i + 0.5) / n_add
            r = np.sqrt(1 - z * z)
            theta = 2 * np.pi * i / n_add
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * shell_radius
            shell_pts.append(p)
        
        return np.vstack([core, np.array(shell_pts)])

def _fcc_cluster(n, spacing=1.0):
    """Generate an FCC-like cluster with n atoms."""
    a = spacing * 0.8
    # FCC lattice points
    points = []
    max_coord = int(np.ceil((n ** (1/3)) * a * 0.7)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([i * a + dx * a, j * a + dy * a, k * a + dz * a])
                    if np.linalg.norm(p) <= max_coord * a * 0.85:
                        points.append(p)
    points = np.array(points)
    if len(points) < n:
        # If not enough, just use random
        radius = a * n ** (1/3)
        points = np.random.uniform(-radius, radius, (n, 3))
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:n]]

def _bcp_cluster(n, spacing=1.0):
    """Generate a body-centered cubic cluster."""
    a = spacing * 0.9
    points = []
    max_coord = int(np.ceil((n ** (1/3)) * a * 0.6)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                p = np.array([i * a, j * a, k * a])
                if np.linalg.norm(p) <= max_coord * a * 0.85:
                    points.append(p)
                # BCC center
                pc = np.array([i * a + a/2, j * a + a/2, k * a + a/2])
                if np.linalg.norm(pc) <= max_coord * a * 0.85:
                    points.append(pc)
    points = np.array(points)
    if len(points) < n:
        radius = a * n ** (1/3)
        points = np.random.uniform(-radius, radius, (n, 3))
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:n]]

def _grow_from_core(core_positions, n_total, spacing=1.0, rng=None):
    """
    Grow a cluster from a core by adding atoms at low-energy positions.
    Uses a greedy approach: add one atom at a time at the position that minimizes energy.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    current = core_positions.copy()
    n_current = len(current)
    
    if n_current >= n_total:
        return current[:n_total]
    
    # Determine the radius of the core
    core_radius = np.max(np.linalg.norm(current, axis=1)) if n_current > 0 else 0.5
    
    # Generate candidate positions on a spherical shell and in interstitial sites
    n_to_add = n_total - n_current
    
    # Strategy: add atoms at positions that are local minima of the pair potential
    # with respect to existing atoms
    for _ in range(n_to_add):
        # Generate candidate positions
        candidates = []
        
        # 1. Positions on a sphere around the core
        n_shell = 20
        shell_radius = core_radius * 1.3
        for i in range(n_shell):
            z = 1 - 2 * (i + 0.5) / n_shell
            r = np.sqrt(1 - z * z)
            theta = 2 * np.pi * i / n_shell
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * shell_radius
            candidates.append(p)
        
        # 2. Random positions in a box
        box_radius = core_radius * 2.0
        for _ in range(15):
            p = rng.uniform(-box_radius, box_radius, 3)
            if np.linalg.norm(p) <= box_radius:
                candidates.append(p)
        
        # 3. Positions near existing atoms (interstitial)
        for i in range(min(n_current, 10)):
            direction = rng.normal(size=3)
            direction = direction / np.linalg.norm(direction)
            p = current[i] + direction * spacing * 0.6
            candidates.append(p)
        
        candidates = np.array(candidates)
        
        # Evaluate energy contribution of adding each candidate
        best_energy = np.inf
        best_pos = None
        
        for cand in candidates:
            if not np.all(np.isfinite(cand)):
                continue
            # Check minimum distance to existing atoms
            dists = np.linalg.norm(current - cand, axis=1)
            if np.min(dists) < 0.35:
                continue
            
            # Compute pair energy contribution
            # E_pair = 1/2 * sum_j [r^-n - c * sqrt(r^-m)]
            # For simplicity, use the full energy gradient
            test_config = np.vstack([current, cand.reshape(1, 3)])
            try:
                energy, _ = sutton_chen_energy_gradient(test_config, 'sutton_chen')
                if np.isfinite(energy) and energy < best_energy:
                    best_energy = energy
                    best_pos = cand
            except:
                continue
        
        if best_pos is None:
            # Fallback: add at a random position
            p = rng.uniform(-core_radius * 1.5, core_radius * 1.5, 3)
            if np.linalg.norm(p) <= core_radius * 1.5:
                dists = np.linalg.norm(current - p, axis=1)
                if np.min(dists) >= 0.35:
                    best_pos = p
        
        if best_pos is not None:
            current = np.vstack([current, best_pos.reshape(1, 3)])
            core_radius = np.max(np.linalg.norm(current, axis=1))
    
    return current

def _generate_initial_positions(n, kind, rng):
    """Generate multiple diverse initial configurations using hierarchical growth."""
    configs = []
    
    # Strategy 1: Icosahedral core + growth
    try:
        if n <= 15:
            if n == 13:
                cfg = _icosahedron_13(spacing=0.75)
            elif n == 14:
                cfg = _icosahedron_14(spacing=0.75)
            elif n == 15:
                cfg = _icosahedron_15(spacing=0.75)
            else:
                cfg = _icosahedron_13(spacing=0.75)[:n]
            if len(cfg) == n and np.all(np.isfinite(cfg)):
                configs.append(cfg)
        else:
            core = _icosahedron_13(spacing=0.75)
            cfg = _grow_from_core(core, n, spacing=0.75, rng=rng)
            if len(cfg) == n and np.all(np.isfinite(cfg)):
                configs.append(cfg)
    except:
        pass
    
    # Strategy 2: FCC core + growth
    try:
        if n <= 20:
            cfg = _fcc_cluster(n, spacing=0.75)
            if len(cfg) == n and np.all(np.isfinite(cfg)):
                configs.append(cfg)
        else:
            core = _fcc_cluster(14, spacing=0.75)
            cfg = _grow_from_core(core, n, spacing=0.75, rng=rng)
            if len(cfg) == n and np.all(np.isfinite(cfg)):
                configs.append(cfg)
    except:
        pass
    
    # Strategy 3: BCC core + growth
    try:
        if n <= 20:
            cfg = _bcp_cluster(n, spacing=0.8)
            if len(cfg) == n and np.all(np.isfinite(cfg)):
                configs.append(cfg)
        else:
            core = _bcp_cluster(14, spacing=0.8)
            cfg = _grow_from_core(core, n, spacing=0.8, rng=rng)
            if len(cfg) == n and np.all(np.isfinite(cfg)):
                configs.append(cfg)
    except:
        pass
    
    # Strategy 4: Random compact with minimum distance
    radius = 0.75 * n ** (1.0 / 3.0) * 1.2
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.4 for q in points):
                points.append(p)
        attempts += 1
    if len(points) == n:
        configs.append(np.array(points))
    
    # Strategy 5: Spherical shell + icosahedral core
    if n >= 13:
        core_n = min(13, n)
        shell_n = n - core_n
        core = _icosahedron_13(spacing=0.7)[:core_n]
        shell_r = 0.7 * (n ** (1/3.0)) * 1.0
        shell_pts = []
        for i in range(shell_n):
            z = 1 - 2 * (i + 0.5) / shell_n
            r = np.sqrt(1 - z * z)
            theta = 2 * np.pi * i / shell_n
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * shell_r
            shell_pts.append(p)
        config5 = np.vstack([core, np.array(shell_pts)])
        if len(config5) == n and np.all(np.isfinite(config5)):
            configs.append(config5)
    
    return configs

def construct(instance):
    """Best of multiple structure-prediction initializations, each relaxed locally."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    rng = np.random.default_rng(size * 7 + 123)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse initial configurations
    configs = _generate_initial_positions(size, kind, rng)
    
    # Add a few more random configurations with different parameters
    for trial in range(3):
        radius = 0.73 * size ** (1.0 / 3.0) * (1.0 + 0.2 * trial)
        points = []
        attempts = 0
        while len(points) < size and attempts < size * 150:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.36 for q in points):
                    points.append(p)
            attempts += 1
        if len(points) == size:
            configs.append(np.array(points))
    
    # Relax each configuration
    for config in configs:
        if config is None or len(config) != size:
            continue
        if not np.all(np.isfinite(config)):
            continue
        # Check minimum distance
        min_dist = np.inf
        for i in range(size):
            for j in range(i + 1, size):
                d = np.linalg.norm(config[i] - config[j])
                if d < min_dist:
                    min_dist = d
        if min_dist < 0.35:
            continue
        
        try:
            energy, positions = relax(config, potential, kind)
            if np.isfinite(energy) and energy < best_energy:
                # Verify validity
                if np.all(np.isfinite(positions)):
                    min_d = np.inf
                    for i in range(size):
                        for j in range(i + 1, size):
                            d = np.linalg.norm(positions[i] - positions[j])
                            if d < min_d:
                                min_d = d
                    if min_d >= 0.35:
                        best_energy = energy
                        best_positions = positions
        except:
            pass
    
    # If no valid result, fall back to a simple random relaxation
    if best_positions is None:
        radius = 0.75 * size ** (1.0 / 3.0)
        points = []
        attempts = 0
        while len(points) < size and attempts < size * 200:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.4 for q in points):
                    points.append(p)
            attempts += 1
        if len(points) == size:
            try:
                energy, positions = relax(np.array(points), potential, kind)
                if np.all(np.isfinite(positions)):
                    best_positions = positions
            except:
                pass
    
    # Final fallback
    if best_positions is None:
        # Create a simple linear arrangement
        positions = np.zeros((size, 3))
        for i in range(size):
            positions[i] = [i * 0.5, 0, 0]
        best_positions = positions
    
    return best_positions
# EVOLVE-BLOCK-END
