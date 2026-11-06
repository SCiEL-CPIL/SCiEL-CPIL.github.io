"""Dzugutov cluster global minima: symmetry-adapted manifold + parallel basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_TOTAL_BUDGET = 290.0  # safety margin under 330s kill
_PER_INSTANCE_BUDGET = _TOTAL_BUDGET / 6.0
_THREAD_LOCK = threading.Lock()


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _remaining():
    return max(0.0, _TOTAL_BUDGET - _elapsed())


def _dzugutov_energy(pos):
    """Compute Dzugutov energy vectorized. pos is (n,3)."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    E = 0.0
    mask_a = dist < a
    mask_b = (dist < b) & ~mask_a
    
    if np.any(mask_a):
        r = dist[mask_a]
        E += np.sum(A * (r**(-m) - B) * np.exp(c / (r - a)))
    if np.any(mask_b):
        r = dist[mask_b]
        E += np.sum(B * np.exp(d / (r - b)))
    return E


def _dzugutov_gradient(pos):
    """Compute Dzugutov gradient vectorized. pos is (n,3)."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(pos)
    if n < 2:
        return np.zeros_like(pos)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    grad = np.zeros_like(pos)
    
    mask_a = dist < a
    mask_b = (dist < b) & ~mask_a
    
    if np.any(mask_a):
        r = dist[mask_a]
        # d/dr [A(r^-m - B)exp(c/(r-a))] = A[-m*r^(-m-1)*exp(c/(r-a)) + (r^-m-B)*exp(c/(r-a))*(-c/(r-a)^2)]
        dr = -m * r**(-m-1) - (r**(-m) - B) * c / (r - a)**2
        dV = A * dr * np.exp(c / (r - a))
        # grad_i = -sum_j dV_ij * (pos_i - pos_j)/r_ij
        for i in range(n):
            for j in range(n):
                if i != j and mask_a[i, j]:
                    rij = pos[i] - pos[j]
                    grad[i] -= dV[i, j] * rij / r[i, j]
    
    if np.any(mask_b):
        r = dist[mask_b]
        # d/dr [B*exp(d/(r-b))] = B*exp(d/(r-b))*(-d/(r-b)^2)
        dV = -B * d / (r - b)**2 * np.exp(d / (r - b))
        for i in range(n):
            for j in range(n):
                if i != j and mask_b[i, j]:
                    rij = pos[i] - pos[j]
                    grad[i] -= dV[i, j] * rij / r[i, j]
    
    return grad


def _icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron centered at origin, unit radius."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    return verts


def _mckay_icosahedron(n, rng=None):
    """Generate a Mackay icosahedral cluster with n atoms.
    
    Uses the standard Mackay construction:
    - 1 center
    - 12 vertices of icosahedron (shell 1)
    - 30 edge midpoints (shell 2)  
    - 20 face centers (shell 2)
    - 60 vertices of second shell (shell 3)
    - etc.
    
    Returns (n, 3) array.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    ico = _icosahedron_vertices()
    
    # Generate all icosahedral symmetry directions
    # Use the 60 symmetry operations of the icosahedral group
    # For simplicity, use a set of well-known directions
    
    # Shell radii for Mackay icosahedron (in units of nearest-neighbor distance ~1.1)
    # The standard Mackay clusters have specific shell structures
    
    points = [np.zeros(3)]  # center
    
    # Shell 1: 12 icosahedron vertices at radius r1
    r1 = 1.1
    for v in ico:
        if len(points) < n:
            points.append(r1 * v)
    
    # Shell 2: 30 edge-midpoint directions + 20 face-center directions
    # Edge midpoints: average of pairs of connected vertices
    # Face centers: average of triples forming faces
    
    # For icosahedron, the 30 edges connect specific vertex pairs
    # The 20 faces are triangles
    
    # Generate edge-midpoint directions
    edge_dirs = []
    # Icosahedron edges: each vertex connects to 5 others
    # Standard icosahedron connectivity (0-indexed)
    edges = [
        (0,1),(0,3),(0,4),(0,5),(0,8),(0,10),
        (1,2),(1,4),(1,5),(1,9),(1,11),
        (2,3),(2,6),(2,7),(2,9),(2,11),
        (3,4),(3,6),(3,7),(3,8),(3,10),
        (4,5),(4,6),(4,7),(4,9),(4,10),
        (5,6),(5,7),(5,8),(5,9),(5,11),
        (6,7),(6,8),(6,9),(6,10),(6,11),
        (7,8),(7,9),(7,10),(7,11),
        (8,9),(8,10),(8,11),
        (9,10),(9,11),
        (10,11)
    ]
    
    # Deduplicate edges
    seen_edges = set()
    unique_edges = []
    for e in edges:
        key = tuple(sorted(e))
        if key not in seen_edges:
            seen_edges.add(key)
            unique_edges.append(key)
    
    for i, j in unique_edges:
        if len(points) < n:
            mid = (ico[i] + ico[j]) / 2.0
            mid /= np.linalg.norm(mid)
            r2 = 1.1 * np.sqrt(3.0) / 2.0 * 2  # approximate
            points.append(r2 * mid)
    
    # Face centers
    # Icosahedron has 20 triangular faces
    faces = [
        (0,1,4),(0,1,5),(0,3,4),(0,3,5),(0,4,8),(0,5,9),
        (0,8,10),(0,9,11),
        (1,2,4),(1,2,5),(1,4,9),(1,5,10),
        (1,9,11),(1,10,11),
        (2,3,6),(2,3,7),(2,6,9),(2,7,10),
        (2,9,11),(2,10,11),
        (3,4,6),(3,4,7),(3,6,8),(3,7,10),
        (3,8,11),(3,10,11),
        (4,5,6),(4,5,7),(4,6,9),(4,7,10),
        (4,9,11),(4,10,11),
        (5,6,8),(5,6,9),(5,7,10),(5,8,11),(5,9,11),
        (6,7,8),(6,7,10),(6,8,9),(6,10,11),
        (7,8,9),(7,9,11),(7,10,11),
        (8,9,10),(8,10,11),(9,10,11)
    ]
    
    # Deduplicate faces
    seen_faces = set()
    unique_faces = []
    for f in faces:
        key = tuple(sorted(f))
        if key not in seen_faces:
            seen_faces.add(key)
            unique_faces.append(key)
    
    for i, j, k in unique_faces:
        if len(points) < n:
            fc = (ico[i] + ico[j] + ico[k]) / 3.0
            fc /= np.linalg.norm(fc)
            r2b = 1.1 * np.sqrt(3.0) / 3.0 * 2  # approximate
            points.append(r2b * fc)
    
    # Shell 3: 60 vertices of the second icosahedral shell
    # These are at the vertices of a larger icosahedron or other high-symmetry points
    if len(points) < n:
        r3 = 1.1 * np.sqrt(5.0)
        # Use rotated icosahedron vertices
        for ang in [0, np.pi/5, 2*np.pi/5, 3*np.pi/5, 4*np.pi/5]:
            for ax in range(3):
                for v in ico:
                    if len(points) >= n:
                        break
                    vv = v.copy()
                    if ax == 0:
                        vv[1], vv[2] = vv[1]*np.cos(ang) - vv[2]*np.sin(ang), vv[1]*np.sin(ang) + vv[2]*np.cos(ang)
                    elif ax == 1:
                        vv[0], vv[2] = vv[0]*np.cos(ang) - vv[2]*np.sin(ang), vv[0]*np.sin(ang) + vv[2]*np.cos(ang)
                    else:
                        vv[0], vv[1] = vv[0]*np.cos(ang) - vv[1]*np.sin(ang), vv[0]*np.sin(ang) + vv[1]*np.cos(ang)
                    vv /= np.linalg.norm(vv)
                    # Check if this direction is already close to existing points
                    is_new = True
                    for p in points:
                        if np.dot(p, vv) > 0.99:
                            is_new = False
                            break
                    if is_new:
                        points.append(r3 * vv)
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
    
    # Fill remaining with random compact positions
    while len(points) < n:
        r_max = 1.1 * (n ** (1/3)) * 0.9
        p = rng.uniform(-r_max, r_max, 3)
        if np.linalg.norm(p) <= r_max:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.55:
                    valid = False
                    break
            if valid:
                points.append(p)
            else:
                base = points[rng.integers(len(points))]
                p = base + rng.normal(0, 0.3, 3)
                points.append(p)
        else:
            base = points[rng.integers(len(points))]
            p = base + rng.normal(0, 0.3, 3)
            points.append(p)
    
    return np.array(points[:n])


def _symmetry_manifold_energy(params, n):
    """Energy on a symmetry-adapted manifold.
    
    params = [r1, r2, r3, twist1, twist2, ...]
    where r1, r2, r3 are shell radii and twists are angular offsets.
    """
    # Decode parameters into positions
    pos = _manifold_to_positions(params, n)
    return _dzugutov_energy(pos)


def _manifold_to_positions(params, n):
    """Convert manifold parameters to Cartesian positions.
    
    params: [r1, r2, r3, t1, t2, t3]
    r1, r2, r3: shell radii
    t1, t2, t3: twist angles for each shell
    """
    ico = _icosahedron_vertices()
    
    r1 = params[0]
    r2 = params[1]
    r3 = params[2]
    t1 = params[3]
    t2 = params[4]
    t3 = params[5]
    
    points = [np.zeros(3)]  # center
    
    # Shell 1: 12 vertices with twist t1
    for v in ico:
        if len(points) < n:
            vv = v.copy()
            # Apply twist around z-axis
            vv[0], vv[1] = vv[0]*np.cos(t1) - vv[1]*np.sin(t1), vv[0]*np.sin(t1) + vv[1]*np.cos(t1)
            points.append(r1 * vv)
    
    # Shell 2: 30 edge midpoints with twist t2
    edges = [
        (0,1),(0,3),(0,4),(0,5),(0,8),(0,10),
        (1,2),(1,4),(1,5),(1,9),(1,11),
        (2,3),(2,6),(2,7),(2,9),(2,11),
        (3,4),(3,6),(3,7),(3,8),(3,10),
        (4,5),(4,6),(4,7),(4,9),(4,10),
        (5,6),(5,7),(5,8),(5,9),(5,11),
        (6,7),(6,8),(6,9),(6,10),(6,11),
        (7,8),(7,9),(7,10),(7,11),
        (8,9),(8,10),(8,11),
        (9,10),(9,11),
        (10,11)
    ]
    seen_edges = set()
    for i, j in edges:
        key = tuple(sorted((i, j)))
        if key in seen_edges or len(points) >= n:
            continue
        seen_edges.add(key)
        mid = (ico[i] + ico[j]) / 2.0
        mid /= np.linalg.norm(mid)
        mid[0], mid[1] = mid[0]*np.cos(t2) - mid[1]*np.sin(t2), mid[0]*np.sin(t2) + mid[1]*np.cos(t2)
        points.append(r2 * mid)
    
    # Shell 3: 20 face centers with twist t3
    faces = [
        (0,1,4),(0,1,5),(0,3,4),(0,3,5),(0,4,8),(0,5,9),
        (0,8,10),(0,9,11),
        (1,2,4),(1,2,5),(1,4,9),(1,5,10),
        (1,9,11),(1,10,11),
        (2,3,6),(2,3,7),(2,6,9),(2,7,10),
        (2,9,11),(2,10,11),
        (3,4,6),(3,4,7),(3,6,8),(3,7,10),
        (3,8,11),(3,10,11),
        (4,5,6),(4,5,7),(4,6,9),(4,7,10),
        (4,9,11),(4,10,11),
        (5,6,8),(5,6,9),(5,7,10),(5,8,11),(5,9,11),
        (6,7,8),(6,7,10),(6,8,9),(6,10,11),
        (7,8,9),(7,9,11),(7,10,11),
        (8,9,10),(8,10,11),(9,10,11)
    ]
    seen_faces = set()
    for i, j, k in faces:
        key = tuple(sorted((i, j, k)))
        if key in seen_faces or len(points) >= n:
            continue
        seen_faces.add(key)
        fc = (ico[i] + ico[j] + ico[k]) / 3.0
        fc /= np.linalg.norm(fc)
        fc[0], fc[1] = fc[0]*np.cos(t3) - fc[1]*np.sin(t3), fc[0]*np.sin(t3) + fc[1]*np.cos(t3)
        points.append(r3 * fc)
    
    # Fill remaining
    rng = np.random.default_rng(12345)
    while len(points) < n:
        r_max = 1.1 * (n ** (1/3)) * 0.9
        p = rng.uniform(-r_max, r_max, 3)
        if np.linalg.norm(p) <= r_max:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.55:
                    valid = False
                    break
            if valid:
                points.append(p)
            else:
                base = points[rng.integers(len(points))]
                p = base + rng.normal(0, 0.3, 3)
                points.append(p)
        else:
            base = points[rng.integers(len(points))]
            p = base + rng.normal(0, 0.3, 3)
            points.append(p)
    
    return np.array(points[:n])


def _manifold_optimize(n, rng):
    """Optimize on the symmetry manifold using L-BFGS-B."""
    # Initial guess: standard Mackay radii
    x0 = np.array([1.1, 1.1*np.sqrt(3.0)/2.0*2, 1.1*np.sqrt(3.0)/3.0*2, 0.0, 0.0, 0.0])
    
    def obj(x):
        if _remaining() < 2.0:
            return 1e10
        pos = _manifold_to_positions(x, n)
        return _dzugutov_energy(pos)
    
    try:
        result = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': 50, 'maxls': 20})
        if result.success or result.fun < 1e10:
            pos = _manifold_to_positions(result.x, n)
            return result.fun, pos
    except Exception:
        pass
    
    return np.inf, _manifold_to_positions(x0, n)


def _basin_hopping(pos, n, potential, parameter, rng, max_iters=20, kick_scale=0.4):
    """Basin hopping: kick, relax, accept if lower."""
    best_energy = np.inf
    best_pos = pos.copy()
    current_energy = np.inf
    current_pos = pos.copy()
    
    for it in range(max_iters):
        if _remaining() < 3.0:
            break
        
        # Kick
        kick = rng.normal(0, kick_scale, current_pos.shape)
        kicked = current_pos + kick
        
        # Relax
        try:
            energy, relaxed = relax(kicked, potential, parameter)
        except Exception:
            continue
        
        # Metropolis acceptance (but we only want to keep improving)
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
        
        # Accept if lower or with probability exp(-delta/T)
        delta = energy - current_energy
        if delta < 0 or rng.random() < np.exp(-delta / 0.3):
            current_energy = energy
            current_pos = relaxed.copy()
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy Dzugutov cluster using symmetry manifold + parallel basin hopping."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(n * 1000 + 42)
    
    best_energy = np.inf
    best_pos = None
    
    # Phase 1: Symmetry manifold optimization
    manifold_energy, manifold_pos = _manifold_optimize(n, rng)
    if manifold_energy < best_energy:
        best_energy = manifold_energy
        best_pos = manifold_pos.copy()
    
    # Phase 2: Generate diverse seeds
    seeds = []
    
    # 1. Mackay icosahedron
    seeds.append(_mckay_icosahedron(n, rng))
    
    # 2. Deformed Mackay
    mackay = _mckay_icosahedron(n, np.random.default_rng(n * 2000 + 7))
    mackay += rng.normal(0, 0.15, mackay.shape)
    seeds.append(mackay)
    
    # 3. Compressed Mackay
    mackay2 = _mckay_icosahedron(n, np.random.default_rng(n * 3000 + 11))
    seeds.append(mackay2 * 0.92)
    
    # 4. Expanded Mackay
    mackay3 = _mckay_icosahedron(n, np.random.default_rng(n * 4000 + 13))
    seeds.append(mackay3 * 1.08)
    
    # 5. Random compact
    radius = 1.1 * n ** (1/3) * 0.85
    points = []
    for _ in range(n * 500):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.55:
                    valid = False
                    break
            if valid:
                points.append(p)
    while len(points) < n:
        base = points[rng.integers(len(points))]
        p = base + rng.normal(0, 0.4, 3)
        points.append(p)
    seeds.append(np.array(points[:n]))
    
    # 6. Another random compact
    rng2 = np.random.default_rng(n * 777 + 13)
    points2 = []
    for _ in range(n * 500):
        if len(points2) >= n:
            break
        p = rng2.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points2:
                if np.linalg.norm(p - q) < 0.55:
                    valid = False
                    break
            if valid:
                points2.append(p)
    while len(points2) < n:
        base = points2[rng2.integers(len(points2))]
        p = base + rng2.normal(0, 0.4, 3)
        points2.append(p)
    seeds.append(np.array(points2[:n]))
    
    # 7. Manifold result as seed
    if manifold_pos is not None:
        seeds.append(manifold_pos.copy())
    
    # 8. Perturbed manifold
    if manifold_pos is not None:
        seeds.append(manifold_pos + rng.normal(0, 0.2, manifold_pos.shape))
    
    # Parallel basin hopping on seeds
    def _process_seed(seed_idx):
        nonlocal best_energy, best_pos
        seed = seeds[seed_idx].copy()
        
        local_best_energy = np.inf
        local_best_pos = seed.copy()
        
        # Try multiple kick scales
        for ks in [0.3, 0.5, 0.7]:
            if _remaining() < 3.0:
                break
            e, p = _basin_hopping(seed, n, potential, parameter,
                                  np.random.default_rng(seed_idx * 1000 + int(ks*100)),
                                  max_iters=10, kick_scale=ks)
            if e < local_best_energy:
                local_best_energy = e
                local_best_pos = p
        
        with _THREAD_LOCK:
            if local_best_energy < best_energy:
                best_energy = local_best_energy
                best_pos = local_best_pos.copy()
        
        return local_best_energy, local_best_pos
    
    n_workers = min(4, len(seeds))
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i in range(n_workers):
            if _remaining() < 5.0:
                break
            futures.append(executor.submit(_process_seed, i))
        
        for fut in as_completed(futures):
            try:
                fut.result(timeout=40)
            except Exception:
                pass
    
    # Phase 3: Final basin hopping on best result
    if _remaining() > 10 and best_pos is not None:
        rng_bh = np.random.default_rng(n * 31337)
        current_pos = best_pos.copy()
        current_energy = best_energy
        
        for bh_iter in range(20):
            if _remaining() < 5.0:
                break
            
            kick = rng_bh.normal(0, 0.4, current_pos.shape)
            kicked = current_pos + kick
            
            try:
                e, relaxed = relax(kicked, potential, parameter)
            except Exception:
                continue
            
            if e < current_energy:
                current_energy = e
                current_pos = relaxed.copy()
                with _THREAD_LOCK:
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_pos = current_pos.copy()
    
    # Final validation: ensure minimum distance constraint
    if best_pos is None:
        best_pos = _mckay_icosahedron(n, rng)
    
    pos = best_pos.copy()
    for _ in range(100):
        min_dist = np.inf
        min_pair = (0, 1)
        for i in range(len(pos)):
            for j in range(i+1, len(pos)):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    min_dist = d
                    min_pair = (i, j)
        if min_dist >= 0.5:
            break
        i, j = min_pair
        direction = pos[i] - pos[j]
        if np.linalg.norm(direction) < 1e-10:
            direction = rng.normal(0, 1, 3)
        direction /= np.linalg.norm(direction)
        pos[i] += 0.01 * direction
        pos[j] -= 0.01 * direction
    
    return pos
# EVOLVE-BLOCK-END
