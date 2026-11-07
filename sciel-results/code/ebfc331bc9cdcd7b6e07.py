"""Sutton-Chen transition-metal cluster global minima: model-and-optimize approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Known reference energies (approximate, from Doye & Wales 1998 literature values)
# These are used as soft guides; the actual search is what matters.
# For each (kind, n) we try multiple strategies and keep the best.

# Deadline guard
_START_TIME = None
_TIME_LIMIT = 310.0  # Leave 20s safety margin before 330s kill

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _make_fcc(n, spacing=0.72):
    """Generate n atoms in an FCC-like arrangement."""
    # FCC lattice vectors
    a = spacing
    # Generate FCC positions in a sphere of appropriate radius
    # FCC unit cell: 4 atoms at (0,0,0), (0,a/2,a/2), (a/2,0,a/2), (a/2,a/2,0)
    positions = []
    # Use a simple approach: place atoms on FCC lattice points within a radius
    r_max = spacing * n ** (1.0/3.0) * 0.8
    # FCC lattice: points at (i+j+k)*a/2 where i+j+k even, plus basis
    max_idx = int(r_max / (a * 0.5)) + 1
    candidates = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # FCC: (i+j+k) must be even for the primary sublattice
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * a * 0.5
                    if np.linalg.norm(p) <= r_max:
                        candidates.append(p)
                else:
                    # Secondary FCC positions
                    p = (np.array([i, j, k]) + np.array([1, 1, 1])) * a * 0.25
                    if np.linalg.norm(p) <= r_max:
                        candidates.append(p)
    # Sort by distance from center and take n
    candidates = np.array(candidates)
    if len(candidates) == 0:
        return None
    d2 = np.sum(candidates**2, axis=1)
    order = np.argsort(d2)
    selected = candidates[order[:n]]
    return selected

def _make_bcc(n, spacing=0.72):
    """Generate n atoms in a BCC-like arrangement."""
    a = spacing
    r_max = spacing * n ** (1.0/3.0) * 0.8
    max_idx = int(r_max / (a * 0.5)) + 1
    candidates = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # BCC: corners and body center
                p_corner = np.array([i, j, k]) * a
                if np.linalg.norm(p_corner) <= r_max:
                    candidates.append(p_corner)
                p_body = (np.array([i, j, k]) + np.array([0.5, 0.5, 0.5])) * a
                if np.linalg.norm(p_body) <= r_max:
                    candidates.append(p_body)
    candidates = np.array(candidates)
    if len(candidates) == 0:
        return None
    d2 = np.sum(candidates**2, axis=1)
    order = np.argsort(d2)
    selected = candidates[order[:n]]
    return selected

def _make_mackay_icosahedral(n, spacing=0.72):
    """Generate n atoms in a Mackay icosahedral arrangement."""
    # Mackay icosahedra have shells of size 1, 12, 42, 120, 220, ...
    # We generate points on these shells and select the closest n to the center.
    
    # Icosahedral vertices (normalized)
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
        [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
        [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    
    # Generate points for several shells
    # Shell 0: 1 atom at center
    # Shell 1: 12 atoms at vertices
    # Shell 2: 42 atoms (12 vertices + 30 edge midpoints)
    # Shell 3: 120 atoms (12 vertices + 30 edges + 60 face centers + 120 new?) 
    # Actually, let's just generate points on spheres of increasing radius 
    # with icosahedral symmetry by rotating the 12 vertices.
    
    # To get icosahedral symmetry, we can take the 12 vertices and their rotations.
    # However, a simpler robust method for "Mackay-like" is to use the 
    # Fibonacci sphere but bias it towards icosahedral directions, 
    # or just use multiple shells of Fibonacci spheres which is effective.
    
    # Let's use a hybrid: 
    # 1. Center atom
    # 2. 12 vertices at r1
    # 3. 30 edge midpoints at r2
    # 4. 60 face centers at r3
    # 5. Fill remaining with Fibonacci sphere on larger shells.
    
    points = [np.zeros(3)] # Center
    
    r1 = spacing * 0.6
    points.extend(verts * r1)
    
    # Edge midpoints
    # 30 edges in icosahedron. We can find them by checking distances between vertices.
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if np.linalg.norm(verts[i] - verts[j]) < 1.0: # Approximate threshold for unit ico edges
                edges.append((i, j))
    # There are 30 edges.
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid /= np.linalg.norm(mid)
        r2 = r1 * np.sqrt(2) # Approximate radius for edge midpoints in ico
        # Actually, for regular ico, edge length a, vertex radius R, mid radius is R/sqrt(2)? 
        # Let's just use a slightly larger radius.
        r_edge = r1 * 1.15
        points.append(mid * r_edge)
        
    # Face centers
    # 20 faces.
    # We can approximate by taking cross products of adjacent edges or just use a 3rd shell.
    # Let's add a 3rd shell of 60 points (or similar) using a denser icosahedral pattern.
    # For simplicity and robustness, let's just generate a "dense icosahedral" set 
    # by taking the 12 vertices and applying all 60 rotations of the icosahedral group?
    # That's complex. 
    # Alternative: Use Fibonacci sphere for the rest, but start with the 12 vertices.
    
    if n <= 13:
        return np.array(points[:n])
    
    # Add more points using Fibonacci sphere on a larger radius
    remaining = n - len(points)
    r3 = spacing * 1.2
    golden = (1 + np.sqrt(5)) / 2
    for i in range(remaining):
        y = 1 - (i + 0.5) / remaining * 2
        r = np.sqrt(1 - y*y)
        theta = golden * i * 2 * np.pi
        p = np.array([r * np.cos(theta), y, r * np.sin(theta)]) * r3
        points.append(p)
        
    return np.array(points[:n])

def _make_spherical_shells(n, spacing=0.72, seed=None):
    """Generate n atoms on concentric spherical shells."""
    rng = np.random.default_rng(seed)
    points = [np.zeros(3)]
    remaining = n - 1
    
    # Estimate radii
    # Volume ~ n * (spacing/2)^3
    # R_total ~ spacing * (n/4)^0.5 ?
    # Let's use a few shells.
    # Shell 1: 12 atoms
    # Shell 2: 30 atoms
    # Shell 3: 60 atoms
    # etc.
    
    # For simplicity, just place atoms on 2-3 shells.
    if n == 1:
        return np.array(points)
        
    r1 = spacing * 0.7
    r2 = spacing * 1.3
    r3 = spacing * 1.9
    
    # Determine how many on each shell
    # Heuristic: distribute roughly proportional to surface area (r^2)
    weights = [r1**2, r2**2, r3**2]
    total_weight = sum(weights)
    counts = [int(remaining * w / total_weight) for w in weights]
    # Adjust to sum to remaining
    diff = remaining - sum(counts)
    counts[-1] += diff
    
    all_points = points
    
    for r, count in zip([r1, r2, r3], counts):
        if count <= 0:
            continue
        # Random points on sphere
        # Use Gaussian distribution normalized
        pts = rng.normal(0, 1, (count, 3))
        norms = np.linalg.norm(pts, axis=1)
        pts /= norms[:, np.newaxis]
        pts *= r
        all_points.extend(pts)
        
    return np.array(all_points[:n])

def _make_random_compact(n, spacing=0.72, seed=None, min_dist=0.5):
    """Random compact cluster with minimum distance constraint (vectorized)."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    max_attempts = 20000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(points) == 0:
                points.append(p)
            else:
                arr = np.array(points)
                dists = np.linalg.norm(arr - p, axis=1)
                if np.min(dists) >= min_dist:
                    points.append(p)
        attempts += 1
    if len(points) < n:
        # Fallback: allow closer packing
        points = []
        attempts = 0
        while len(points) < n and attempts < max_attempts:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if len(points) == 0:
                    points.append(p)
                else:
                    arr = np.array(points)
                    dists = np.linalg.norm(arr - p, axis=1)
                    if np.min(dists) >= 0.36:
                        points.append(p)
            attempts += 1
        if len(points) < n:
            # Last resort: just fill randomly
            while len(points) < n:
                p = rng.uniform(-radius, radius, 3)
                points.append(p)
    return np.array(points)

def _make_deformed(n, base_positions, spacing=0.72, seed=None):
    """Create a deformed version of base positions (vectorized collision fix)."""
    rng = np.random.default_rng(seed)
    positions = base_positions.copy()
    # Add small random perturbations
    noise = rng.normal(0, 0.05 * spacing, positions.shape)
    positions += noise
    # Ensure no atoms are too close (vectorized)
    n_pts = len(positions)
    for _ in range(3):  # Iterate a few times to resolve collisions
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, np.inf)
        # Find pairs too close
        mask = dists < 0.36
        if not np.any(mask):
            break
        for i in range(n_pts):
            close = np.where(mask[i])[0]
            for j in close:
                if j <= i:
                    continue
                d = dists[i, j]
                if d > 0:
                    direction = (positions[i] - positions[j]) / d
                    push = (0.36 - d) / 2 + 0.01
                    positions[i] += direction * push
                    positions[j] -= direction * push
    return positions

def construct(instance):
    """
    Construct a low-energy Sutton-Chen cluster using model-and-optimize approach.
    
    Strategy:
    1. Generate multiple diverse initial structures (FCC, BCC, icosahedral, random)
    2. Relax each using the provided L-BFGS-B relaxer
    3. Use multi-start with perturbation for harder cases
    4. Return the best energy configuration found within time budget
    """
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    potential = instance.get("potential", "sutton_chen")
    
    # Determine spacing based on kind (equilibrium distance ~0.72 for SC)
    spacing = 0.72
    
    best_energy = np.inf
    best_positions = None
    
    def _try_relax(positions, label=""):
        nonlocal best_energy, best_positions
        if _time_left() < 2.0:
            return
        try:
            energy, relaxed = relax(positions, potential, kind)
            if np.isfinite(energy) and energy < best_energy:
                # Check minimum distance using vectorized operations
                # cdist is efficient
                dists = cdist(relaxed, relaxed)
                np.fill_diagonal(dists, np.inf)
                min_d = np.min(dists)
                if min_d >= 0.35:
                    best_energy = energy
                    best_positions = relaxed.copy()
        except Exception:
            pass
    
    # Generate diverse starting structures
    start_structures = []
    
    # 1. FCC-like
    try:
        fcc = _make_fcc(n, spacing)
        if fcc is not None and len(fcc) == n:
            start_structures.append(fcc)
    except Exception:
        pass
    
    # 2. BCC-like
    try:
        bcc = _make_bcc(n, spacing)
        if bcc is not None and len(bcc) == n:
            start_structures.append(bcc)
    except Exception:
        pass
    
    # 3. Mackay Icosahedral
    try:
        ico = _make_mackay_icosahedral(n, spacing)
        if ico is not None and len(ico) == n:
            start_structures.append(ico)
    except Exception:
        pass
        
    # 3b. Spherical Shells
    for seed in [n+5, n+6]:
        try:
            sh = _make_spherical_shells(n, spacing, seed=seed)
            if sh is not None and len(sh) == n:
                start_structures.append(sh)
        except Exception:
            pass
    
    # 4-7. Random compact with different seeds and densities
    for seed in [n, n+1, n+2, n+3]:
        try:
            rnd = _make_random_compact(n, spacing, seed=seed, min_dist=0.5)
            if len(rnd) == n:
                start_structures.append(rnd)
        except Exception:
            pass
    
    # 8. Random with tighter packing
    for seed in [n+10, n+11]:
        try:
            rnd = _make_random_compact(n, spacing, seed=seed, min_dist=0.38)
            if len(rnd) == n:
                start_structures.append(rnd)
        except Exception:
            pass
    
    # 9. Deformed FCC variants with different noise levels
    if _make_fcc(n, spacing) is not None:
        for seed in [n+20, n+21]:
            try:
                fcc_base = _make_fcc(n, spacing)
                deformed = _make_deformed(n, fcc_base, spacing, seed=seed)
                start_structures.append(deformed)
            except Exception:
                pass
    
    # Relax all starting structures
    for i, structure in enumerate(start_structures):
        if _time_left() < 3.0:
            break
        _try_relax(structure, f"struct_{i}")
    
    # If we have time, do perturbation-based local search from the best
    # Use multiple scales of perturbation to escape local minima
    if best_positions is not None and _time_left() > 10.0:
        n_perturb = 15
        for k in range(n_perturb):
            if _time_left() < 4.0:
                break
            seed = n + 100 + k
            # Vary perturbation scale: start small, grow larger
            scale = 0.03 + 0.01 * k
            perturbed = _make_deformed(n, best_positions, spacing, seed=seed)
            rng = np.random.default_rng(seed)
            noise = rng.normal(0, scale * spacing, perturbed.shape)
            perturbed += noise
            _try_relax(perturbed, f"perturb_{k}")
    
    # If still time, try more diverse random starts
    extra_starts = 5
    for k in range(extra_starts):
        if _time_left() < 4.0:
            break
        seed = n + 200 + k
        try:
            # Vary min_dist to get different packing densities
            md = 0.4 + 0.05 * (k % 3)
            rnd = _make_random_compact(n, spacing, seed=seed, min_dist=md)
            if len(rnd) == n:
                _try_relax(rnd, f"extra_{k}")
        except Exception:
            pass
    
    # Second round of perturbations from best if time allows
    if best_positions is not None and _time_left() > 15.0:
        n_perturb2 = 8
        for k in range(n_perturb2):
            if _time_left() < 4.0:
                break
            seed = n + 300 + k
            # Larger perturbations to explore new basins
            scale = 0.15 + 0.02 * k
            rng = np.random.default_rng(seed)
            noise = rng.normal(0, scale * spacing, best_positions.shape)
            perturbed = best_positions + noise
            _try_relax(perturbed, f"perturb2_{k}")
    
    # Final fallback: if we still don't have a valid result, use a simple compact arrangement
    if best_positions is None or not np.all(np.isfinite(best_positions)):
        # Simple spiral packing
        positions = np.zeros((n, 3))
        for i in range(n):
            theta = i * 2.399  # golden angle
            r = spacing * 0.3 * np.sqrt(i)
            positions[i] = [r * np.cos(theta), r * np.sin(theta), 0.2 * i * 0.05]
        # Relax this fallback
        try:
            energy, relaxed = relax(positions, potential, kind)
            if np.isfinite(energy):
                best_positions = relaxed
        except Exception:
            best_positions = positions
    
    # Final polish: one more relaxation from the best found
    if best_positions is not None and np.all(np.isfinite(best_positions)) and _time_left() > 5.0:
        try:
            energy, relaxed = relax(best_positions, potential, kind)
            if np.isfinite(energy):
                dists = cdist(relaxed, relaxed)
                np.fill_diagonal(dists, np.inf)
                if np.min(dists) >= 0.35:
                    best_positions = relaxed
        except Exception:
            pass
    
    # Ensure validity: check all coordinates are finite and min distance >= 0.35
    if not np.all(np.isfinite(best_positions)):
        # Last resort: spread out on a grid
        positions = np.zeros((n, 3))
        side = int(np.ceil(n ** (1/3)))
        for i in range(n):
            ix = i % side
            iy = (i // side) % side
            iz = i // (side * side)
            positions[i] = [ix * 0.5, iy * 0.5, iz * 0.5]
        best_positions = positions
    
    return best_positions
# EVOLVE-BLOCK-END
