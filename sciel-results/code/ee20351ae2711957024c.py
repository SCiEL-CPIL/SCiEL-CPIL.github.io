"""Sutton-Chen transition-metal cluster global minima search.
Uses deterministic growth (atom-by-atom insertion at best interstitial sites)
combined with a portfolio of structural seeds and basin-hopping refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START = time.time()
_DEADLINE = 290.0  # safety margin before 330s kill


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _is_valid(positions, min_dist=0.35):
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    # Vectorized pairwise distance check
    from scipy.spatial.distance import pdist
    d = pdist(positions, 'euclidean')
    if np.any(d < min_dist):
        return False
    return True


def _candidate_positions_for_growth(existing, n_total, seed=42):
    """Generate candidate positions for adding atoms to an existing cluster.
    
    Candidates come from:
    1. Midpoints between existing atom pairs (interstitial sites)
    2. Positions offset from existing atoms along bond directions
    3. Random positions near the cluster surface
    """
    rng = np.random.default_rng(seed)
    candidates = []
    
    # 1. Midpoints and offsets between pairs
    n = len(existing)
    for i in range(n):
        for j in range(i + 1, n):
            mid = (existing[i] + existing[j]) / 2.0
            candidates.append(mid)
            # Also try positions slightly offset from midpoint
            d = existing[j] - existing[i]
            dist = np.linalg.norm(d)
            if dist > 1e-6:
                d_hat = d / dist
                candidates.append(mid + 0.3 * d_hat)
                candidates.append(mid - 0.3 * d_hat)
    
    # 2. Positions offset from existing atoms (surface-like)
    for i in range(n):
        # Radial direction from center of mass
        com = np.mean(existing, axis=0)
        r_vec = existing[i] - com
        r_norm = np.linalg.norm(r_vec)
        if r_norm > 1e-6:
            r_hat = r_vec / r_norm
            for scale in [1.2, 1.5, 1.8]:
                candidates.append(existing[i] + scale * r_hat * 0.5)
        # Random directions from existing atom
        for _ in range(3):
            rand_dir = rng.normal(0, 1, 3)
            rand_dir /= (np.linalg.norm(rand_dir) + 1e-10)
            candidates.append(existing[i] + 0.8 * rand_dir)
    
    # 3. Random positions in a shell around the cluster
    com = np.mean(existing, axis=0)
    radius = np.max(np.linalg.norm(existing - com, axis=1)) + 1.0
    for _ in range(20):
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        r = radius * rng.uniform(0.5, 1.5)
        p = com + np.array([r * np.sin(phi) * np.cos(theta),
                           r * np.sin(phi) * np.sin(theta),
                           r * np.cos(phi)])
        candidates.append(p)
    
    return np.array(candidates)


def _make_fcc(n):
    """Generate a compact FCC-like cluster of n atoms."""
    pts = []
    mc = int(np.ceil(n ** (1.0 / 3.0))) + 2
    a = 1.0
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    if len(pts) < n:
                        pts.append(p)
    if len(pts) < n:
        rng = np.random.default_rng(42)
        while len(pts) < n:
            p = rng.uniform(-mc * a, mc * a, 3)
            pts.append(p)
    return np.array(pts[:n])


def _make_bcc(n):
    """Generate a compact BCC-like cluster of n atoms."""
    pts = []
    mc = int(np.ceil(n ** (1.0 / 3.0))) + 2
    a = 1.0
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                p = np.array([i, j, k]) * a
                if len(pts) < n:
                    pts.append(p)
                pc = np.array([i + 0.5, j + 0.5, k + 0.5]) * a
                if len(pts) < n:
                    pts.append(pc)
    if len(pts) < n:
        rng = np.random.default_rng(42)
        while len(pts) < n:
            p = rng.uniform(-mc * a, mc * a, 3)
            pts.append(p)
    return np.array(pts[:n])


def _make_random_compact(n, seed, spacing=0.72):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
                points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _make_mackay(n):
    """Generate a Mackay icosahedron-like cluster."""
    pts = []
    pts.append(np.array([0.0, 0.0, 0.0]))
    
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ]
    ico = np.array(ico, dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    
    for v in ico:
        if len(pts) < n:
            pts.append(v * 1.0)
    
    if len(pts) < n:
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(ico[i] - ico[j])
                if 0.9 < d < 1.1:
                    mid = (ico[i] + ico[j]) / 2.0
                    mid = mid / np.linalg.norm(mid) * 1.5
                    edges.append(mid)
        for e in edges:
            if len(pts) < n:
                pts.append(e)
    
    if len(pts) < n:
        rng = np.random.default_rng(123)
        while len(pts) < n:
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(0, np.pi)
            r = 2.0
            p = np.array([r * np.sin(phi) * np.cos(theta),
                         r * np.sin(phi) * np.sin(theta),
                         r * np.cos(phi)])
            pts.append(p)
    
    return np.array(pts[:n])


def _make_deformed_fcc(n, seed, deform=0.3):
    """FCC with random deformation."""
    pts = _make_fcc(n)
    rng = np.random.default_rng(seed)
    pts = pts + rng.normal(0, deform, pts.shape)
    return pts


def _deterministic_growth(n, kind, seed=42, max_candidates_per_step=15):
    """Build cluster atom-by-atom by finding best interstitial site.
    
    At each step, generate candidate positions near the existing cluster,
    try inserting each candidate, relax, and keep the one with lowest energy.
    """
    rng = np.random.default_rng(seed)
    
    # Start with a small seed cluster
    if n <= 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Initialize with a few atoms
    if n == 2:
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elif n == 3:
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866, 0.0]])
    else:
        # Start with a small FCC-like or icosahedral seed
        pos = _make_fcc(min(n, 5))
    
    # Relax initial seed
    try:
        _, pos = relax(pos, 'sutton_chen', kind)
    except Exception:
        pass
    
    current_n = len(pos)
    
    # Grow atom by atom
    while current_n < n:
        if _time_left() < 5.0:
            break
        
        # Generate candidates
        candidates = _candidate_positions_for_growth(pos, n, seed=seed + current_n)
        
        # Limit candidates for efficiency
        if len(candidates) > max_candidates_per_step:
            # Sample candidates: prioritize those closer to existing atoms
            com = np.mean(pos, axis=0)
            dists = np.linalg.norm(candidates - com, axis=1)
            idx = np.argsort(dists)[:max_candidates_per_step]
            candidates = candidates[idx]
        
        best_energy = np.inf
        best_new_pos = None
        
        for cand in candidates:
            if _time_left() < 3.0:
                break
            
            new_pos = np.vstack([pos, cand.reshape(1, 3)])
            
            if not _is_valid(new_pos, min_dist=0.35):
                continue
            
            try:
                energy, relaxed = relax(new_pos, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed, min_dist=0.35):
                    if energy < best_energy:
                        best_energy = energy
                        best_new_pos = relaxed.copy()
            except Exception:
                continue
        
        if best_new_pos is not None:
            pos = best_new_pos
            current_n = len(pos)
        else:
            # Fallback: add atom at a random position
            com = np.mean(pos, axis=0)
            rand_dir = rng.normal(0, 1, 3)
            rand_dir /= (np.linalg.norm(rand_dir) + 1e-10)
            new_atom = com + 1.5 * rand_dir
            pos = np.vstack([pos, new_atom.reshape(1, 3)])
            current_n = len(pos)
            try:
                _, pos = relax(pos, 'sutton_chen', kind)
            except Exception:
                pass
    
    return pos


def _basin_hop(positions, kind, n_iter=10, perturb_scale=0.3, seed=42):
    """Simple basin hopping: perturb, relax, keep if lower energy."""
    rng = np.random.default_rng(seed)
    best_pos = positions.copy()
    best_energy, _ = sutton_chen_energy_gradient(positions, kind)
    
    for i in range(n_iter):
        if _time_left() < 5.0:
            break
        
        # Perturb
        perturbation = rng.normal(0, perturb_scale, best_pos.shape)
        perturbed = best_pos + perturbation
        
        if not _is_valid(perturbed):
            continue
        
        try:
            energy, relaxed = relax(perturbed, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed.copy()
        except Exception:
            continue
    
    return best_energy, best_pos


def construct(instance):
    size = instance["n"]
    kind = instance.get("kind", instance.get("rho", "12-6"))
    
    best_energy = np.inf
    best_positions = None
    
    def _try_relax(pos):
        nonlocal best_energy, best_positions
        if not _is_valid(pos):
            return
        try:
            energy, relaxed = relax(pos, "sutton_chen", kind)
            if np.isfinite(energy) and _is_valid(relaxed):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed.copy()
        except Exception:
            pass
    
    # Strategy 1: Deterministic growth (primary structural change)
    # Allocate ~40% of time budget to growth
    growth_time_budget = _time_left() * 0.4
    growth_start = time.time()
    
    if size >= 2:
        try:
            growth_pos = _deterministic_growth(size, kind, seed=42, max_candidates_per_step=10)
            _try_relax(growth_pos)
        except Exception:
            pass
    
    growth_elapsed = time.time() - growth_start
    
    # Strategy 2: Portfolio of structural seeds
    starts = []
    
    # FCC-based
    starts.append(_make_fcc(size))
    starts.append(_make_deformed_fcc(size, seed=1, deform=0.2))
    starts.append(_make_deformed_fcc(size, seed=2, deform=0.4))
    
    # BCC-based
    starts.append(_make_bcc(size))
    
    # Mackay-like
    starts.append(_make_mackay(size))
    
    # Random compact
    for s in range(4):
        starts.append(_make_random_compact(size, seed=100 + s))
    
    # Relax each starting point (sequentially, with time check)
    for idx, pos in enumerate(starts):
        if _time_left() < 15.0:
            break
        _try_relax(pos)
    
    # Strategy 3: Basin hopping from best solution
    if best_positions is not None and _time_left() > 20.0:
        # Multiple basin hopping runs with different perturbation scales
        for scale, n_iter in [(0.2, 15), (0.4, 10), (0.6, 8)]:
            if _time_left() < 10.0:
                break
            try:
                bh_energy, bh_pos = _basin_hop(best_positions, kind, 
                                               n_iter=n_iter, 
                                               perturb_scale=scale, 
                                               seed=1000 + int(scale * 100))
                if bh_energy < best_energy:
                    best_energy = bh_energy
                    best_positions = bh_pos
            except Exception:
                continue
    
    # Strategy 4: Final refinement with smaller perturbations
    if best_positions is not None and _time_left() > 15.0:
        rng = np.random.default_rng(777)
        for attempt in range(15):
            if _time_left() < 8.0:
                break
            perturbation = rng.normal(0, 0.1, best_positions.shape)
            perturbed = best_positions + perturbation
            _try_relax(perturbed)
    
    if best_positions is None:
        best_positions = _make_fcc(size)
        _try_relax(best_positions)
    
    # Final validation
    if not _is_valid(best_positions):
        best_positions = np.zeros((size, 3))
        for i in range(size):
            best_positions[i] = [i * 0.5, (i // 10) * 0.5, (i % 10) * 0.5]
    
    return best_positions
# EVOLVE-BLOCK-END
