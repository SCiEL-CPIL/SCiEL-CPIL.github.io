"""Program for Sutton-Chen TM cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    """Check if positions are valid (finite, no atoms closer than min_dist)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n - 1):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True


def _generate_fcc(n):
    """Generate n atoms in FCC-like arrangement (closest to origin)."""
    max_r = int(np.ceil(n ** (1 / 3))) + 4
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k], dtype=float) / np.sqrt(2.0)
                    candidates.append((np.dot(p, p), p))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_bcc(n):
    """Generate n atoms in BCC-like arrangement."""
    max_r = int(np.ceil(n ** (1 / 3))) + 4
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k], dtype=float)
                candidates.append((np.dot(p1, p1), p1))
                p2 = p1 + 0.5
                candidates.append((np.dot(p2, p2), p2))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_mackay(n):
    """Generate Mackay icosahedral cluster."""
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices
    v = [
        np.array([0, 1, phi]), np.array([0, 1, -phi]),
        np.array([0, -1, phi]), np.array([0, -1, -phi]),
        np.array([1, phi, 0]), np.array([1, -phi, 0]),
        np.array([-1, phi, 0]), np.array([-1, -phi, 0]),
        np.array([phi, 0, 1]), np.array([phi, 0, -1]),
        np.array([-phi, 0, 1]), np.array([-phi, 0, -1])
    ]
    v = [p / np.linalg.norm(p) for p in v]
    
    # Shell radii for Mackay: 1, 2, 3...
    # Shell 0: 1 atom (center)
    # Shell 1: 12 atoms (icosahedron)
    # Shell 2: 42 atoms
    # Shell 3: 92 atoms
    
    pts = [np.zeros(3)]
    if n > 1:
        # Shell 1: 12 vertices
        pts.extend(v[:min(12, n - 1)])
    
    if n > 13:
        # Shell 2: 42 atoms
        # These are at positions that form the second shell of Mackay icosahedron
        # We can generate them by finding all points at distance 2 from center
        # that are on the icosahedral symmetry
        # For simplicity, use a denser set of points on sphere of radius 2
        rng = np.random.default_rng(42)
        remaining = min(42, n - len(pts))
        # Generate points on sphere using icosahedral symmetry
        # The 42 points of second shell are: 30 edge midpoints + 12 face centers
        # Edge midpoints: midpoints of the 30 edges
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(v[i] - v[j])
                if 1.5 < d < 2.5:  # adjacent vertices
                    edges.append((i, j))
        
        shell2_pts = []
        for i, j in edges[:30]:
            mid = (v[i] + v[j]) / 2
            mid = mid * 2.0 / np.linalg.norm(mid)
            shell2_pts.append(mid)
        
        # Face centers: 20 faces
        # Find faces (triples of mutually adjacent vertices)
        faces = []
        for i in range(12):
            for j in range(i + 1, 12):
                for k in range(j + 1, 12):
                    d1 = np.linalg.norm(v[i] - v[j])
                    d2 = np.linalg.norm(v[j] - v[k])
                    d3 = np.linalg.norm(v[i] - v[k])
                    if 1.5 < d1 < 2.5 and 1.5 < d2 < 2.5 and 1.5 < d3 < 2.5:
                        center = (v[i] + v[j] + v[k]) / 3
                        center = center * 2.0 / np.linalg.norm(center)
                        faces.append(center)
        
        shell2_pts.extend(faces[:12])  # Take 12 face centers to get 42 total
        
        # Deduplicate and scale
        unique_pts = []
        for p in shell2_pts:
            if not any(np.linalg.norm(p - q) < 0.1 for q in unique_pts):
                unique_pts.append(p)
        
        pts.extend(unique_pts[:remaining])
    
    if n > 55:
        # Shell 3: 92 atoms
        rng = np.random.default_rng(123)
        remaining = min(92, n - len(pts))
        # Generate points on sphere of radius 3
        for _ in range(remaining * 3):
            if len(pts) >= n:
                break
            p = rng.normal(size=3)
            p = p * 3.0 / np.linalg.norm(p)
            # Check minimum distance
            if all(np.linalg.norm(p - q) > 0.5 for q in pts):
                pts.append(p)
    
    return np.array(pts[:n])


def _generate_random(n, seed):
    """Generate random positions in a sphere."""
    rng = np.random.default_rng(seed)
    r = 0.4 * np.sqrt(n)
    pts = rng.normal(size=(n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    pts = pts / norms * r * rng.random((n, 1)) ** (1 / 3)
    return pts


def _growth_cluster(n, kind, seed=0):
    """
    Deterministic growth algorithm: build cluster atom by atom.
    Start from a small cluster, then add atoms at low-energy interstitial sites.
    """
    rng = np.random.default_rng(seed + n)
    
    # Start from a small relaxed cluster
    if n <= 4:
        base = _generate_fcc(n)
        if _is_valid(base):
            e, relaxed = relax(base, 'sutton_chen', kind)
            if np.isfinite(e) and _is_valid(relaxed):
                return relaxed
        return _generate_fcc(n)
    
    # Start with 5 atoms (trigonal bipyramid or similar)
    base = _generate_fcc(5)
    if not _is_valid(base):
        base = _generate_bcc(5)
    e, current = relax(base, 'sutton_chen', kind)
    if not (np.isfinite(e) and _is_valid(current)):
        current = _generate_fcc(5)
    
    # Grow to n atoms
    for target_n in range(len(current) + 1, n + 1):
        if _time_left() < 30.0:
            break
        
        # Generate candidate positions for new atom
        # 1. Interstitial sites: midpoints between existing atoms, offset outward
        # 2. Random positions on a shell around the cluster
        
        candidates = []
        
        # Interstitial sites from existing pairs
        n_atoms = len(current)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                mid = (current[i] + current[j]) / 2
                # Offset outward from center
                center = np.mean(current, axis=0)
                direction = mid - center
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    direction /= norm
                else:
                    direction = np.array([1.0, 0, 0])
                
                # Try several offsets
                for offset in [0.5, 0.8, 1.2, 1.5]:
                    pos = mid + direction * offset
                    candidates.append(pos)
        
        # Random shell positions
        center = np.mean(current, axis=0)
        radius = 1.0 + 0.3 * np.sqrt(target_n)
        for _ in range(20):
            p = rng.normal(size=3)
            p = p * radius / np.linalg.norm(p) + center
            candidates.append(p)
        
        # Evaluate candidates: place new atom, relax, check energy
        best_e = np.inf
        best_pos = None
        
        # Sort candidates by distance to center (prefer closer)
        center = np.mean(current, axis=0)
        candidates = np.array(candidates)
        dists = np.linalg.norm(candidates - center, axis=1)
        order = np.argsort(dists)
        
        # Try top candidates
        n_to_try = min(30, len(candidates))
        for idx in order[:n_to_try]:
            if _time_left() < 15.0:
                break
            
            new_atom = candidates[idx]
            # Check minimum distance to existing atoms
            d_to_existing = np.linalg.norm(current - new_atom, axis=1)
            if np.min(d_to_existing) < 0.35:
                continue
            
            new_cluster = np.vstack([current, new_atom.reshape(1, 3)])
            if not _is_valid(new_cluster):
                continue
            
            try:
                e, relaxed = relax(new_cluster, 'sutton_chen', kind)
                if np.isfinite(e) and _is_valid(relaxed):
                    if e < best_e:
                        best_e = e
                        best_pos = relaxed
            except Exception:
                continue
        
        if best_pos is not None:
            current = best_pos
        else:
            # Fallback: add random atom
            p = rng.normal(size=3)
            p = p * radius / np.linalg.norm(p) + center
            new_cluster = np.vstack([current, p.reshape(1, 3)])
            if not _is_valid(new_cluster):
                # Try several random positions
                for _ in range(10):
                    p = rng.normal(size=3)
                    p = p * radius / np.linalg.norm(p) + center
                    new_cluster = np.vstack([current, p.reshape(1, 3)])
                    if _is_valid(new_cluster):
                        break
            try:
                e, relaxed = relax(new_cluster, 'sutton_chen', kind)
                if np.isfinite(e) and _is_valid(relaxed):
                    current = relaxed
            except Exception:
                current = new_cluster
    
    return current


def _basin_hopping(positions, kind, n_trials=20, sigma_start=0.15):
    """
    Basin hopping: perturb, relax, keep if lower energy.
    """
    rng = np.random.default_rng(12345 + len(positions))
    n = len(positions)
    
    e, current = relax(positions, 'sutton_chen', kind)
    if not (np.isfinite(e) and _is_valid(current)):
        return positions, np.inf
    
    best_e = e
    best_pos = current.copy()
    
    for i in range(n_trials):
        if _time_left() < 15.0:
            break
        
        sigma = sigma_start * (0.85 ** i)
        pos = current + rng.normal(0, sigma, (n, 3))
        
        if not _is_valid(pos):
            continue
        
        try:
            e_new, relaxed = relax(pos, 'sutton_chen', kind)
            if np.isfinite(e_new) and _is_valid(relaxed):
                if e_new < best_e:
                    best_e = e_new
                    best_pos = relaxed.copy()
                # Always accept for basin hopping (or Metropolis, but we want global)
                current = relaxed
        except Exception:
            continue
    
    return best_pos, best_e


def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    # Phase 1: Try multiple templates with relaxation
    generators = [_generate_fcc, _generate_bcc, _generate_mackay]
    
    for gen in generators:
        if _time_left() < 20.0:
            break
        try:
            base = gen(n)
            if not _is_valid(base):
                base = base * 1.3
            if _is_valid(base):
                energy, relaxed = relax(base, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed.copy()
        except Exception:
            continue
    
    # Phase 2: Growth algorithm with different seeds
    if _time_left() > 60.0:
        for seed in [0, 1, 2]:
            if _time_left() < 30.0:
                break
            try:
                grown = _growth_cluster(n, kind, seed=seed)
                if _is_valid(grown):
                    energy, relaxed = relax(grown, 'sutton_chen', kind)
                    if np.isfinite(energy) and _is_valid(relaxed):
                        if energy < best_energy:
                            best_energy = energy
                            best_pos = relaxed.copy()
            except Exception:
                continue
    
    # Phase 3: Basin hopping from best found
    if best_pos is not None and _time_left() > 40.0:
        bh_pos, bh_energy = _basin_hopping(best_pos, kind, n_trials=25, sigma_start=0.15)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos
    
    # Phase 4: More random restarts if time allows
    if _time_left() > 30.0:
        for i in range(5):
            if _time_left() < 15.0:
                break
            pos = _generate_random(n, 10000 + i * 7 + n)
            if not _is_valid(pos):
                continue
            try:
                energy, relaxed = relax(pos, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed.copy()
            except Exception:
                continue
    
    # Fallback
    if best_pos is None:
        best_pos = _generate_fcc(n)
        if not _is_valid(best_pos):
            best_pos = _generate_bcc(n)
        if not _is_valid(best_pos):
            rng2 = np.random.default_rng(999)
            best_pos = rng2.uniform(-3, 3, (n, 3))
            for attempt in range(50):
                if _is_valid(best_pos):
                    break
                best_pos = rng2.uniform(-3, 3, (n, 3))
    
    return best_pos

# EVOLVE-BLOCK-END
