"""Program for Sutton-Chen TM cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 290.0  # Safety margin before 300s kill


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
    """Generate a cluster of n atoms in an FCC-like structure."""
    max_r = int(np.ceil(n ** (1 / 3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * (1.0 / np.sqrt(2.0))
                    candidates.append((np.linalg.norm(p), p))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_bcc(n):
    """Generate a cluster of n atoms in a BCC-like structure."""
    max_r = int(np.ceil(n ** (1 / 3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k])
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5])
                candidates.append((np.linalg.norm(p1), p1))
                candidates.append((np.linalg.norm(p2), p2))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_icosahedron(n):
    """Generate a Mackay icosahedral cluster of n atoms."""
    # Mackay icosahedra: 1, 13, 55, 147
    # Build by concentric shells
    candidates = []
    # Center
    candidates.append(np.array([0.0, 0.0, 0.0]))
    
    # Icosahedron vertices (12) at radius r1
    phi = (1 + np.sqrt(5)) / 2
    r1 = 1.0
    ico_verts = []
    for sign1 in [1, -1]:
        for sign2 in [1, -1]:
            for sign3 in [1, -1]:
                ico_verts.append(np.array([0, sign1, sign2 * phi]) * sign3 / np.sqrt(1 + phi**2))
    # Actually let me use proper icosahedron vertices
    ico_verts = []
    t = phi
    vertices = [
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1],
    ]
    for v in vertices:
        v = np.array(v, dtype=float)
        v = v / np.linalg.norm(v) * r1
        ico_verts.append(v)
    candidates.extend(ico_verts)
    
    # Second shell: icosahedron vertices at radius r2, plus face centers
    r2 = 2.0 * r1
    for v in ico_verts:
        candidates.append(v * r2 / r1)
    
    # Third shell
    r3 = 3.0 * r1
    for v in ico_verts:
        candidates.append(v * r3 / r1)
    
    # Also add octahedral sites for better packing
    octa_sites = []
    for i in range(len(ico_verts)):
        for j in range(i+1, len(ico_verts)):
            mid = (ico_verts[i] + ico_verts[j]) / 2
            mid = mid / np.linalg.norm(mid) * r1
            octa_sites.append(mid)
    candidates.extend(octa_sites)
    
    # Sort by distance from center and take n
    candidates.sort(key=lambda p: np.linalg.norm(p))
    pts = candidates[:n]
    return np.array(pts)


def _generate_random_compact(n, rng, radius=2.0):
    """Generate n random points within a sphere."""
    pts = []
    for _ in range(n):
        while True:
            p = rng.normal(0, 1, 3)
            norm = np.linalg.norm(p)
            if norm <= 1.0:
                break
        pts.append(p * radius)
    return np.array(pts)


def _growth_construct(n, kind, max_time_per_atom=5.0):
    """
    Deterministic growth algorithm: build cluster atom-by-atom.
    For each new atom, try candidate interstitial sites and keep the one
    that gives the lowest total energy after relaxation.
    """
    if n <= 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Start with 2 atoms
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    try:
        energy, pos = relax(pos, 'sutton_chen', kind)
    except Exception:
        pass
    
    current_n = 2
    
    while current_n < n:
        if _time_left() < 20.0:
            break
        
        # Generate candidate positions for the next atom
        # Strategy: try positions near existing atoms (interstitial sites)
        # and some positions at larger distances
        candidates = []
        
        # For each existing atom, try positions in a shell around it
        for i in range(current_n):
            atom = pos[i]
            # Try icosahedral directions from this atom
            t = (1 + np.sqrt(5)) / 2
            ico_dirs = [
                [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
                [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
                [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1],
            ]
            for d in ico_dirs:
                d = np.array(d, dtype=float)
                d = d / np.linalg.norm(d)
                for r in [1.0, 1.2, 1.5]:
                    candidates.append(atom + d * r)
        
        # Also try positions near the centroid at various distances
        centroid = np.mean(pos, axis=0)
        for r in [1.5, 2.0, 2.5]:
            t = (1 + np.sqrt(5)) / 2
            ico_dirs = [
                [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
                [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
                [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1],
            ]
            for d in ico_dirs:
                d = np.array(d, dtype=float)
                d = d / np.linalg.norm(d)
                candidates.append(centroid + d * r)
        
        # Remove duplicates (within tolerance)
        unique_candidates = []
        for c in candidates:
            is_dup = False
            for uc in unique_candidates:
                if np.linalg.norm(c - uc) < 0.1:
                    is_dup = True
                    break
            if not is_dup:
                unique_candidates.append(c)
        
        # Limit candidates to keep computation manageable
        max_candidates = min(len(unique_candidates), 60)
        if len(unique_candidates) > max_candidates:
            # Sort by distance to centroid (prefer compact)
            centroid = np.mean(pos, axis=0)
            unique_candidates.sort(key=lambda c: np.linalg.norm(c - centroid))
            unique_candidates = unique_candidates[:max_candidates]
        
        # Evaluate each candidate by adding it and relaxing
        best_energy = np.inf
        best_pos = None
        
        for cand in unique_candidates:
            if _time_left() < 15.0:
                break
            
            new_pos = np.vstack([pos, cand.reshape(1, 3)])
            try:
                e, relaxed = relax(new_pos, 'sutton_chen', kind)
                if np.isfinite(e) and _is_valid(relaxed):
                    if e < best_energy:
                        best_energy = e
                        best_pos = relaxed
            except Exception:
                continue
        
        if best_pos is None:
            # Fallback: just add at a random-ish position
            centroid = np.mean(pos, axis=0)
            new_atom = centroid + np.array([1.0, 0.5, 0.3])
            new_pos = np.vstack([pos, new_atom.reshape(1, 3)])
            try:
                e, relaxed = relax(new_pos, 'sutton_chen', kind)
                if np.isfinite(e):
                    best_pos = relaxed
            except Exception:
                best_pos = new_pos
        
        pos = best_pos
        current_n += 1
    
    return pos


def _basin_hopping(positions, kind, n_steps=200, step_size=0.5, rng=None):
    """
    Simple basin hopping: perturb, relax, accept if lower energy.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    current_energy, current_pos = relax(positions.copy(), 'sutton_chen', kind)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    for step in range(n_steps):
        if _time_left() < 10.0:
            break
        
        # Perturb
        noise = rng.normal(0, step_size, current_pos.shape)
        perturbed = current_pos + noise
        
        try:
            e, relaxed = relax(perturbed, 'sutton_chen', kind)
            if not np.isfinite(e):
                continue
            if not _is_valid(relaxed):
                continue
            
            # Metropolis acceptance (T=0 for simplicity, but allow small uphill moves)
            if e < current_energy:
                current_energy = e
                current_pos = relaxed
                if e < best_energy:
                    best_energy = e
                    best_pos = relaxed.copy()
            else:
                # Allow small uphill moves with decreasing probability
                delta_e = e - current_energy
                if delta_e < 0.5:
                    current_energy = e
                    current_pos = relaxed
        except Exception:
            continue
    
    return best_energy, best_pos


def _polish(positions, kind, n_rounds=3):
    """Multiple rounds of relaxation to ensure convergence."""
    pos = positions.copy()
    for _ in range(n_rounds):
        if _time_left() < 5.0:
            break
        try:
            e, pos = relax(pos, 'sutton_chen', kind)
            if not np.isfinite(e):
                break
        except Exception:
            break
    return pos


def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    rng = np.random.default_rng(42 + n)
    
    def _update_best(e, p):
        nonlocal best_energy, best_pos
        if np.isfinite(e) and _is_valid(p) and e < best_energy:
            best_energy = e
            best_pos = p.copy()
    
    # Phase 1: Growth construction (deterministic, exploits structure)
    if _time_left() > 60.0:
        try:
            growth_pos = _growth_construct(n, kind)
            if _is_valid(growth_pos):
                e, relaxed = relax(growth_pos, 'sutton_chen', kind)
                if np.isfinite(e):
                    _update_best(e, relaxed)
        except Exception:
            pass
    
    # Phase 2: Basin hopping from multiple diverse starting points
    starting_points = []
    
    # From growth result if available
    if best_pos is not None:
        starting_points.append(best_pos.copy())
    
    # FCC template
    fcc_pos = _generate_fcc(n)
    if _is_valid(fcc_pos):
        starting_points.append(fcc_pos)
    
    # BCC template
    bcc_pos = _generate_bcc(n)
    if _is_valid(bcc_pos):
        starting_points.append(bcc_pos)
    
    # Icosahedral template
    ico_pos = _generate_icosahedron(n)
    if _is_valid(ico_pos):
        starting_points.append(ico_pos)
    
    # Random compact
    for _ in range(3):
        rand_pos = _generate_random_compact(n, rng, radius=2.5)
        if _is_valid(rand_pos):
            starting_points.append(rand_pos)
    
    # Relax each starting point and run basin hopping
    for sp in starting_points:
        if _time_left() < 15.0:
            break
        
        try:
            e, relaxed = relax(sp, 'sutton_chen', kind)
            if np.isfinite(e) and _is_valid(relaxed):
                _update_best(e, relaxed)
        except Exception:
            continue
    
    # Phase 3: Basin hopping from the best found so far
    if best_pos is not None and _time_left() > 30.0:
        try:
            bh_energy, bh_pos = _basin_hopping(best_pos, kind, n_steps=300, step_size=0.4, rng=rng)
            _update_best(bh_energy, bh_pos)
        except Exception:
            pass
    
    # Phase 4: Additional basin hopping from other starting points if time remains
    if _time_left() > 30.0:
        for sp in starting_points[:4]:
            if _time_left() < 15.0:
                break
            try:
                e, relaxed = relax(sp, 'sutton_chen', kind)
                if np.isfinite(e) and _is_valid(relaxed):
                    bh_e, bh_p = _basin_hopping(relaxed, kind, n_steps=100, step_size=0.3, rng=rng)
                    _update_best(bh_e, bh_p)
            except Exception:
                continue
    
    # Phase 5: Final polish
    if best_pos is not None and _time_left() > 10.0:
        polished = _polish(best_pos, kind, n_rounds=3)
        try:
            e, _ = sutton_chen_energy_gradient(polished, kind)
            if np.isfinite(e) and _is_valid(polished) and e < best_energy:
                best_energy = e
                best_pos = polished
        except Exception:
            pass
    
    # Final fallback
    if best_pos is None:
        best_pos = _generate_fcc(n)
        try:
            e, relaxed = relax(best_pos, 'sutton_chen', kind)
            if np.isfinite(e):
                best_pos = relaxed
        except Exception:
            pass
    
    return best_pos

# EVOLVE-BLOCK-END
