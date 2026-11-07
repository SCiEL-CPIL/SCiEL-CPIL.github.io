"""Program for Sutton-Chen TM cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    """Check if positions are valid (finite, no atoms closer than min_dist)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    dists = np.sqrt(np.sum((positions[:, np.newaxis, :] - positions[np.newaxis, :, :]) ** 2, axis=-1))
    mask = ~np.eye(n, dtype=bool)
    min_d = np.min(dists[mask])
    if min_d < min_dist:
        return False
    return True


def _generate_fcc(n):
    """Generate n atoms in FCC-like arrangement (closest to origin)."""
    max_r = int(np.ceil(n ** (1 / 3))) + 5
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
    max_r = int(np.ceil(n ** (1 / 3))) + 5
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
    v = [
        np.array([0, 1, phi]), np.array([0, 1, -phi]),
        np.array([0, -1, phi]), np.array([0, -1, -phi]),
        np.array([1, phi, 0]), np.array([1, -phi, 0]),
        np.array([-1, phi, 0]), np.array([-1, -phi, 0]),
        np.array([phi, 0, 1]), np.array([phi, 0, -1]),
        np.array([-phi, 0, 1]), np.array([-phi, 0, -1])
    ]
    v = [p / np.linalg.norm(p) for p in v]
    
    pts = [np.zeros(3)]
    if n > 1:
        pts.extend(v[:min(12, n - 1)])
    
    if n > 13:
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(v[i] - v[j])
                if 1.5 < d < 2.5:
                    edges.append((i, j))
        
        shell2_pts = []
        for i, j in edges[:30]:
            mid = (v[i] + v[j]) / 2
            mid = mid * 2.0 / np.linalg.norm(mid)
            shell2_pts.append(mid)
        
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
        
        shell2_pts.extend(faces[:12])
        
        unique_pts = []
        for p in shell2_pts:
            if not any(np.linalg.norm(p - q) < 0.1 for q in unique_pts):
                unique_pts.append(p)
        
        remaining = min(42, n - len(pts))
        pts.extend(unique_pts[:remaining])
    
    if n > 55:
        rng = np.random.default_rng(123)
        remaining = min(92, n - len(pts))
        for _ in range(remaining * 5):
            if len(pts) >= n:
                break
            p = rng.normal(size=3)
            p = p * 3.0 / np.linalg.norm(p)
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
    Systematic growth: build cluster atom by atom at low-energy interstitial sites.
    """
    rng = np.random.default_rng(seed + n * 100)
    
    # Start from a small relaxed cluster (4-6 atoms)
    start_n = min(6, n)
    base = _generate_fcc(start_n)
    if not _is_valid(base):
        base = _generate_bcc(start_n)
    if not _is_valid(base):
        base = _generate_random(start_n, seed)
    
    e, current = relax(base, 'sutton_chen', kind)
    if not (np.isfinite(e) and _is_valid(current)):
        current = base
    
    # Grow to n atoms
    for target_n in range(len(current) + 1, n + 1):
        if _time_left() < 15.0:
            break
        
        n_atoms = len(current)
        center = np.mean(current, axis=0)
        
        # Generate candidate positions for new atom
        candidates = []
        
        # 1. Interstitial sites from existing pairs (midpoints offset outward)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                mid = (current[i] + current[j]) / 2
                direction = mid - center
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    direction /= norm
                else:
                    direction = np.array([1.0, 0, 0])
                
                for offset in [0.4, 0.6, 0.8, 1.0, 1.3]:
                    pos = mid + direction * offset
                    candidates.append(pos)
        
        # 2. Random shell positions
        radius = 0.8 + 0.25 * np.sqrt(target_n)
        for _ in range(15):
            p = rng.normal(size=3)
            p = p * radius / np.linalg.norm(p) + center
            candidates.append(p)
        
        # 3. Positions near existing atoms (slightly displaced)
        for i in range(min(8, n_atoms)):
            for _ in range(3):
                p = current[i] + rng.normal(0, 0.3, 3)
                candidates.append(p)
        
        # Evaluate top candidates by quick energy estimate
        best_e = np.inf
        best_pos = None
        
        # Sort candidates by distance to center (prefer compact)
        candidates = np.array(candidates)
        dists_to_center = np.linalg.norm(candidates - center, axis=1)
        order = np.argsort(dists_to_center)
        
        # Try up to 40 best candidates
        max_try = min(40, len(order))
        for idx in order[:max_try]:
            if _time_left() < 15.0:
                break
            
            pos_new = candidates[idx].copy()
            
            # Quick validity check
            if not np.all(np.isfinite(pos_new)):
                continue
            if np.any(np.linalg.norm(pos_new - current, axis=1) < 0.35):
                continue
            
            trial = np.vstack([current, pos_new])
            
            try:
                e_trial, relaxed = relax(trial, 'sutton_chen', kind)
                if np.isfinite(e_trial) and _is_valid(relaxed):
                    if e_trial < best_e:
                        best_e = e_trial
                        best_pos = relaxed.copy()
            except Exception:
                continue
        
        if best_pos is None:
            # Fallback: add atom at a random shell position
            p = rng.normal(size=3)
            p = p * (1.0 + 0.3 * np.sqrt(target_n)) / np.linalg.norm(p) + center
            trial = np.vstack([current, p])
            try:
                e_trial, relaxed = relax(trial, 'sutton_chen', kind)
                if np.isfinite(e_trial) and _is_valid(relaxed):
                    best_pos = relaxed
            except Exception:
                best_pos = trial
        
        if best_pos is not None:
            current = best_pos
    
    return current


def _basin_hop_chain(positions, kind, n_steps, rng):
    """Run a basin hopping chain."""
    e, current = relax(positions, 'sutton_chen', kind)
    if not (np.isfinite(e) and _is_valid(current)):
        return positions, np.inf
    
    best_e = e
    best_pos = current.copy()
    
    for i in range(n_steps):
        if _time_left() < 8.0:
            break
        
        sigma = 0.08 + 0.08 * (1 + np.sin(i * 0.15)) / 2
        pos = current + rng.normal(0, sigma, (len(current), 3))
        
        if not _is_valid(pos):
            continue
        
        try:
            e_new, relaxed = relax(pos, 'sutton_chen', kind)
            if np.isfinite(e_new) and _is_valid(relaxed):
                if e_new < best_e:
                    best_e = e_new
                    best_pos = relaxed.copy()
                current = relaxed
        except Exception:
            continue
            
    return best_pos, best_e


def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    # Phase 1: Generate diverse initial structures via multiple methods
    initial_candidates = []
    
    # Growth-based initial structures (primary strategy)
    for seed in [0, 1, 2]:
        if _time_left() < 25.0:
            break
        try:
            grown = _growth_cluster(n, kind, seed=seed)
            if _is_valid(grown):
                energy, relaxed = relax(grown, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    initial_candidates.append((energy, relaxed))
        except Exception:
            continue
    
    # FCC initial
    if _time_left() > 20.0:
        try:
            base = _generate_fcc(n)
            if _is_valid(base):
                energy, relaxed = relax(base, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    initial_candidates.append((energy, relaxed))
        except Exception:
            pass
    
    # BCC initial
    if _time_left() > 20.0:
        try:
            base = _generate_bcc(n)
            if _is_valid(base):
                energy, relaxed = relax(base, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    initial_candidates.append((energy, relaxed))
        except Exception:
            pass
    
    # Mackay initial (especially good for n=55)
    if n in [13, 55, 147] or _time_left() > 20.0:
        try:
            base = _generate_mackay(n)
            if _is_valid(base):
                energy, relaxed = relax(base, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    initial_candidates.append((energy, relaxed))
        except Exception:
            pass
    
    # Random initial structures
    for seed in [10, 20, 30]:
        if _time_left() < 15.0:
            break
        try:
            base = _generate_random(n, seed)
            if _is_valid(base):
                energy, relaxed = relax(base, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    initial_candidates.append((energy, relaxed))
        except Exception:
            continue
    
    if not initial_candidates:
        best_pos = _generate_fcc(n)
        if not _is_valid(best_pos):
            best_pos = _generate_random(n, 0)
        return best_pos
    
    # Sort by energy
    initial_candidates.sort(key=lambda x: x[0])
    
    # Update best from initial candidates
    for e, pos in initial_candidates:
        if e < best_energy:
            best_energy = e
            best_pos = pos.copy()
    
    # Phase 2: Basin Hopping from the best candidates
    n_chains = min(4, len(initial_candidates))
    rngs = [np.random.default_rng(200 + i) for i in range(n_chains)]
    
    for i in range(n_chains):
        if _time_left() < 20.0:
            break
        
        e_init, pos_init = initial_candidates[i]
        
        # Run BH chain with adaptive steps based on remaining time
        time_per_step = max(0.5, _time_left() / 100.0)
        n_steps = max(10, int(min(80, _time_left() / max(time_per_step, 0.5))))
        
        bh_pos, bh_energy = _basin_hop_chain(pos_init, kind, n_steps=n_steps, rng=rngs[i])
        
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos.copy()
    
    # Phase 3: If time remains, continue BH from the global best
    if best_pos is not None and _time_left() > 15.0:
        rng_final = np.random.default_rng(999)
        n_steps_final = max(5, int(_time_left() / 2.0))
        bh_pos, bh_energy = _basin_hop_chain(best_pos, kind, n_steps=n_steps_final, rng=rng_final)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos.copy()
    
    # Final validation
    if best_pos is not None and not _is_valid(best_pos):
        # Try to repair
        try:
            e, repaired = relax(best_pos, 'sutton_chen', kind)
            if np.isfinite(e) and _is_valid(repaired):
                best_pos = repaired
        except Exception:
            best_pos = _generate_fcc(n)
    
    return best_pos

# EVOLVE-BLOCK-END
