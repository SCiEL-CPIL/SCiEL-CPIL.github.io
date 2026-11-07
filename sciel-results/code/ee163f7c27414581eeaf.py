"""Sutton-Chen transition-metal cluster global minima: round-robin basin hopping with Mackay+decahedral seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = None
_TIME_LIMIT = 305.0

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _min_dist(positions):
    d = cdist(positions, positions)
    np.fill_diagonal(d, np.inf)
    return float(np.min(d))

# --- Proper Mackay icosahedral shells ---
def _ico_shells():
    """Return unit vectors for Mackay icosahedral shells."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 0.8:
                mid = (verts[i] + verts[j]) / 2.0
                mid = mid / np.linalg.norm(mid)
                edges.append(mid)
    edges_u = []
    for e in edges:
        if not any(np.allclose(e, u, atol=1e-6) for u in edges_u):
            edges_u.append(e)
    
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[k] - verts[i])
                if d1 < 0.8 and d2 < 0.8 and d3 < 0.8:
                    fc = (verts[i] + verts[j] + verts[k]) / 3.0
                    fc = fc / np.linalg.norm(fc)
                    faces.append(fc)
    faces_u = []
    for f in faces:
        if not any(np.allclose(f, u, atol=1e-6) for u in faces_u):
            faces_u.append(f)
    
    return verts, edges_u, faces_u

def _make_mackay(n, spacing=0.72):
    """Generate n atoms in proper Mackay icosahedral shells."""
    verts, edges, faces = _ico_shells()
    
    r1 = spacing * 0.6
    r2 = r1 * np.sqrt(2.0)
    r3 = r1 * np.sqrt(3.0)
    r4 = r1 * 2.0
    
    points = [np.zeros(3)]
    
    for v in verts:
        points.append(v * r1)
    
    for e in edges:
        points.append(e * r2)
    
    for f in faces:
        points.append(f * r3)
    
    if n <= len(points):
        return np.array(points[:n])
    
    remaining = n - len(points)
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    for i in range(remaining):
        y = 1.0 - (i + 0.5) / remaining * 2.0
        r = np.sqrt(max(0.0, 1.0 - y * y))
        theta = 2.0 * np.pi * golden * i
        p = np.array([r * np.cos(theta), y, r * np.sin(theta)])
        points.append(p * r4)
    
    return np.array(points[:n])

def _make_decahedral(n, spacing=0.72):
    """Generate a decahedral (5-fold twinned FCC) cluster.
    
    A decahedral cluster is formed by 5 tetrahedra of FCC atoms sharing a common
    central axis (the 5-fold axis). This is a well-known low-energy structure
    for transition-metal clusters.
    """
    a = spacing  # FCC lattice parameter
    # FCC positions within a tetrahedron (relative to one vertex)
    # We build 5 FCC tetrahedra rotated about the z-axis by 72 degrees.
    
    # Generate FCC points in a bounded box, then keep those forming compact clusters
    # Simpler approach: generate FCC lattice points, select n closest to origin,
    # then apply 5-fold twinning by rotating copies.
    
    # Build a small FCC cluster
    max_idx = int(spacing * n ** (1.0/3.0) / (a * 0.5)) + 2
    fcc_points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * a * 0.5
                    fcc_points.append(p)
                else:
                    p = (np.array([i, j, k]) + np.array([1, 1, 1])) * a * 0.25
                    fcc_points.append(p)
    fcc_points = np.array(fcc_points)
    
    # Select n points closest to origin
    d2 = np.sum(fcc_points**2, axis=1)
    order = np.argsort(d2)
    base = fcc_points[order[:n]]
    
    # Center the cluster
    base = base - np.mean(base, axis=0)
    
    # Apply 5-fold twinning: rotate by 72, 144, 216, 288 degrees about z-axis
    # and superimpose (take the union, then select n closest to center)
    all_points = [base]
    for angle_deg in [72, 144, 216, 288]:
        angle = np.radians(angle_deg)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        rotated = base @ R.T
        all_points.append(rotated)
    
    combined = np.vstack(all_points)
    # Select n points closest to origin
    d2 = np.sum(combined**2, axis=1)
    order = np.argsort(d2)
    result = combined[order[:n]]
    
    # Resolve any close contacts
    for _ in range(5):
        d = cdist(result, result)
        np.fill_diagonal(d, np.inf)
        mask = d < 0.36
        if not np.any(mask):
            break
        for i in range(len(result)):
            close = np.where(mask[i])[0]
            for j in close:
                if j <= i:
                    continue
                dist = d[i, j]
                if dist > 1e-10:
                    direction = (result[i] - result[j]) / dist
                    push = (0.36 - dist) / 2.0 + 0.005
                    result[i] += direction * push
                    result[j] -= direction * push
    
    return result

def _make_fcc(n, spacing=0.72):
    a = spacing
    r_max = spacing * n ** (1.0/3.0) * 0.85
    max_idx = int(r_max / (a * 0.5)) + 1
    candidates = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * a * 0.5
                    if np.linalg.norm(p) <= r_max:
                        candidates.append(p)
                else:
                    p = (np.array([i, j, k]) + np.array([1, 1, 1])) * a * 0.25
                    if np.linalg.norm(p) <= r_max:
                        candidates.append(p)
    if not candidates:
        return None
    candidates = np.array(candidates)
    d2 = np.sum(candidates**2, axis=1)
    order = np.argsort(d2)
    return candidates[order[:n]]

def _make_bcc(n, spacing=0.72):
    a = spacing
    r_max = spacing * n ** (1.0/3.0) * 0.85
    max_idx = int(r_max / a) + 1
    candidates = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                p = np.array([i, j, k]) * a
                if np.linalg.norm(p) <= r_max:
                    candidates.append(p)
                p2 = (np.array([i, j, k]) + np.array([0.5, 0.5, 0.5])) * a
                if np.linalg.norm(p2) <= r_max:
                    candidates.append(p2)
    if not candidates:
        return None
    candidates = np.array(candidates)
    d2 = np.sum(candidates**2, axis=1)
    order = np.argsort(d2)
    return candidates[order[:n]]

def _make_random_compact(n, spacing=0.72, seed=None, min_dist=0.5):
    """Random compact generation using vectorized distance checks."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.85
    points = []
    max_attempts = 2000
    
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        batch_size = min(50, n - len(points))
        batch = rng.uniform(-radius, radius, (batch_size, 3))
        batch = batch[np.linalg.norm(batch, axis=1) <= radius]
        
        for p in batch:
            if len(points) >= n:
                break
            if not points:
                points.append(p)
            else:
                arr = np.array(points)
                dists = np.linalg.norm(arr - p, axis=1)
                if np.min(dists) >= min_dist:
                    points.append(p)
    
    if len(points) < n:
        for _ in range(max_attempts):
            if len(points) >= n:
                break
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) > radius:
                continue
            if not points:
                points.append(p)
            else:
                arr = np.array(points)
                dists = np.linalg.norm(arr - p, axis=1)
                if np.min(dists) >= 0.36:
                    points.append(p)
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])

def _make_deformed(base_positions, spacing=0.72, seed=None, scale=0.05):
    rng = np.random.default_rng(seed)
    positions = base_positions.copy()
    noise = rng.normal(0, scale * spacing, positions.shape)
    positions += noise
    # Resolve collisions
    for _ in range(3):
        d = cdist(positions, positions)
        np.fill_diagonal(d, np.inf)
        mask = d < 0.36
        if not np.any(mask):
            break
        for i in range(len(positions)):
            close = np.where(mask[i])[0]
            for j in close:
                if j <= i:
                    continue
                dist = d[i, j]
                if dist > 0:
                    direction = (positions[i] - positions[j]) / dist
                    push = (0.36 - dist) / 2.0 + 0.01
                    positions[i] += direction * push
                    positions[j] -= direction * push
    return positions

def _basin_hopping_single(start_pos, potential, kind, spacing, rng,
                          best_energy, best_positions, kick_scale=0.15,
                          max_iter=30, t0=0.5, t1=0.01):
    """Single basin-hopping trajectory with adaptive kick. Returns (best_e, best_pos, cur_e, cur_pos)."""
    cur_e, cur_pos = relax(start_pos, potential, kind)
    if not np.isfinite(cur_e):
        return best_energy, best_positions, cur_e, cur_pos
    if cur_e < best_energy and _min_dist(cur_pos) >= 0.35:
        best_energy = cur_e
        best_positions = cur_pos.copy()
    
    for it in range(max_iter):
        if _time_left() < 3.0:
            break
        T = t0 * (t1 / t0) ** (it / max(1, max_iter - 1))
        # Adaptive kick: shrink as we converge
        current_kick = kick_scale * spacing * (0.7 ** (it // 10))
        noise = rng.normal(0, current_kick, cur_pos.shape)
        kicked = cur_pos + noise
        try:
            new_e, new_pos = relax(kicked, potential, kind)
        except Exception:
            continue
        if not np.isfinite(new_e):
            continue
        if new_e < best_energy and _min_dist(new_pos) >= 0.35:
            best_energy = new_e
            best_positions = new_pos.copy()
        if new_e < cur_e:
            cur_e, cur_pos = new_e, new_pos
        else:
            delta = new_e - cur_e
            if rng.uniform() < np.exp(-delta / max(T, 1e-10)):
                cur_e, cur_pos = new_e, new_pos
    return best_energy, best_positions, cur_e, cur_pos

def _round_robin_basin_hopping(seeds, potential, kind, spacing, rng,
                                best_energy, best_positions, time_budget=180.0):
    """Round-robin basin hopping: cycle through all seeds, giving each one
    a few iterations per round. This ensures diverse exploration early on,
    then focuses on the best basins as time runs out."""
    if not seeds:
        return best_energy, best_positions
    
    # Track current state per seed
    seed_states = []
    for label, start_pos in seeds:
        seed_states.append({
            'label': label,
            'cur_e': np.inf,
            'cur_pos': start_pos.copy(),
            'iters_done': 0,
        })
    
    # First pass: relax all seeds
    for st in seed_states:
        if _time_left() < 5.0:
            break
        try:
            e, pos = relax(st['cur_pos'], potential, kind)
            if np.isfinite(e):
                st['cur_e'] = e
                st['cur_pos'] = pos
                if e < best_energy and _min_dist(pos) >= 0.35:
                    best_energy = e
                    best_positions = pos.copy()
        except Exception:
            pass
    
    # Round-robin: each round, give each seed one kick+relax
    max_iter_per_seed = 25
    kick_scale = 0.15
    t0, t1 = 0.5, 0.01
    
    for round_num in range(max_iter_per_seed):
        if _time_left() < 5.0:
            break
        # Sort seeds by current energy (best first) so we spend more time on good ones
        valid_states = [st for st in seed_states if np.isfinite(st['cur_e'])]
        if not valid_states:
            break
        valid_states.sort(key=lambda s: s['cur_e'])
        
        for st in valid_states:
            if _time_left() < 3.0:
                break
            if st['iters_done'] >= max_iter_per_seed:
                continue
            
            it = st['iters_done']
            T = t0 * (t1 / t0) ** (it / max(1, max_iter_per_seed - 1))
            current_kick = kick_scale * spacing * (0.75 ** (it // 8))
            
            noise = rng.normal(0, current_kick, st['cur_pos'].shape)
            kicked = st['cur_pos'] + noise
            try:
                new_e, new_pos = relax(kicked, potential, kind)
            except Exception:
                st['iters_done'] += 1
                continue
            
            st['iters_done'] += 1
            
            if not np.isfinite(new_e):
                continue
            
            if new_e < best_energy and _min_dist(new_pos) >= 0.35:
                best_energy = new_e
                best_positions = new_pos.copy()
            
            if new_e < st['cur_e']:
                st['cur_e'] = new_e
                st['cur_pos'] = new_pos
            else:
                delta = new_e - st['cur_e']
                if rng.uniform() < np.exp(-delta / max(T, 1e-10)):
                    st['cur_e'] = new_e
                    st['cur_pos'] = new_pos
    
    return best_energy, best_positions

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    potential = instance.get("potential", "sutton_chen")
    spacing = 0.72
    
    best_energy = np.inf
    best_positions = None
    rng = np.random.default_rng(n + 42)
    
    def _try_relax(positions):
        nonlocal best_energy, best_positions
        if _time_left() < 2.0:
            return
        try:
            energy, relaxed = relax(positions, potential, kind)
            if np.isfinite(energy) and energy < best_energy:
                if _min_dist(relaxed) >= 0.35:
                    best_energy = energy
                    best_positions = relaxed.copy()
        except Exception:
            pass
    
    # Generate diverse seeds
    seeds = []
    
    # Primary: proper Mackay icosahedral
    try:
        mackay = _make_mackay(n, spacing)
        if mackay is not None and len(mackay) == n:
            seeds.append(("mackay", mackay))
    except Exception:
        pass
    
    # Decahedral (5-fold twinned FCC) — new structural family
    try:
        deca = _make_decahedral(n, spacing)
        if deca is not None and len(deca) == n:
            seeds.append(("decahedral", deca))
    except Exception:
        pass
    
    # FCC
    try:
        fcc = _make_fcc(n, spacing)
        if fcc is not None and len(fcc) == n:
            seeds.append(("fcc", fcc))
    except Exception:
        pass
    
    # BCC
    try:
        bcc = _make_bcc(n, spacing)
        if bcc is not None and len(bcc) == n:
            seeds.append(("bcc", bcc))
    except Exception:
        pass
    
    # Random compact seeds (fewer, since we have more structural seeds)
    for seed_val in [n, n + 1]:
        try:
            rnd = _make_random_compact(n, spacing, seed=seed_val, min_dist=0.5)
            if len(rnd) == n:
                seeds.append((f"rand_{seed_val}", rnd))
        except Exception:
            pass
    
    # Deformed Mackay variants
    if seeds:
        for k in range(2):
            try:
                deformed = _make_deformed(seeds[0][1], spacing, seed=n + 10 + k, scale=0.04)
                seeds.append((f"deform_mackay_{k}", deformed))
            except Exception:
                pass
    
    # Relax all seeds first to get initial best
    for label, pos in seeds:
        if _time_left() < 4.0:
            break
        _try_relax(pos)
    
    # Round-robin basin hopping with remaining time
    if best_positions is not None and _time_left() > 15.0:
        time_budget = min(_time_left() - 10.0, 200.0)
        best_energy, best_positions = _round_robin_basin_hopping(
            seeds, potential, kind, spacing, rng,
            best_energy, best_positions, time_budget
        )
    
    # Final polish: a few more relaxations from the best position with small kicks
    if best_positions is not None and np.all(np.isfinite(best_positions)) and _time_left() > 8.0:
        for polish_kick in [0.03, 0.02, 0.01]:
            if _time_left() < 4.0:
                break
            try:
                noise = rng.normal(0, polish_kick * spacing, best_positions.shape)
                kicked = best_positions + noise
                energy, relaxed = relax(kicked, potential, kind)
                if np.isfinite(energy) and energy < best_energy and _min_dist(relaxed) >= 0.35:
                    best_energy = energy
                    best_positions = relaxed.copy()
            except Exception:
                pass
        # Final tight relaxation
        try:
            energy, relaxed = relax(best_positions, potential, kind)
            if np.isfinite(energy) and _min_dist(relaxed) >= 0.35:
                best_positions = relaxed
        except Exception:
            pass
    
    # Fallback
    if best_positions is None or not np.all(np.isfinite(best_positions)):
        positions = np.zeros((n, 3))
        for i in range(n):
            theta = i * 2.399
            r = spacing * 0.3 * np.sqrt(i)
            positions[i] = [r * np.cos(theta), r * np.sin(theta), 0.1 * i * 0.05]
        try:
            energy, relaxed = relax(positions, potential, kind)
            if np.isfinite(energy):
                best_positions = relaxed
        except Exception:
            best_positions = positions
    
    if not np.all(np.isfinite(best_positions)):
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
