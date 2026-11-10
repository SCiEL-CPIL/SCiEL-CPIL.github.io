# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = _START_TIME + 285.0

def _remaining():
    return _DEADLINE - time.time()

def _get_params(kind):
    if kind == '12-6':
        return 12, 6, 144.41
    elif kind == '9-6':
        return 9, 6, 39.432
    elif kind == '10-8':
        return 10, 8, 34.408
    return 12, 6, 144.41

def _get_spacing(kind):
    n_exp, m_exp, c = _get_params(kind)
    return (2 * n_exp / (c * m_exp)) ** (1.0 / (n_exp - m_exp / 2.0))

def _fcc_positions(n, a=1.0):
    basis = np.array([[0,0,0],[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]]) * a
    max_idx = int(np.ceil(n**(1/3))) + 4
    sites = []
    for i in range(-max_idx, max_idx+1):
        for j in range(-max_idx, max_idx+1):
            for k in range(-max_idx, max_idx+1):
                for b in basis:
                    pos = np.array([i,j,k])*a + b
                    sites.append((np.linalg.norm(pos), pos))
    sites.sort(key=lambda x: x[0])
    seen = set()
    positions = []
    for d, pos in sites:
        key = tuple(np.round(pos, 6))
        if key not in seen:
            seen.add(key)
            positions.append(pos)
            if len(positions) >= n:
                break
    return np.array(positions[:n])

def _min_distance(pos):
    if len(pos) < 2:
        return np.inf
    d = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return np.min(d)

def _valid(energy, pos):
    if not np.all(np.isfinite(pos)):
        return False
    if _min_distance(pos) < 0.35:
        return False
    return True

def _relax_one(pos, potential, kind):
    try:
        e, p = relax(pos, potential, kind)
        if _valid(e, p):
            return e, p
    except Exception:
        pass
    return np.inf, pos

def _incremental_energy(existing, new_atom, kind):
    """Energy contribution of adding new_atom to existing cluster."""
    n_exp, m_exp, c = _get_params(kind)
    if len(existing) == 0:
        return 0.0
    d = existing - new_atom
    r2 = np.sum(d*d, axis=1)
    r2 = np.maximum(r2, 1e-12)
    r_inv_n = r2 ** (-n_exp/2.0)
    r_inv_m = r2 ** (-m_exp/2.0)
    pair_sum = np.sum(r_inv_n)
    coh_sum = np.sum(r_inv_m)
    return 0.5 * pair_sum - c * np.sqrt(coh_sum)

def _find_best_insertion(existing, kind, spacing, rng, n_candidates=12):
    """Find the best position to insert a new atom given existing cluster."""
    if len(existing) == 0:
        return np.zeros(3)
    
    best_pos = None
    best_e = np.inf
    
    existing_arr = np.array(existing)
    
    # Generate candidate positions: near existing atoms (surface sites)
    candidates = []
    
    # For each existing atom, try positions in its coordination shell
    for _ in range(n_candidates):
        anchor_idx = rng.integers(0, len(existing_arr))
        anchor = existing_arr[anchor_idx]
        direction = rng.normal(0, 1, 3)
        direction /= (np.linalg.norm(direction) + 1e-12)
        dist = spacing * rng.uniform(0.85, 1.15)
        candidate = anchor + direction * dist
        
        # Check minimum distance
        dists = np.linalg.norm(existing_arr - candidate, axis=1)
        if np.min(dists) < 0.36:
            continue
        candidates.append(candidate)
    
    # Also try some positions in interstitial sites (between pairs of atoms)
    if len(existing_arr) >= 2:
        for _ in range(n_candidates // 2):
            i1 = rng.integers(0, len(existing_arr))
            i2 = rng.integers(0, len(existing_arr))
            if i1 == i2:
                continue
            mid = (existing_arr[i1] + existing_arr[i2]) / 2.0
            offset = rng.normal(0, spacing * 0.3, 3)
            candidate = mid + offset
            dists = np.linalg.norm(existing_arr - candidate, axis=1)
            if np.min(dists) < 0.36:
                continue
            candidates.append(candidate)
    
    # Also try a few random shell positions
    for _ in range(4):
        r = spacing * rng.uniform(0.8, 2.0)
        theta = rng.uniform(0, 2*np.pi)
        phi = rng.uniform(0, np.pi)
        candidate = np.array([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
        ])
        dists = np.linalg.norm(existing_arr - candidate, axis=1)
        if np.min(dists) < 0.36:
            continue
        candidates.append(candidate)
    
    if len(candidates) == 0:
        # Fallback: random position
        candidate = rng.normal(0, spacing, 3)
        candidates = [candidate]
    
    # Evaluate incremental energy for each candidate
    for cand in candidates:
        e = _incremental_energy(existing_arr, cand, kind)
        if e < best_e:
            best_e = e
            best_pos = cand
    
    if best_pos is None:
        best_pos = candidates[0]
    
    return best_pos

def _gradient_refine_insertion(existing, new_atom, kind, spacing, rng, max_iters=50):
    """Refine the position of new_atom using gradient of incremental energy."""
    n_exp, m_exp, c = _get_params(kind)
    existing_arr = np.array(existing)
    pos = new_atom.copy()
    
    lr = spacing * 0.05
    for it in range(max_iters):
        if _remaining() < 20.0:
            break
        # Compute gradient of incremental energy w.r.t. new_atom position
        d = existing_arr - pos
        r2 = np.sum(d*d, axis=1)
        r2 = np.maximum(r2, 1e-12)
        r = np.sqrt(r2)
        
        # dE/dr for pair part: -n * r^(-n-1)
        # dE/dr for cohesion: -c/2 * m * r^(-m/2 - 1) / sqrt(coh_sum)
        r_inv_n = r ** (-n_exp)
        r_inv_m = r ** (-m_exp/2.0)
        coh_sum = np.sum(r_inv_m)
        coh_sqrt = np.sqrt(max(coh_sum, 1e-12))
        
        # Gradient of incremental energy w.r.t. new_atom position
        # dE/dx_new = sum over existing atoms of dE/dr * dr/dx_new
        # dr/dx_new = -(x_new - x_i)/r = d/r (where d = x_i - x_new)
        
        pair_grad = -n_exp * r_inv_n * (d / r[:, None])  # shape (n, 3)
        coh_grad = (-c/2.0) * (m_exp/2.0) * r_inv_m * (d / r[:, None]) / coh_sqrt
        
        grad = np.sum(pair_grad + coh_grad, axis=0)
        
        # Check min distance constraint
        dists = np.linalg.norm(existing_arr - pos, axis=1)
        min_d = np.min(dists)
        
        # Gradient step
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        pos_new = pos - lr * grad / max(grad_norm, 1e-10)
        
        # Enforce minimum distance
        dists_new = np.linalg.norm(existing_arr - pos_new, axis=1)
        if np.min(dists_new) < 0.36:
            # Push away from closest atom
            closest_idx = np.argmin(dists_new)
            push_dir = pos_new - existing_arr[closest_idx]
            push_dir /= (np.linalg.norm(push_dir) + 1e-12)
            pos_new = existing_arr[closest_idx] + push_dir * 0.37
            dists_new = np.linalg.norm(existing_arr - pos_new, axis=1)
            if np.min(dists_new) < 0.36:
                # Reject this step
                lr *= 0.5
                continue
        
        # Check if energy improved
        e_old = _incremental_energy(existing_arr, pos, kind)
        e_new = _incremental_energy(existing_arr, pos_new, kind)
        
        if e_new < e_old:
            pos = pos_new
            lr = min(lr * 1.1, spacing * 0.2)
        else:
            lr *= 0.7
    
    return pos

def _build_cluster_greedy(n, kind, spacing, seed=0, use_gradient_refine=True):
    """Build cluster atom-by-atom using incremental energy minimization."""
    rng = np.random.default_rng(seed)
    positions = [np.zeros(3)]
    
    for count in range(1, n):
        if _remaining() < 25.0:
            # Not enough time: fill remaining with random positions
            for _ in range(n - count):
                p = rng.normal(0, spacing * 1.5, 3)
                positions.append(p)
            break
        
        # Find best insertion position
        new_atom = _find_best_insertion(positions, kind, spacing, rng, n_candidates=16)
        
        # Refine with gradient if time permits
        if use_gradient_refine and _remaining() > 30.0:
            new_atom = _gradient_refine_insertion(positions, new_atom, kind, spacing, rng, max_iters=30)
        
        positions.append(new_atom)
    
    return np.array(positions[:n])

def _random_cluster(n, spacing, seed, min_dist=None):
    if min_dist is None:
        min_dist = spacing * 0.85
    rng = np.random.default_rng(seed)
    radius = spacing * n**(1.0/3.0) * 1.0
    points = []
    max_attempts = 8000
    while len(points) < n and max_attempts > 0:
        max_attempts -= 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius*1.5, radius*1.5, 3)
            points.append(p)
    return np.array(points)

def _molten_cluster(n, spacing, seed):
    rng = np.random.default_rng(seed)
    radius = spacing * n**(1.0/3.0) * 1.1
    points = []
    min_d = spacing * 0.55
    for _ in range(n):
        p = rng.normal(0, radius/2.5, 3)
        attempts = 0
        while any(np.linalg.norm(p - q) < min_d for q in points) and attempts < 30:
            p = rng.normal(0, radius/2.5, 3)
            attempts += 1
        points.append(p)
    return np.array(points)

def _basin_hop(energy, positions, kind, potential, rng, max_kicks=12):
    """Basin hopping: kick + relax, keep best."""
    best_e = energy
    best_p = positions.copy()
    spacing = _get_spacing(kind)
    
    kick_base = spacing * 0.5
    for kick_iter in range(max_kicks):
        if _remaining() < 20.0:
            break
        
        # Adaptive kick size: start large, decrease
        kick_size = kick_base * (0.85 ** kick_iter)
        
        # Generate multiple kicks in parallel
        n_kicks = min(4, max(1, int(_remaining() / 15)))
        tasks = []
        for _ in range(n_kicks):
            noise = rng.normal(0, kick_size, positions.shape)
            perturbed = positions + noise
            tasks.append((perturbed, potential, kind))
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_relax_one, t[0], t[1], t[2]) for t in tasks]
            for fut in as_completed(futures):
                if _remaining() < 10.0:
                    break
                try:
                    e, p = fut.result()
                    if e < best_e - 1e-10:
                        best_e = e
                        best_p = p
                except Exception:
                    pass
    
    return best_e, best_p

def construct(instance):
    global _START_TIME, _DEADLINE
    # Reset deadline per call if needed (but keep global for multi-call scenarios)
    
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    spacing = _get_spacing(kind)
    
    candidates = []
    
    # --- Phase 1: Generate diverse initial candidates ---
    
    # A) Gradient-guided greedy construction (primary strategy)
    if _remaining() > 80.0:
        for seed in [0, 1, 2]:
            try:
                grown = _build_cluster_greedy(n, kind, spacing, seed, use_gradient_refine=True)
                if len(grown) == n:
                    candidates.append(grown)
            except Exception:
                pass
    
    # B) Quick greedy without gradient refinement (faster)
    if _remaining() > 50.0:
        for seed in [10, 11]:
            try:
                grown = _build_cluster_greedy(n, kind, spacing, seed, use_gradient_refine=False)
                if len(grown) == n:
                    candidates.append(grown)
            except Exception:
                pass
    
    # C) Geometric seeds (FCC, HCP, icosahedral)
    try:
        fcc = _fcc_positions(n, spacing)
        if len(fcc) == n:
            candidates.append(fcc)
            # Perturbed FCC
            rng_f = np.random.default_rng(100)
            candidates.append(fcc + rng_f.normal(0, spacing*0.12, fcc.shape))
    except Exception:
        pass
    
    try:
        # Simple icosahedral seed
        phi = (1 + np.sqrt(5))/2
        verts = np.array([
            [0,1,phi],[0,-1,phi],[0,1,-phi],[0,-1,-phi],
            [1,phi,0],[-1,phi,0],[1,-phi,0],[-1,-phi,0],
            [phi,0,1],[phi,0,-1],[-phi,0,1],[-phi,0,-1]
        ]) * spacing / np.sqrt(1+phi**2)
        ico_pos = [np.zeros(3)] + [v.copy() for v in verts]
        # Add face centers if needed
        if n > 13:
            faces = [[0,1,8],[0,2,9],[1,3,8],[2,4,9],[1,0,5],[3,2,6],
                     [0,4,5],[2,6,7],[5,8,10],[6,9,11],[8,1,10],[9,3,11],
                     [10,5,7],[11,6,4],[8,10,0],[9,11,2],[5,7,1],[6,4,3],
                     [10,8,0],[11,9,2]]
            for f in faces:
                if len(ico_pos) >= n:
                    break
                fc = (verts[f[0]] + verts[f[1]] + verts[f[2]])/3.0
                ico_pos.append(fc)
        while len(ico_pos) < n:
            ico_pos.append(verts[rng_f.integers(0,12)].copy() + rng_f.normal(0, spacing*0.1, 3))
        candidates.append(np.array(ico_pos[:n]))
    except Exception:
        pass
    
    # D) Random clusters
    for seed in [42, 123]:
        try:
            rand = _random_cluster(n, spacing, seed)
            candidates.append(rand)
        except Exception:
            pass
    
    # E) Molten cluster
    try:
        molten = _molten_cluster(n, spacing, 77)
        candidates.append(molten)
    except Exception:
        pass
    
    # Ensure minimum number of candidates
    while len(candidates) < 6:
        try:
            rand = _random_cluster(n, spacing, np.random.randint(1000, 9999))
            candidates.append(rand)
        except Exception:
            break
    
    # --- Phase 2: Parallel relaxation of all candidates ---
    best_energy = np.inf
    best_positions = None
    
    if _remaining() > 25.0:
        tasks = [(pos, potential, kind) for pos in candidates]
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_relax_one, t[0], t[1], t[2]): t for t in tasks}
            for fut in as_completed(futures):
                if _remaining() < 8.0:
                    break
                try:
                    e, p = fut.result()
                    if e < best_energy:
                        best_energy = e
                        best_positions = p
                except Exception:
                    pass
    
    # Fallback sequential relaxation if nothing worked
    if best_positions is None:
        for pos in candidates:
            if _remaining() < 10.0:
                break
            e, p = _relax_one(pos, potential, kind)
            if e < best_energy:
                best_energy = e
                best_positions = p
    
    if best_positions is None:
        # Last resort
        best_positions = _random_cluster(n, spacing, 99999, min_dist=0.4)
        best_energy = np.inf
        try:
            e, p = relax(best_positions, potential, kind)
            if _valid(e, p):
                best_energy = e
                best_positions = p
        except Exception:
            pass
    
    # --- Phase 3: Basin hopping refinement ---
    if best_positions is not None and _remaining() > 40.0:
        rng_bh = np.random.default_rng(12345)
        max_kicks = max(1, int((_remaining() - 25) / 12))
        best_energy, best_positions = _basin_hop(best_energy, best_positions, kind, potential, rng_bh, max_kicks=max_kicks)
    
    # --- Phase 4: Second round of basin hopping with smaller kicks ---
    if best_positions is not None and _remaining() > 30.0:
        rng_bh2 = np.random.default_rng(54321)
        max_kicks2 = max(1, int((_remaining() - 15) / 10))
        best_energy, best_positions = _basin_hop(best_energy, best_positions, kind, potential, rng_bh2, max_kicks=max_kicks2)
    
    # --- Phase 5: Final polish ---
    if best_positions is not None and _remaining() > 10.0:
        try:
            e, p = relax(best_positions, potential, kind)
            if _valid(e, p) and e < best_energy:
                best_energy = e
                best_positions = p
        except Exception:
            pass
    
    # Final validity check
    if best_positions is None or not _valid(best_energy, best_positions):
        # Construct a valid fallback
        best_positions = np.zeros((n, 3))
        for i in range(n):
            best_positions[i] = [i * 0.5, (i % 3) * 0.3, (i % 5) * 0.2]
        # Ensure min distance
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(best_positions[i] - best_positions[j])
                if d < 0.36:
                    best_positions[j] = best_positions[i] + np.array([0.36, 0.1, 0.05])
    
    return best_positions
# EVOLVE-BLOCK-END
