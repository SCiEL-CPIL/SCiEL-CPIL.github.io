# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0

def _remaining():
    return _DEADLINE - time.time()

def _get_spacing(kind):
    if kind == '12-6':
        n, m, c = 12, 6, 144.41
    elif kind == '9-6':
        n, m, c = 9, 6, 39.432
    elif kind == '10-8':
        n, m, c = 10, 8, 34.408
    else:
        return 1.0
    return (2 * n / (c * m)) ** (1.0 / (n - m / 2.0))

def _fcc_positions(n, a=1.0):
    basis = np.array([[0,0,0],[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]]) * a
    positions = []
    max_idx = int(np.ceil(n**(1/3)) + 4)
    sites = []
    for i in range(-max_idx, max_idx+1):
        for j in range(-max_idx, max_idx+1):
            for k in range(-max_idx, max_idx+1):
                for b in basis:
                    pos = np.array([i,j,k])*a + b
                    sites.append((np.linalg.norm(pos), pos))
    sites.sort(key=lambda x: x[0])
    seen = set()
    for d, pos in sites:
        key = tuple(np.round(pos, 6))
        if key not in seen:
            seen.add(key)
            positions.append(pos)
            if len(positions) >= n:
                break
    return np.array(positions[:n])

def _hcp_positions(n, a=1.0):
    c = a * np.sqrt(8/3)
    positions = []
    max_idx = int(np.ceil(n**(1/3)) + 4)
    sites = []
    for layer in range(-max_idx, max_idx+1):
        z = layer * c / 2
        for i in range(-max_idx, max_idx+1):
            for j in range(-max_idx, max_idx+1):
                sites.append((np.linalg.norm(pos), pos))
                pos = np.array([i*a, j*a*np.sqrt(3)/2, z])
                sites[-1] = (np.linalg.norm(pos), pos)
        for i in range(-max_idx, max_idx+1):
            for j in range(-max_idx, max_idx+1):
                pos = np.array([(i+0.5)*a, (j+1/6)*a*np.sqrt(3), z+c/2])
                sites.append((np.linalg.norm(pos), pos))
    sites.sort(key=lambda x: x[0])
    seen = set()
    for d, pos in sites:
        key = tuple(np.round(pos, 6))
        if key not in seen:
            seen.add(key)
            positions.append(pos)
            if len(positions) >= n:
                break
    return np.array(positions[:n])

def _bcc_positions(n, a=1.0):
    basis = np.array([[0,0,0],[0.5,0.5,0.5]]) * a
    positions = []
    max_idx = int(np.ceil(n**(1/3)) + 4)
    sites = []
    for i in range(-max_idx, max_idx+1):
        for j in range(-max_idx, max_idx+1):
            for k in range(-max_idx, max_idx+1):
                for b in basis:
                    pos = np.array([i,j,k])*a + b
                    sites.append((np.linalg.norm(pos), pos))
    sites.sort(key=lambda x: x[0])
    seen = set()
    for d, pos in sites:
        key = tuple(np.round(pos, 6))
        if key not in seen:
            seen.add(key)
            positions.append(pos)
            if len(positions) >= n:
                break
    return np.array(positions[:n])

def _icosahedral_positions(n, a=1.0):
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0,1,phi],[0,-1,phi],[0,1,-phi],[0,-1,-phi],
        [1,phi,0],[-1,phi,0],[1,-phi,0],[-1,-phi,0],
        [phi,0,1],[phi,0,-1],[-phi,0,1],[-phi,0,-1]
    ]) * a / np.sqrt(1+phi**2)
    positions = [np.zeros(3)]
    for v in verts:
        positions.append(v)
    if n > 13:
        faces = [
            [0,1,8],[0,2,9],[1,3,8],[2,4,9],[1,0,5],[3,2,6],
            [0,4,5],[2,6,7],[5,8,10],[6,9,11],[8,1,10],[9,3,11],
            [10,5,7],[11,6,4],[8,10,0],[9,11,2],[5,7,1],[6,4,3],
            [10,8,0],[11,9,2]
        ]
        for f in faces:
            if len(positions) >= n:
                break
            fc = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
            positions.append(fc)
        if n > 33:
            edges = []
            for i in range(12):
                for j in range(i+1, 12):
                    d = np.linalg.norm(verts[i]-verts[j])
                    if d < 2.1*a/np.sqrt(1+phi**2):
                        edges.append((i,j))
            for e in edges:
                if len(positions) >= n:
                    break
                em = (verts[e[0]] + verts[e[1]]) / 2.0
                positions.append(em)
    while len(positions) < n:
        idx = np.random.randint(0, 12)
        pos = verts[idx] + np.random.normal(0, 0.1*a, 3)
        positions.append(pos)
    return np.array(positions[:n])

def _random_cluster(n, spacing, seed, min_dist=None):
    if min_dist is None:
        min_dist = spacing * 0.9
    rng = np.random.default_rng(seed)
    radius = spacing * n**(1.0/3.0) * 0.9
    points = []
    max_attempts = 5000
    while len(points) < n and max_attempts > 0:
        max_attempts -= 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p-q) >= min_dist for q in points):
            points.append(p)
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)

def _molten_cluster(n, spacing, seed, temp_scale=1.0):
    rng = np.random.default_rng(seed)
    radius = spacing * n**(1.0/3.0) * 1.2
    points = []
    min_d = spacing * 0.5 * temp_scale
    max_attempts = 10000
    while len(points) < n and max_attempts > 0:
        max_attempts -= 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p-q) >= min_d for q in points):
            points.append(p)
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)

def _deformed_fcc_positions(n, a=1.0, seed=0):
    rng = np.random.default_rng(seed)
    pos = _fcc_positions(n, a)
    shear = rng.uniform(-0.05, 0.05, (3,3))
    np.fill_diagonal(shear, 0)
    return pos @ (np.eye(3) + shear)

def _incremental_energy(pos_new, positions, kind):
    """Compute the incremental energy contribution of adding pos_new to positions."""
    if kind == '12-6':
        n_exp, m_exp, c = 12, 6, 144.41
    elif kind == '9-6':
        n_exp, m_exp, c = 9, 6, 39.432
    elif kind == '10-8':
        n_exp, m_exp, c = 10, 8, 34.408
    else:
        return 0.0
    
    if len(positions) == 0:
        return 0.0
    
    dists = np.linalg.norm(pos_new - positions, axis=1)
    dists = np.maximum(dists, 1e-10)
    r_n = np.sum(dists**(-n_exp))
    r_m = np.sum(dists**(-m_exp))
    return 0.5 * r_n - c * np.sqrt(r_m)

def _gradient_guided_insertion(n, spacing, kind, seed=0):
    """
    Build cluster by sequentially inserting atoms at positions that
    minimize the incremental Sutton-Chen energy contribution.
    Uses a local grid search around existing surface atoms.
    """
    rng = np.random.default_rng(seed)
    positions = [np.zeros(3)]
    
    # Seed first few atoms with a small grid search
    for i in range(1, min(n, 6)):
        if _remaining() < 30.0:
            break
        best_pos = None
        best_e = np.inf
        
        # Search shell around each existing atom
        for existing in positions:
            for dx in np.linspace(-spacing*0.8, spacing*0.8, 5):
                for dy in np.linspace(-spacing*0.8, spacing*0.8, 5):
                    for dz in np.linspace(-spacing*0.8, spacing*0.8, 5):
                        cand = existing + np.array([dx, dy, dz])
                        if np.linalg.norm(cand) < 0.36:
                            continue
                        min_d = min(np.linalg.norm(cand - p) for p in positions)
                        if min_d < 0.36:
                            continue
                        e = _incremental_energy(cand, positions, kind)
                        if e < best_e:
                            best_e = e
                            best_pos = cand
        
        if best_pos is None:
            best_pos = positions[-1] + np.array([spacing, 0, 0])
        positions.append(best_pos)
    
    # Grow by adding atoms near the surface using incremental energy
    while len(positions) < n:
        if _remaining() < 30.0:
            break
        
        centroid = np.mean(positions, axis=0)
        dists_from_center = np.array([np.linalg.norm(p - centroid) for p in positions])
        threshold = np.percentile(dists_from_center, 70)
        surface_indices = [i for i, d in enumerate(dists_from_center) if d >= threshold]
        if not surface_indices:
            surface_indices = list(range(len(positions)))
        
        best_pos = None
        best_e = np.inf
        
        n_trials = 15
        for _ in range(n_trials):
            anchor_idx = surface_indices[rng.integers(0, len(surface_indices))]
            anchor = positions[anchor_idx]
            direction = anchor - centroid
            dir_norm = np.linalg.norm(direction)
            if dir_norm < 1e-6:
                direction = rng.normal(0, 1, 3)
                dir_norm = np.linalg.norm(direction)
            direction /= dir_norm
            direction += rng.normal(0, 0.3, 3)
            direction /= np.linalg.norm(direction)
            
            dist = spacing * (0.85 + 0.3 * rng.random())
            p = anchor + direction * dist
            
            min_d = min(np.linalg.norm(p - q) for q in positions)
            if min_d < 0.36:
                continue
            
            e = _incremental_energy(p, positions, kind)
            if e < best_e:
                best_e = e
                best_pos = p
        
        if best_pos is None:
            for _ in range(10):
                p = rng.uniform(-spacing*2, spacing*2, 3)
                if all(np.linalg.norm(p-q) >= 0.36 for q in positions):
                    best_pos = p
                    break
            if best_pos is None:
                best_pos = positions[-1] + rng.normal(0, spacing*0.5, 3)
        
        positions.append(best_pos)
    
    return np.array(positions[:n])

def _min_distance_check(pos):
    n = len(pos)
    if n < 2:
        return np.inf
    dists = np.linalg.norm(pos[:, np.newaxis] - pos[np.newaxis, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists)

def _valid_result(energy, pos):
    if not np.all(np.isfinite(pos)):
        return False
    if _min_distance_check(pos) < 0.35:
        return False
    return True

def _relax_task(args):
    pos, potential, kind = args
    try:
        energy, relaxed_pos = relax(pos, potential, kind)
        if _valid_result(energy, relaxed_pos):
            return energy, relaxed_pos
        else:
            return np.inf, relaxed_pos
    except Exception:
        return np.inf, pos

def _simulated_annealing(pos, potential, kind, n_steps=200, t_start=1.0, t_end=0.01, seed=0):
    rng = np.random.default_rng(seed)
    current_pos = pos.copy()
    current_e, _ = sutton_chen_energy_gradient(current_pos, kind)
    if not np.isfinite(current_e):
        return np.inf, pos
    best_pos = current_pos.copy()
    best_e = current_e
    n = len(current_pos)
    
    for step in range(n_steps):
        if _remaining() < 15.0:
            break
        t = t_start * (t_end / t_start) ** (step / max(n_steps-1, 1))
        kick_size = t * 0.5
        new_pos = current_pos + rng.normal(0, kick_size, current_pos.shape)
        if not np.all(np.isfinite(new_pos)):
            continue
        if _min_distance_check(new_pos) < 0.35:
            continue
        try:
            new_e, _ = sutton_chen_energy_gradient(new_pos, kind)
        except Exception:
            continue
        if not np.isfinite(new_e):
            continue
        dE = new_e - current_e
        if dE < 0 or rng.random() < np.exp(-dE / max(t, 1e-10)):
            current_pos = new_pos
            current_e = new_e
            if current_e < best_e:
                best_e = current_e
                best_pos = current_pos.copy()
    
    try:
        final_e, final_pos = relax(best_pos, potential, kind)
        if _valid_result(final_e, final_pos):
            return final_e, final_pos
    except Exception:
        pass
    return best_e, best_pos

def construct(instance):
    global _START_TIME, _DEADLINE
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    spacing = _get_spacing(kind)
    
    candidates = []
    
    # 1. Gradient-guided insertion with energy-based scoring
    for seed in [0, 42, 123]:
        try:
            if _remaining() < 60.0:
                break
            gg = _gradient_guided_insertion(n, spacing, kind, seed)
            if len(gg) == n and _min_distance_check(gg) > 0.36:
                candidates.append(gg)
        except Exception:
            pass
    
    # 2-5. Standard geometries
    for gen_func in [_fcc_positions, _hcp_positions, _bcc_positions, _icosahedral_positions]:
        try:
            pos = gen_func(n, spacing)
            if len(pos) == n and _min_distance_check(pos) > 0.36:
                candidates.append(pos)
        except Exception:
            pass
    
    # 6. Deformed FCC
    try:
        dfcc = _deformed_fcc_positions(n, spacing, seed=7)
        if len(dfcc) == n and _min_distance_check(dfcc) > 0.36:
            candidates.append(dfcc)
    except Exception:
        pass
    
    # 7-10. Perturbed geometries
    for base_func, seed, scale in [(_fcc_positions, 1, 0.08), (_hcp_positions, 2, 0.08),
                                     (_fcc_positions, 3, 0.2), (_hcp_positions, 4, 0.2)]:
        try:
            base = base_func(n, spacing)
            rng = np.random.default_rng(seed)
            noise = rng.normal(0, spacing*scale, base.shape)
            perturbed = base + noise
            if _min_distance_check(perturbed) > 0.36:
                candidates.append(perturbed)
        except Exception:
            pass
    
    # 11-14. Molten clusters
    for seed, ts in [(555, 1.0), (666, 1.5), (777, 0.8), (888, 2.0)]:
        try:
            molten = _molten_cluster(n, spacing, seed, temp_scale=ts)
            if _min_distance_check(molten) > 0.36:
                candidates.append(molten)
        except Exception:
            pass
    
    # 15-16. Random clusters
    for seed in [42, 123]:
        try:
            rand = _random_cluster(n, spacing, seed)
            if _min_distance_check(rand) > 0.36:
                candidates.append(rand)
        except Exception:
            pass
    
    valid_candidates = [c for c in candidates if len(c) == n and np.all(np.isfinite(c)) and _min_distance_check(c) > 0.36]
    if not valid_candidates:
        valid_candidates = [_random_cluster(n, spacing, 999, min_dist=0.4)]
    
    best_energy = np.inf
    best_positions = None
    
    # Phase 1: Parallel relaxation
    if _remaining() > 20.0:
        tasks = [(pos, potential, kind) for pos in valid_candidates]
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_relax_task, t): t for t in tasks}
            for future in as_completed(futures):
                if _remaining() < 5.0:
                    break
                try:
                    energy, pos = future.result()
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = pos
                except Exception:
                    pass
    
    if best_positions is None:
        for pos in valid_candidates:
            if _remaining() < 10.0:
                break
            try:
                energy, relaxed_pos = relax(pos, potential, kind)
                if _valid_result(energy, relaxed_pos) and energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed_pos
            except Exception:
                pass
    
    # Phase 2: Iterative perturbation + SA
    if best_positions is not None and _remaining() > 30.0:
        rng = np.random.default_rng(999)
        iteration = 0
        no_improve = 0
        while _remaining() > 40.0:
            iteration += 1
            base_kick = spacing * 0.6 * (0.7 ** iteration)
            kicks = [base_kick * s for s in [0.3, 0.5, 0.8, 1.0, 1.4, 1.8]]
            tasks = []
            for kick in kicks:
                noise = rng.normal(0, kick, best_positions.shape)
                tasks.append((best_positions + noise, potential, kind))
            
            improved = False
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(_relax_task, t): t for t in tasks}
                for future in as_completed(futures):
                    if _remaining() < 10.0:
                        break
                    try:
                        energy, pos = future.result()
                        if energy < best_energy:
                            best_energy = energy
                            best_positions = pos
                            improved = True
                    except Exception:
                        pass
            
            if _remaining() > 50.0:
                try:
                    sa_e, sa_pos = _simulated_annealing(
                        best_positions, potential, kind,
                        n_steps=150, t_start=0.8, t_end=0.05,
                        seed=iteration*100+42)
                    if sa_e < best_energy:
                        best_energy = sa_e
                        best_positions = sa_pos
                        improved = True
                except Exception:
                    pass
            
            if not improved:
                no_improve += 1
                if no_improve >= 3:
                    break
            else:
                no_improve = 0
    
    # Phase 3: Additional diversification
    if best_positions is not None and _remaining() > 50.0:
        rng2 = np.random.default_rng(2024)
        for kick_scale in [0.3, 0.5, 0.8]:
            if _remaining() < 20.0:
                break
            noise = rng2.normal(0, spacing*kick_scale, best_positions.shape)
            perturbed = best_positions + noise
            try:
                energy, pos = _relax_task((perturbed, potential, kind))
                if energy < best_energy:
                    best_energy = energy
                    best_positions = pos
            except Exception:
                pass
        
        if _remaining() > 60.0:
            for sa_seed in [777, 888]:
                if _remaining() < 25.0:
                    break
                noise = rng2.normal(0, spacing*0.6, best_positions.shape)
                perturbed = best_positions + noise
                try:
                    sa_e, sa_pos = _simulated_annealing(
                        perturbed, potential, kind,
                        n_steps=100, t_start=0.5, t_end=0.03,
                        seed=sa_seed)
                    if sa_e < best_energy:
                        best_energy = sa_e
                        best_positions = sa_pos
                except Exception:
                    pass
    
    # Phase 4: Final polish
    if best_positions is not None and _remaining() > 10.0:
        try:
            energy, relaxed_pos = relax(best_positions, potential, kind)
            if _valid_result(energy, relaxed_pos) and energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos
        except Exception:
            pass
    
    if best_positions is None:
        try:
            pos = _random_cluster(n, spacing, 12345, min_dist=0.4)
            energy, relaxed_pos = relax(pos, potential, kind)
            if _valid_result(energy, relaxed_pos):
                best_positions = relaxed_pos
            else:
                best_positions = np.zeros((n, 3))
                for i in range(n):
                    best_positions[i] = [i*0.8, 0, 0]
        except Exception:
            best_positions = np.zeros((n, 3))
            for i in range(n):
                best_positions[i] = [i*0.8, 0, 0]
    
    return best_positions
# EVOLVE-BLOCK-END
