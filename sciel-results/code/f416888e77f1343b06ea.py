"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
TIME_LIMIT = 285.0  # Leave 45s safety margin for 330s limit


def _check_time():
    return time.time() - _START_TIME < TIME_LIMIT


def _get_icosahedral_points():
    """
    Returns the 12 vertices, 20 face centers, and 30 edge midpoints of a regular icosahedron.
    All points are normalized to unit radius.
    """
    phi = (1 + np.sqrt(5)) / 2
    
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    
    norm = np.linalg.norm(v[0])
    v = v / norm
    
    dists = np.linalg.norm(v[:, np.newaxis, :] - v[np.newaxis, :, :], axis=2)
    min_dist = np.min(dists)
    
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if np.isclose(dists[i, j], min_dist):
                edges.append((i, j))
    
    face_centers = []
    seen_faces = set()
    for i, j in edges:
        for k in range(12):
            if k == i or k == j:
                continue
            edge_ik = (i, k) in edges or (k, i) in edges
            edge_jk = (j, k) in edges or (k, j) in edges
            if edge_ik and edge_jk:
                face_key = tuple(sorted([i, j, k]))
                if face_key not in seen_faces:
                    seen_faces.add(face_key)
                    center = (v[i] + v[j] + v[k]) / 3.0
                    center = center / np.linalg.norm(center)
                    face_centers.append(center)
    
    edge_centers = []
    for i, j in edges:
        center = (v[i] + v[j]) / 2.0
        center = center / np.linalg.norm(center)
        edge_centers.append(center)
    
    return v, np.array(face_centers), np.array(edge_centers)


_V, _F, _E = _get_icosahedral_points()


def _generate_icosahedral(n, rng, variant=0, scale=1.0):
    """
    Generate n points with icosahedral symmetry.
    variant 0: Center + 12 vertices + structured outer (face/edge centers)
    variant 1: Center + 12 vertices + random outer shell
    variant 2: Random compact with minimum spacing
    variant 3: Mackay-like icosahedral shell
    """
    r1 = 1.05 * scale
    
    points = [np.array([0.0, 0.0, 0.0])]
    shell1 = _V * r1
    points.extend(shell1.tolist())
    
    if n <= len(points):
        return np.array(points[:n])
    
    needed = n - len(points)
    
    if variant == 0:
        r2 = r1 * 1.55
        pool = np.vstack([_F, _E])
        idx = rng.permutation(len(pool))
        selected = pool[idx[:needed]] * r2
        for p in selected:
            points.append(p.tolist())
            
    elif variant == 1:
        r2 = r1 * 1.5
        for _ in range(needed):
            direction = rng.normal(size=3)
            direction = direction / np.linalg.norm(direction)
            p = direction * r2
            points.append(p.tolist())
            
    elif variant == 3:
        r2 = r1 * 1.6
        pool = np.vstack([_F, _E])
        idx = rng.permutation(len(pool))
        selected = pool[idx[:needed]] * r2
        for p in selected:
            points.append(p.tolist())
    else:
        radius = 1.2 * (n / (4.0 / 3.0 * np.pi)) ** (1.0 / 3.0)
        pts = []
        attempts = 0
        while len(pts) < n and attempts < 5000:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                ok = True
                for q in pts:
                    if np.linalg.norm(p - np.array(q)) < 0.55:
                        ok = False
                        break
                if ok:
                    pts.append(p.tolist())
            attempts += 1
        while len(pts) < n:
            p = rng.uniform(-radius, radius, 3)
            pts.append(p.tolist())
        return np.array(pts)
    
    return np.array(points)


def _relax_with_fallback(positions, potential, parameter):
    """Relax with error handling."""
    try:
        energy, relaxed = relax(positions, potential, parameter)
        return energy, relaxed
    except Exception:
        return np.inf, positions


def _l_bfgs_refine(positions, potential, parameter, maxiter=200):
    """
    L-BFGS-B refinement with tight tolerances for deeper local minima.
    """
    if not _check_time():
        return np.inf, positions.copy()
    
    n_atoms = len(positions)
    
    def objective(x):
        try:
            e, _ = dzugutov_energy_gradient(x.reshape(n_atoms, 3))
            return e
        except Exception:
            return 1e10
    
    def gradient(x):
        try:
            _, g = dzugutov_energy_gradient(x.reshape(n_atoms, 3))
            return g.ravel()
        except Exception:
            return np.zeros(x.shape)
    
    try:
        result = minimize(
            objective,
            positions.ravel(),
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        if result.success or np.isfinite(result.fun):
            return float(result.fun), result.x.reshape(n_atoms, 3)
    except Exception:
        pass
    
    return np.inf, positions.copy()


def _kick_and_relax(positions, potential, parameter, n_kicks=8, kick_size=0.3, rng=None):
    """
    Kick-and-relax: perturb positions, then L-BFGS refine to escape shallow minima.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n_atoms = len(positions)
    best_energy, best_pos = np.inf, positions.copy()
    
    # First try the original
    try:
        e, p = _l_bfgs_refine(positions, potential, parameter)
        if np.isfinite(e) and e < best_energy:
            best_energy, best_pos = e, p
    except Exception:
        pass
    
    for k in range(n_kicks):
        if not _check_time():
            break
        
        # Random kick
        kick = rng.normal(0, kick_size, size=(n_atoms, 3))
        new_pos = positions + kick
        
        # Check minimum distance
        dists = np.linalg.norm(new_pos[:, np.newaxis, :] - new_pos[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 0.5:
            continue
        
        # Relax the kicked structure
        try:
            e, p = _l_bfgs_refine(new_pos, potential, parameter)
            if np.isfinite(e) and e < best_energy:
                best_energy, best_pos = e, p
        except Exception:
            continue
    
    return best_energy, best_pos


def construct(instance):
    """
    Find low-energy Dzugutov cluster using icosahedral starting points
    with L-BFGS-B refinement and kick-and-relax for global search.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse icosahedral seeds with varied scale and twist
    seed_configs = [
        (42, 1.0, 0),
        (123, 0.95, 0),
        (456, 1.05, 1),
        (789, 1.0, 3),
        (101, 0.98, 1),
        (202, 1.02, 0),
        (303, 0.96, 3),
        (404, 1.04, 1),
    ]
    
    # Phase 1: Generate, relax, and L-BFGS refine candidates
    candidates = []
    for seed, scale, variant in seed_configs:
        if not _check_time():
            break
        rng = np.random.default_rng(seed)
        
        try:
            pos = _generate_icosahedral(size, rng, variant=variant, scale=scale)
            if len(pos) == size:
                # Quick relaxation first
                energy, positions = _relax_with_fallback(pos, potential, parameter)
                if np.isfinite(energy):
                    # L-BFGS refine for deeper minimum
                    e2, p2 = _l_bfgs_refine(positions, potential, parameter)
                    if np.isfinite(e2) and e2 < energy:
                        energy, positions = e2, p2
                    candidates.append((energy, positions.copy()))
        except Exception:
            pass
    
    if not candidates:
        # Fallback
        rng = np.random.default_rng(size)
        best_positions = rng.uniform(-1, 1, (size, 3))
        try:
            _, best_positions = relax(best_positions, potential, parameter)
        except Exception:
            pass
        return best_positions
    
    # Sort by energy and take top candidates for kick-and-relax
    candidates.sort(key=lambda x: x[0])
    top_candidates = candidates[:min(len(candidates), 6)]
    
    # Phase 2: Apply kick-and-relax to top candidates
    for idx, (init_energy, init_pos) in enumerate(top_candidates):
        if not _check_time():
            break
        
        # Allocate more kicks to better candidates
        if idx == 0:
            n_kicks = 12
            kick_size = 0.25
        elif idx < 3:
            n_kicks = 8
            kick_size = 0.3
        else:
            n_kicks = 5
            kick_size = 0.35
        
        rng_kr = np.random.default_rng(1000 + idx)
        
        try:
            kr_energy, kr_pos = _kick_and_relax(
                init_pos, potential, parameter,
                n_kicks=n_kicks, kick_size=kick_size, rng=rng_kr
            )
        except Exception:
            continue
        
        if kr_energy < best_energy:
            best_energy = kr_energy
            best_positions = kr_pos.copy()
    
    # Phase 3: Final L-BFGS polishing on the best result
    if best_positions is not None and _check_time():
        try:
            final_energy, final_pos = _l_bfgs_refine(
                best_positions, potential, parameter, maxiter=300
            )
            if final_energy < best_energy:
                best_energy = final_energy
                best_positions = final_pos
        except Exception:
            pass
    
    # Final fallback
    if best_positions is None:
        rng = np.random.default_rng(size)
        best_positions = rng.uniform(-1, 1, (size, 3))
        try:
            _, best_positions = relax(best_positions, potential, parameter)
        except Exception:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
