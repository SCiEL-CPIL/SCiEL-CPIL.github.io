"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
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


def _simulated_annealing(positions, potential, parameter, n_iter=2000, t_start=2.0, t_end=0.01, rng=None):
    """
    Simulated annealing to escape local minima.
    Uses gradient-based moves with Metropolis acceptance.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    pos = positions.copy()
    try:
        energy, grad = dzugutov_energy_gradient(pos)
    except Exception:
        return np.inf, positions
    
    if not np.isfinite(energy):
        return np.inf, positions
    
    best_energy = energy
    best_pos = pos.copy()
    n_atoms = len(pos)
    
    for i in range(n_iter):
        if not _check_time():
            break
        
        # Linear cooling schedule
        t = t_start * (t_end / t_start) ** (i / max(n_iter - 1, 1))
        
        # Generate perturbation: random displacement + gradient step
        step_size = 0.05 * (1.0 + 0.5 * np.exp(-i / 500.0))
        noise = rng.normal(0, step_size, size=pos.shape)
        
        # Combine gradient direction with random noise
        if np.linalg.norm(grad) > 1e-10:
            grad_dir = -grad / np.linalg.norm(grad)
        else:
            grad_dir = np.zeros(3)
        
        new_pos = pos + 0.3 * step_size * grad_dir + noise
        
        # Check minimum distance constraint
        if n_atoms <= 60:
            dists = np.linalg.norm(new_pos[:, np.newaxis, :] - new_pos[np.newaxis, :, :], axis=2)
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) < 0.5:
                continue
        
        try:
            new_energy, new_grad = dzugutov_energy_gradient(new_pos)
        except Exception:
            continue
        
        if not np.isfinite(new_energy):
            continue
        
        # Metropolis acceptance
        delta_e = new_energy - energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / t):
            pos = new_pos
            energy = new_energy
            grad = new_grad
            
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
        
        # Check convergence
        if np.linalg.norm(grad) < 1e-7 and i > 100:
            break
    
    return best_energy, best_pos


def _local_minimization(positions, potential, parameter, max_steps=300, step_size=0.05):
    """
    Gradient descent with adaptive step size for final polishing.
    """
    pos = positions.copy()
    best_energy = np.inf
    best_pos = pos.copy()
    
    for step in range(max_steps):
        if not _check_time():
            break
            
        try:
            energy, grad = dzugutov_energy_gradient(pos)
        except Exception:
            break
            
        if not np.isfinite(energy):
            break
            
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-7:
            break
        
        # Adaptive step: shrink if energy increases, grow if it decreases
        new_pos = pos - step_size * grad
        
        # Check minimum distance
        dists = np.linalg.norm(new_pos[:, np.newaxis, :] - new_pos[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 0.5:
            step_size *= 0.5
            continue
        
        try:
            new_energy, _ = dzugutov_energy_gradient(new_pos)
        except Exception:
            step_size *= 0.5
            continue
        
        if new_energy < energy:
            pos = new_pos
            step_size = min(step_size * 1.1, 0.1)
        else:
            step_size *= 0.7
    
    return best_energy, best_pos


def construct(instance):
    """
    Find low-energy Dzugutov cluster using icosahedral starting points
    with simulated annealing for global search and gradient descent for polishing.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse icosahedral seeds
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    scales = [0.95, 1.0, 1.05]
    variants = [0, 3, 1]
    
    # Phase 1: Generate and relax all candidate starting points
    candidates = []
    for seed in seeds:
        if not _check_time():
            break
        rng = np.random.default_rng(seed)
        
        for scale in scales:
            if not _check_time():
                break
            for variant in variants:
                if not _check_time():
                    break
                try:
                    pos = _generate_icosahedral(size, rng, variant=variant, scale=scale)
                    if len(pos) == size:
                        # Quick relaxation to get to a local minimum
                        energy, positions = _relax_with_fallback(pos, potential, parameter)
                        if np.isfinite(energy):
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
    
    # Sort by energy and take top candidates for SA
    candidates.sort(key=lambda x: x[0])
    top_candidates = candidates[:min(len(candidates), 8)]
    
    # Phase 2: Apply simulated annealing to top candidates
    for idx, (init_energy, init_pos) in enumerate(top_candidates):
        if not _check_time():
            break
        
        # Allocate more SA iterations to better candidates
        if idx == 0:
            n_iter = 3000
        elif idx < 3:
            n_iter = 2000
        else:
            n_iter = 1000
        
        rng_sa = np.random.default_rng(1000 + idx)
        
        try:
            sa_energy, sa_pos = _simulated_annealing(
                init_pos, potential, parameter,
                n_iter=n_iter, t_start=3.0, t_end=0.01, rng=rng_sa
            )
        except Exception:
            continue
        
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_positions = sa_pos.copy()
    
    # Phase 3: Final gradient descent polishing on the best result
    if best_positions is not None and _check_time():
        try:
            final_energy, final_pos = _local_minimization(
                best_positions, potential, parameter, max_steps=200, step_size=0.03
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
