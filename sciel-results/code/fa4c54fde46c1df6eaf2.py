"""Basin-hopping + multi-start relaxation for SC cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _time_left(start_time, deadline=290.0):
    return deadline - (time.time() - start_time)


def _fcc_shells(n, spacing=1.0):
    """Generate n points from FCC lattice shells, centered."""
    if n == 0:
        return np.zeros((0, 3))
    
    points = []
    max_coord = int(np.ceil(n ** (1/3) * 2)) + 2
    
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                for bx, by, bz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    x = (ix + bx) * spacing
                    y = (iy + by) * spacing
                    z = (iz + bz) * spacing
                    if x*x + y*y + z*z <= (n ** (1/3) * spacing * 1.2) ** 2:
                        points.append([x, y, z])
    
    points = np.array(points)
    if len(points) > n:
        dists = np.sum(points**2, axis=1)
        idx = np.argsort(dists)[:n]
        points = points[idx]
    elif len(points) < n:
        rng = np.random.default_rng(42)
        while len(points) < n:
            p = rng.uniform(-n**(1/3)*spacing, n**(1/3)*spacing, 3)
            points = np.vstack([points, [p]])
    
    points -= np.mean(points, axis=0)
    return points


def _hcp_shells(n, spacing=1.0):
    """Generate n points from HCP lattice shells."""
    if n == 0:
        return np.zeros((0, 3))
    
    points = []
    a = spacing
    c = np.sqrt(8/3) * a
    max_coord = int(np.ceil(n ** (1/3) * 2)) + 2
    
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                if iz % 2 == 0:
                    x = ix * a + 0.5 * iy * a * 0.5
                    y = iy * a * np.sqrt(3)/2
                else:
                    x = ix * a + 0.5 * a + 0.5 * iy * a * 0.5
                    y = iy * a * np.sqrt(3)/2
                z = iz * c / 2
                if x*x + y*y + z*z <= (n ** (1/3) * spacing * 1.2) ** 2:
                    points.append([x, y, z])
    
    points = np.array(points)
    if len(points) > n:
        dists = np.sum(points**2, axis=1)
        idx = np.argsort(dists)[:n]
        points = points[idx]
    elif len(points) < n:
        rng = np.random.default_rng(42)
        while len(points) < n:
            p = rng.uniform(-n**(1/3)*spacing, n**(1/3)*spacing, 3)
            points = np.vstack([points, [p]])
    
    points -= np.mean(points, axis=0)
    return points


def _icosahedron_based(n, spacing=1.0):
    """Generate points based on icosahedral symmetry with nested shells."""
    phi = (1 + 5 ** 0.5) / 2
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    vertices = vertices / np.linalg.norm(vertices[0]) * spacing
    
    points = [[0.0, 0.0, 0.0]]
    points.extend([v.tolist() for v in vertices])
    
    if n > 13:
        dodec = np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
            [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
            [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
            [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0]
        ])
        dodec = dodec / np.linalg.norm(dodec[0]) * spacing * 1.8
        for v in dodec:
            if len(points) >= n:
                break
            points.append(v.tolist())
    
    rng = np.random.default_rng(42)
    while len(points) < n:
        base = points[rng.integers(0, len(points))]
        p = np.array(base) + rng.normal(0, 0.3 * spacing, 3)
        points.append(p.tolist())
    
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points


def _random_compact(n, spacing=1.0, seed=None):
    """Generate a random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    min_dist = 0.5 * spacing
    
    points = []
    attempts = 0
    max_attempts = n * 200
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - np.array(q)) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    return points


def _perturb(positions, sigma=0.1, seed=None):
    """Add small random perturbations to positions."""
    rng = np.random.default_rng(seed)
    return positions + rng.normal(0, sigma, positions.shape)


def _check_validity(positions, min_dist=0.35):
    """Check if positions are valid (finite and no atoms too close)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(positions[i] - positions[j]) < min_dist:
                return False
    return True


def _sc_energy_and_grad(x_flat, kind):
    """Energy and gradient for scipy minimize from flat array."""
    pos = x_flat.reshape(-1, 3)
    e, g = sutton_chen_energy_gradient(pos, kind)
    return e, g.flatten()


def _refine(positions, kind, start_time, deadline=290.0, maxiter=200):
    """Refine a structure using BFGS with the SC gradient. Returns (energy, positions)."""
    if _time_left(start_time, deadline) < 5.0:
        e, _ = sutton_chen_energy_gradient(positions, kind)
        return e, positions
    try:
        res = minimize(_sc_energy_and_grad, positions.flatten(), jac=True,
                       method='BFGS', options={'maxiter': maxiter, 'gtol': 1e-6})
        if res.success and res.fun < np.inf:
            return res.fun, res.x.reshape(-1, 3)
        # If BFGS failed, try L-BFGS-B as fallback
        e, pos = relax(positions, 'sutton_chen', kind)
        return e, pos
    except Exception:
        e, pos = relax(positions, 'sutton_chen', kind)
        return e, pos


def _basin_hopping(positions, kind, start_time, deadline=290.0, n_steps=50,
                   step_size=0.3, temperature=0.5, rng=None):
    """
    Basin-hopping: Monte Carlo random kicks + local minimization.
    This is the standard global optimization method for atomic clusters.
    Returns best (energy, positions) found.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    # First minimize the starting structure
    best_energy, best_pos = _refine(positions, kind, start_time, deadline)
    
    if not np.isfinite(best_energy):
        return best_energy, best_pos
    
    current_energy = best_energy
    current_pos = best_pos.copy()
    
    for step in range(n_steps):
        if _time_left(start_time, deadline) < 15.0:
            break
        
        # Random perturbation (Monte Carlo move)
        perturbation = rng.normal(0, step_size, current_pos.shape)
        proposed = current_pos + perturbation
        
        # Local minimization of the perturbed structure
        try:
            e_proposed, pos_proposed = _refine(proposed, kind, start_time, deadline, maxiter=100)
        except Exception:
            continue
        
        if not np.isfinite(e_proposed):
            continue
        
        # Metropolis acceptance criterion
        delta_e = e_proposed - current_energy
        if delta_e < 0:
            # Always accept downhill moves
            current_energy = e_proposed
            current_pos = pos_proposed.copy()
        else:
            # Accept uphill moves with probability exp(-delta_e / temperature)
            if rng.random() < np.exp(-delta_e / max(temperature, 1e-10)):
                current_energy = e_proposed
                current_pos = pos_proposed.copy()
        
        # Track best
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
        
        # Adaptively adjust step size: increase if stuck, decrease if improving too much
        if delta_e > 0 and rng.random() < 0.5:
            step_size *= 1.1
        elif delta_e < -0.5:
            step_size *= 0.9
        step_size = np.clip(step_size, 0.1, 1.5)
    
    return best_energy, best_pos


def construct(instance):
    """
    Basin-hopping + multi-start relaxation for SC cluster global minima.
    Uses diverse structural templates as starting points, then applies
    basin-hopping (MC kicks + BFGS refinement) to escape local minima.
    """
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    start_time = time.time()
    deadline = 290.0
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starting structures
    starters = []
    
    # FCC-like (most important for SC clusters)
    starters.append(_fcc_shells(size, spacing=1.0))
    
    # HCP-like
    starters.append(_hcp_shells(size, spacing=1.0))
    
    # Icosahedron-based
    starters.append(_icosahedron_based(size, spacing=1.0))
    
    # Multiple random compact structures with different seeds
    for seed in [size, size + 1, size + 2, size + 3, size + 4, size + 5]:
        starters.append(_random_compact(size, spacing=1.0, seed=seed))
    
    # Phase 1: Quick relaxation of all starters to get good initial minima
    for i, starter in enumerate(starters):
        if _time_left(start_time, deadline) < 20.0:
            break
        
        try:
            energy, positions = relax(starter, potential, kind)
            
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue
    
    # Phase 2: Basin-hopping from the best structure
    if best_positions is not None:
        rng = np.random.default_rng(size + 100)
        
        # Run basin-hopping from the best found structure
        bh_energy, bh_pos = _basin_hopping(
            best_positions, kind, start_time, deadline,
            n_steps=80, step_size=0.25, temperature=0.3, rng=rng
        )
        
        if np.isfinite(bh_energy) and bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_pos.copy()
        
        # Basin-hopping from second-best starter (if different)
        if _time_left(start_time, deadline) > 60.0:
            rng2 = np.random.default_rng(size + 200)
            # Use a different starter for diversity
            alt_starter = starters[1] if len(starters) > 1 else starters[0]
            alt_energy, alt_pos = _basin_hopping(
                alt_starter, kind, start_time, deadline,
                n_steps=40, step_size=0.3, temperature=0.4, rng=rng2
            )
            
            if np.isfinite(alt_energy) and alt_energy < best_energy:
                best_energy = alt_energy
                best_positions = alt_pos.copy()
        
        # Phase 3: Final fine-tuning with tight BFGS
        if _time_left(start_time, deadline) > 20.0 and best_positions is not None:
            final_energy, final_pos = _refine(
                best_positions, kind, start_time, deadline, maxiter=300
            )
            if np.isfinite(final_energy) and final_energy < best_energy:
                best_energy = final_energy
                best_positions = final_pos.copy()
    
    # Fallback: return a simple valid configuration if nothing worked
    if best_positions is None:
        best_positions = _random_compact(size, spacing=1.0, seed=size)
    
    # Final validity check
    if not _check_validity(best_positions):
        # Try to fix by spreading out atoms
        rng = np.random.default_rng(size + 500)
        for _ in range(10):
            fixed = best_positions + rng.normal(0, 0.1, best_positions.shape)
            if _check_validity(fixed):
                best_positions = fixed
                break
    
    return best_positions
# EVOLVE-BLOCK-END
