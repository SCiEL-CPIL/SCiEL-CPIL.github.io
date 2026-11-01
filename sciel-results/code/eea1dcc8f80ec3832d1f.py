"""Lennard-Jones 55-atom global minimum: Mackay icosahedron seed + multi-stage L-BFGS polish."""
import numpy as np
import time

N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _lj_energy(pos):
    """Compute total LJ energy for a set of positions (vectorized)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3
    e = 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return np.sum(e) / 2.0

def _lj_energy_grad(pos):
    """Compute total LJ energy and gradient (vectorized, numerically stable)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 * r2 * r2
    r12 = r6 * r6

    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0

    with np.errstate(divide='ignore', invalid='ignore'):
        factor = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    np.fill_diagonal(factor, 0.0)

    sum_factor = np.sum(factor, axis=1, keepdims=True)
    sum_factor_pos = np.sum(factor[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    grad = pos * sum_factor - sum_factor_pos

    return energy, grad

def _make_mackay_icosahedron(r1=1.05, r2=1.75, r3=1.95):
    """Construct a two-shell Mackay icosahedron for 55 atoms.
    1 center + 12 first shell + 30 edge-midpoints + 12 vertex-like = 55
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)

    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms

    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(verts_unit[i] - verts_unit[j])

    min_dist = np.min(dists[dists > 1e-6])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))

    edge_mids = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            edge_mids.append(mid / norm)
    edge_mids = np.array(edge_mids)

    center = np.array([[0.0, 0.0, 0.0]])
    first_shell = verts_unit * r1
    second_inner = edge_mids * r2
    second_outer = verts_unit * r3

    all_pos = np.vstack([center, first_shell, second_inner, second_outer])
    return all_pos

def _relax_lbfgs(pos, deadline=None, maxiter=500, gtol=1e-10):
    """Relax using L-BFGS-B with deadline guard."""
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()

    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()

    x0 = pos.flatten()

    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-14}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass

    return best_pos, best_energy

def _polish(pos, deadline=None):
    """Multi-stage high-precision polish: progressively tighten L-BFGS tolerance."""
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    x0 = pos.flatten()

    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()

    stages = [
        {'maxiter': 2000, 'gtol': 1e-12, 'maxls': 200, 'ftol': 1e-15, 'maxcor': 30},
        {'maxiter': 5000, 'gtol': 1e-14, 'maxls': 500, 'ftol': 1e-16, 'maxcor': 50},
        {'maxiter': 10000, 'gtol': 1e-16, 'maxls': 1000, 'ftol': 1e-16, 'maxcor': 100},
        {'maxiter': 20000, 'gtol': 1e-18, 'maxls': 2000, 'ftol': 1e-18, 'maxcor': 200},
    ]

    for stage_opts in stages:
        if deadline is not None and time.time() > deadline - 5.0:
            break
        try:
            result = minimize(
                obj_and_grad, x0, jac=True, method='L-BFGS-B',
                options=stage_opts
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3)
                x0 = result.x
            else:
                break  # No improvement, stop polishing
        except Exception:
            break

    return best_pos, best_energy

def _optimize_shell_radii():
    """Optimize the three shell radii (r1, r2, r3) to minimize LJ energy.
    Uses Nelder-Mead on the 3D radius space, then L-BFGS to relax the full geometry.
    """
    def objective(radii):
        r1, r2, r3 = radii
        # Enforce basic sanity: r1 > 0, r2 > r1, r3 >= r2
        if r1 < 0.5 or r2 < r1 + 0.1 or r3 < r2 - 0.2 or r3 > 3.0:
            return 1e10
        pos = _make_mackay_icosahedron(r1, r2, r3)
        # Quick L-BFGS relaxation to get a good energy estimate
        try:
            relaxed, energy = _relax_lbfgs(pos, maxiter=200, gtol=1e-8)
            return energy
        except Exception:
            return 1e10

    # Start from the known good Mackay geometry
    x0 = np.array([1.05, 1.75, 1.95])
    try:
        result = minimize(
            objective, x0, method='Nelder-Mead',
            options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        if result.fun < 1e9:
            return result.x
    except Exception:
        pass
    return x0

def construct():
    """Construct 55-atom LJ cluster near global minimum."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    # Phase 0: Optimize shell radii via Nelder-Mead
    if time.time() < deadline - 80.0:
        optimal_radii = _optimize_shell_radii()
        r1_opt, r2_opt, r3_opt = optimal_radii
    else:
        r1_opt, r2_opt, r3_opt = 1.05, 1.75, 1.95

    # Build candidate set: optimized radii + curated combinations + perturbations
    radius_combos = [
        (r1_opt, r2_opt, r3_opt),
        (1.05, 1.75, 1.95),
        (1.00, 1.70, 1.90),
        (1.10, 1.80, 2.00),
        (1.05, 1.80, 2.00),
        (1.00, 1.75, 1.95),
        (1.08, 1.78, 1.98),
        (1.02, 1.72, 1.92),
        (1.12, 1.82, 2.02),
        (1.05, 1.72, 1.92),
        (1.00, 1.80, 2.00),
    ]

    candidates = []
    for r1, r2, r3 in radius_combos:
        cand = _make_mackay_icosahedron(r1, r2, r3)
        candidates.append(cand)

    # Add small perturbations of the optimized Mackay at multiple noise levels
    np.random.seed(42)
    base = _make_mackay_icosahedron(r1_opt, r2_opt, r3_opt)
    for noise in [0.01, 0.03, 0.05, 0.08]:
        perturbed = base + np.random.randn(55, 3) * noise
        candidates.append(perturbed)

    # Phase 1: Quick relaxation of each candidate
    results = []
    for cand in candidates:
        if time.time() > deadline - 30.0:
            break
        relaxed, energy = _relax_lbfgs(cand, deadline=deadline)
        results.append((energy, relaxed))

    # Sort by energy, keep top 5
    results.sort(key=lambda x: x[0])
    top_results = results[:5]

    # Phase 2: Deep polish on the best candidates
    best_energy = float('inf')
    best_pos = top_results[0][1].copy()

    for energy, pos in top_results:
        if time.time() > deadline - 5.0:
            break
        polished, pol_energy = _polish(pos, deadline=deadline)
        if pol_energy < best_energy:
            best_energy = pol_energy
            best_pos = polished

    # Phase 3: Final ultra-tight polish on the single best configuration
    if time.time() < deadline - 5.0:
        final_pos, final_energy = _polish(best_pos, deadline=deadline)
        if final_energy < best_energy:
            best_energy = final_energy
            best_pos = final_pos

    # Ensure minimum inter-atomic distance >= 0.5
    min_dist_found = float('inf')
    for i in range(55):
        for j in range(i + 1, 55):
            d = np.linalg.norm(best_pos[i] - best_pos[j])
            if d < min_dist_found:
                min_dist_found = d

    if min_dist_found < 0.5:
        scale = 0.5 / min_dist_found * 1.01
        best_pos = best_pos * scale

    return best_pos
# EVOLVE-BLOCK-END
