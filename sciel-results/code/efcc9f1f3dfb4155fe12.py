"""
Strategy: Parallel Multi-Chain Basin Hopping with L-BFGS-B and PT Seeding.

Structural differences from the parent:
- 4 concurrent basin-hopping chains (ThreadPoolExecutor) instead of 1 sequential chain.
- L-BFGS-B local minimization with vectorized Dzugutov energy/gradient instead of
  the black-box relax() call, yielding far more iterations per second.
- Polytetrahedral (icosahedral-shell) seeding for physically motivated starts.
- Adaptive perturbation step size per chain.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Global deadline
# ---------------------------------------------------------------------------
_START_TIME = time.time()
_HARD_LIMIT = 315.0  # 15 s safety margin before 330 s kill

def _remaining():
    return max(0.1, _HARD_LIMIT - (time.time() - _START_TIME))

# ---------------------------------------------------------------------------
# Vectorized Dzugutov energy & gradient
# V(r) = A*(r^{-m} - B)*exp(c/(r-a))   for r < a
#      = B*exp(d/(r-b))                 for r < b
#      = 0                              for r >= b
# ---------------------------------------------------------------------------
_A, _B, _a, _b, _c, _d, _m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16

def _dzugutov_energy_gradient(pos):
    """Return (energy, gradient) for a (n,3) array.  Fully vectorized."""
    n = pos.shape[0]
    # Pairwise differences  (n, n, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=2) + 1e-12)  # (n, n)
    np.fill_diagonal(r, np.inf)

    # Masks
    mask_inner = r < _a
    mask_mid = (r >= _a) & (r < _b)

    # Energy components
    e_inner = np.zeros_like(r)
    e_mid = np.zeros_like(r)

    ri = r[mask_inner]
    if ri.size:
        e_inner[mask_inner] = _A * (ri ** (-_m) - _B) * np.exp(_c / (ri - _a))

    rm = r[mask_mid]
    if rm.size:
        e_mid[mask_mid] = _B * np.exp(_d / (rm - _b))

    energy = 0.5 * np.sum(e_inner + e_mid)

    # Gradient: dE/dpos_i = sum_j dV(r_ij)/dr_ij * (pos_i - pos_j)/r_ij
    # dV/dr for inner:
    #   V = A*(r^{-m}-B)*exp(c/(r-a))
    #   dV/dr = A*[ -m*r^{-m-1}*exp(c/(r-a)) + (r^{-m}-B)*exp(c/(r-a))*(-c/(r-a)^2) ]
    # dV/dr for mid:
    #   V = B*exp(d/(r-b))
    #   dV/dr = B*exp(d/(r-b))*(-d/(r-b)^2)

    dV = np.zeros_like(r)

    if ri.size:
        exp_i = np.exp(_c / (ri - _a))
        dV_i = _A * (-_m * ri ** (-_m - 1) * exp_i
                     + (ri ** (-_m) - _B) * exp_i * (-_c / (ri - _a) ** 2))
        dV[mask_inner] = dV_i

    if rm.size:
        exp_m = np.exp(_d / (rm - _b))
        dV_m = _B * exp_m * (-_d / (rm - _b) ** 2)
        dV[mask_mid] = dV_m

    # Gradient: grad_i = sum_{j!=i} dV(r_ij) * (pos_i - pos_j) / r_ij
    # dV * diff / r  has shape (n,n,3); sum over j (axis=1)
    # diff[i,j] = pos_i - pos_j, so grad_i = sum_j dV[i,j] * diff[i,j] / r[i,j]
    factor = dV / r  # (n, n)
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)  # (n, 3)

    return energy, grad

def _local_minimize(pos, maxiter=200):
    """L-BFGS-B local minimisation.  Returns (energy, positions)."""
    def obj(x):
        e, _ = _dzugutov_energy_gradient(x)
        return e

    def grad(x):
        _, g = _dzugutov_energy_gradient(x)
        return g

    res = minimize(obj, pos, jac=grad, method='L-BFGS-B',
                   options={'maxiter': maxiter, 'maxls': 50, 'ftol': 1e-10})
    e, _ = _dzugutov_energy_gradient(res.x)
    return e, res.x

# ---------------------------------------------------------------------------
# Structure generators
# ---------------------------------------------------------------------------
def _generate_icosahedron(n_atoms, rng):
    """Icosahedral-shell structure (Mackay-like) for n_atoms."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    positions = [np.zeros(3)]  # centre
    r1 = 1.9
    for v in verts:
        positions.append(v * r1)

    if n_atoms <= 13:
        return np.array(positions[:n_atoms])

    # Outer shell: distribute remaining atoms on a sphere of radius r2
    r2 = r1 * 1.75
    num_outer = n_atoms - 13
    outer = []
    for _ in range(num_outer):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        outer.append(v * r2)
    positions.extend(outer)
    return np.array(positions[:n_atoms])

def _random_compact(n_atoms, rng, spacing=1.0):
    """Random compact structure in a sphere."""
    radius = spacing * n_atoms ** (1.0 / 3.0) * 0.85
    points = []
    for _ in range(n_atoms * 200):
        if len(points) >= n_atoms:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if not points:
            points.append(p)
        else:
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if dists.min() >= 0.6 * spacing:
                points.append(p)
    if len(points) < n_atoms:
        pts = np.array(points) if points else np.zeros((0, 3))
        extra = rng.uniform(-radius, radius, (n_atoms - len(points), 3))
        pts = np.vstack([pts, extra]) if len(points) else extra
        return pts
    return np.array(points)

# ---------------------------------------------------------------------------
# Basin-hopping chain (runs on one thread)
# ---------------------------------------------------------------------------
def _basin_hopping_chain(n_atoms, seed, time_budget, rng):
    """One basin-hopping chain.  Returns (best_energy, best_positions)."""
    # Generate a diverse start
    start_type = rng.integers(0, 3)
    if start_type == 0:
        start = _generate_icosahedron(n_atoms, rng)
    elif start_type == 1:
        start = _random_compact(n_atoms, rng)
    else:
        start = _generate_icosahedron(n_atoms, rng) + rng.normal(scale=0.15, size=(n_atoms, 3))

    # Initial local minimisation
    try:
        cur_e, cur_pos = _local_minimize(start, maxiter=300)
    except Exception:
        cur_e, cur_pos = _dzugutov_energy_gradient(start)

    best_e, best_pos = cur_e, cur_pos.copy()

    step_size = 0.25
    min_step = 0.05
    max_step = 0.6
    stuck_count = 0
    max_stuck = 5

    while _remaining() > 2.0 and (time.time() - time_budget) < 300:
        # Perturb
        perturbed = cur_pos + rng.normal(scale=step_size, size=cur_pos.shape)

        # Quick validity check (min distance 0.5)
        # Use a fast check: only for larger step sizes
        if step_size > 0.1:
            diff = perturbed[:, np.newaxis, :] - perturbed[np.newaxis, :, :]
            r2 = np.sum(diff * diff, axis=2)
            np.fill_diagonal(r2, np.inf)
            if r2.min() < 0.25:  # 0.5^2
                step_size = max(min_step, step_size * 0.7)
                continue

        # Local minimisation
        try:
            new_e, new_pos = _local_minimize(perturbed, maxiter=200)
        except Exception:
            new_e, new_pos = _dzugutov_energy_gradient(perturbed)

        if new_e < cur_e:
            cur_e, cur_pos = new_e, new_pos
            if cur_e < best_e:
                best_e, best_pos = cur_e, cur_pos.copy()
            stuck_count = 0
            # Slightly increase step on acceptance to explore
            step_size = min(max_step, step_size * 1.05)
        else:
            stuck_count += 1
            if stuck_count >= max_stuck:
                # Jump: bigger perturbation to escape basin
                step_size = min(max_step, step_size * 1.5)
                stuck_count = 0
            else:
                step_size = max(min_step, step_size * 0.9)

    return best_e, best_pos

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))

    t0 = time.time()
    # Per-instance time budget: total 300 s / 6 instances ≈ 50 s, use 42 s
    time_budget = 42.0

    rng = np.random.default_rng(size)

    # Generate 4 diverse initial structures
    starts = []
    for i in range(4):
        if i == 0:
            s = _generate_icosahedron(size, rng)
        elif i == 1:
            s = _random_compact(size, rng)
        elif i == 2:
            s = _generate_icosahedron(size, rng) + rng.normal(scale=0.2, size=(size, 3))
        else:
            s = _random_compact(size, rng, spacing=0.8)
        # Ensure minimum distance 0.5
        # Simple pairwise check
        valid = True
        for a_i in range(size):
            for b_i in range(a_i + 1, size):
                if np.linalg.norm(s[a_i] - s[b_i]) < 0.5:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            starts.append(s)

    if not starts:
        # Fallback: random
        starts.append(_random_compact(size, rng))

    # Run 4 parallel basin-hopping chains
    results = []
    n_chains = min(4, len(starts))

    with ThreadPoolExecutor(max_workers=n_chains) as executor:
        futures = {}
        for i in range(n_chains):
            chain_rng = np.random.default_rng(size * 1000 + i * 7)
            f = executor.submit(_basin_hopping_chain, size, i, t0, chain_rng)
            futures[f] = i

        for f in as_completed(futures):
            try:
                e, pos = f.result()
                results.append((e, pos))
            except Exception:
                pass

    # Also do a quick single relax on the best start if we have time
    if _remaining() > 5.0 and results:
        # Pick best from results
        results.sort(key=lambda x: x[0])
        best_e, best_pos = results[0]
    elif results:
        results.sort(key=lambda x: x[0])
        best_e, best_pos = results[0]
    else:
        # Fallback: just return the first start
        best_pos = starts[0]
        best_e = _dzugutov_energy_gradient(best_pos)[0]

    # Final polish: one more L-BFGS-B on the best structure
    if _remaining() > 3.0:
        try:
            final_e, final_pos = _local_minimize(best_pos, maxiter=300)
            if final_e < best_e:
                best_e, best_pos = final_e, final_pos
        except Exception:
            pass

    return best_pos
# EVOLVE-BLOCK-END
