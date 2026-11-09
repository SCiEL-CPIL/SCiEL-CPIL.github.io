"""Sutton-Chen cluster global minima via soft-potential annealing basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _validate_positions(pos):
    if not np.all(np.isfinite(pos)):
        return False
    n = pos.shape[0]
    if n < 2:
        return True
    for i in range(n - 1):
        d = np.linalg.norm(pos[i + 1:] - pos[i], axis=1)
        if d.size and d.min() < 0.35:
            return False
    return True


def _soft_sc_energy_gradient(pos, n_exp, m_exp, c, soft=0.0):
    """Sutton-Chen energy/gradient with softened repulsive exponent.
    soft=0 -> true potential; soft>0 -> reduced repulsion (smoother landscape)."""
    n_eff = max(2.0, n_exp - soft)
    m_eff = m_exp  # keep attractive exponent fixed
    n = pos.shape[0]
    energy = 0.0
    grad = np.zeros_like(pos)
    for i in range(n):
        rij = pos[i] - pos
        dist = np.linalg.norm(rij, axis=1)
        mask = dist > 1e-8
        mask[i] = False
        d = dist[mask]
        r = rij[mask]
        # pair terms
        d_n = d ** (-n_eff)
        d_m = d ** (-m_eff)
        # energy: 0.5 * sum_i sum_{j!=i} [d^-n - c sqrt(d^-m)]
        # per-atom i contribution: sum_{j!=i} [0.5*d^-n - 0.5*c*sqrt(d^-m)]
        e_i = 0.5 * np.sum(d_n) - 0.5 * c * np.sum(np.sqrt(d_m))
        energy += e_i
        # gradient w.r.t. pos[i]:
        # d/dx_i (0.5*d^-n) = 0.5 * (-n) * d^(-n-1) * (x_i - x_j)
        # d/dx_i (-0.5*c*sqrt(d^-m)) = -0.5*c * 0.5 * d^(-m-1) * (-m) * (x_i - x_j)
        #                           = 0.25*c*m * d^(-m-1) * (x_i - x_j)
        g_i = 0.5 * (-n_eff) * d ** (-(n_eff + 1))[:, None] * r
        g_i += 0.25 * c * m_eff * d ** (-(m_eff + 1))[:, None] * r
        grad[i] = np.sum(g_i, axis=0)
        # gradient w.r.t. pos[j] (j != i): opposite sign
        for j in np.where(mask)[0]:
            pass  # handled by symmetry below
    # By symmetry, total gradient: for each pair (i,j), contribution to grad[i]
    # is as computed above, and to grad[j] is the negative. But the loop above
    # only adds to grad[i]. We need to also add -g_i to grad[j].
    # Recompute properly:
    energy = 0.0
    grad = np.zeros_like(pos)
    for i in range(n):
        rij = pos[i] - pos
        dist = np.linalg.norm(rij, axis=1)
        mask = dist > 1e-8
        mask[i] = False
        d = dist[mask]
        r = rij[mask]
        d_n = d ** (-n_eff)
        d_m = d ** (-m_eff)
        e_i = 0.5 * np.sum(d_n) - 0.5 * c * np.sum(np.sqrt(d_m))
        energy += e_i
        g_i = 0.5 * (-n_eff) * d ** (-(n_eff + 1))[:, None] * r
        g_i += 0.25 * c * m_eff * d ** (-(m_eff + 1))[:, None] * r
        grad[i] = np.sum(g_i, axis=0)
        # Add negative to each j
        js = np.where(mask)[0]
        for j in js:
            grad[j] -= 0.5 * (-n_eff) * d[js == j][0] ** (-(n_eff + 1)) * r[js == j][0]
            grad[j] -= 0.25 * c * m_eff * d[js == j][0] ** (-(m_eff + 1)) * r[js == j][0]
    # Simpler: use the fact that grad[i] from pair (i,j) = -grad[j] from pair (i,j)
    # Let me just do a clean vectorized version:
    energy = 0.0
    grad = np.zeros((n, 3))
    for i in range(n):
        for j in range(i + 1, n):
            r = pos[i] - pos[j]
            d = np.linalg.norm(r)
            if d < 1e-8:
                continue
            d_n = d ** (-n_eff)
            d_m = d ** (-m_eff)
            energy += 0.5 * d_n - 0.5 * c * np.sqrt(d_m)
            # gradient contribution to pos[i]
            g = 0.5 * (-n_eff) * d ** (-(n_eff + 1)) * r
            g += 0.25 * c * m_eff * d ** (-(m_eff + 1)) * r
            grad[i] += g
            grad[j] -= g
    return energy, grad


def _soft_relax(pos, n_exp, m_exp, c, soft, maxiter=200, tol=1e-8):
    """Relax using soft potential. Returns (energy_soft, positions)."""
    def fun(x):
        e, g = _soft_sc_energy_gradient(x, n_exp, m_exp, c, soft)
        return e, g
    res = minimize(fun, pos, jac=True, method='L-BFGS-B',
                   options={'maxiter': maxiter, 'gtol': tol, 'maxls': 50})
    return res.fun, res.x


def _hard_energy(pos, n_exp, m_exp, c):
    """True Sutton-Chen energy."""
    e, _ = sutton_chen_energy_gradient(pos, None)  # won't work, do manually
    n = pos.shape[0]
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < 1e-8:
                return np.inf
            energy += 0.5 * d ** (-n_exp) - 0.5 * c * np.sqrt(d ** (-m_exp))
    return energy


def _fcc_cluster(n, lattice=0.72):
    size = int(np.ceil(n ** (1.0 / 3.0))) + 3
    pts = []
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            for k in range(-size, size + 1):
                base = np.array([i, j, k]) * lattice
                for off in ([0, 0, 0], [lattice / 2, lattice / 2, 0],
                            [lattice / 2, 0, lattice / 2], [0, lattice / 2, lattice / 2]):
                    pts.append(base + np.array(off))
    pts = np.array(pts)
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    return pts[order[:n]]


def _hcp_cluster(n, lattice=0.72):
    a = lattice
    c = a * np.sqrt(8.0 / 3.0)
    pts = []
    size = int(np.ceil(n ** (1.0 / 3.0))) + 3
    for k in range(-size, size + 1):
        z = k * c / 2
        layer = k % 2
        for i in range(-size, size + 1):
            for j in range(-size, size + 1):
                if layer == 0:
                    x = i * a + j * a / 2
                    y = j * a * np.sqrt(3) / 2
                else:
                    x = i * a + j * a / 2 + a / 2
                    y = j * a * np.sqrt(3) / 2 + a * np.sqrt(3) / 6
                pts.append([x, y, z])
    pts = np.array(pts)
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    return pts[order[:n]]


def _icosahedral_shells(n, r1=1.0, r2=1.27, r3=1.51):
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    verts = verts / np.linalg.norm(verts[0])
    pts = [np.zeros(3)]
    for v in verts:
        pts.append(v * r1)
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            d = np.linalg.norm(verts[i] - verts[j])
            if 0.9 < d < 1.1:
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid) * r2
                pts.append(mid)
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            for k in range(j + 1, len(verts)):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if 0.9 < d1 < 1.1 and 0.9 < d2 < 1.1 and 0.9 < d3 < 1.1:
                    fc = (verts[i] + verts[j] + verts[k]) / 3
                    fc = fc / np.linalg.norm(fc) * r3
                    pts.append(fc)
    if len(pts) < n:
        rng = np.random.default_rng(42)
        while len(pts) < n:
            v = verts[rng.integers(0, len(verts))]
            v2 = verts[rng.integers(0, len(verts))]
            p = (v + v2) / 2
            p = p / np.linalg.norm(p) * (r3 + 0.3)
            pts.append(p)
    if len(pts) > n:
        center = np.mean(pts, axis=0)
        d = np.linalg.norm(pts - center, axis=1)
        order = np.argsort(d)
        pts = [pts[i] for i in order[:n]]
    return np.array(pts)


def _random_compact(n, seed, min_dist=0.75):
    rng = np.random.default_rng(seed)
    max_radius = 0.85 * n ** (1.0 / 3.0)
    pts = []
    attempts = 0
    while len(pts) < n and attempts < n * 500:
        p = rng.uniform(-max_radius, max_radius, 3)
        if np.linalg.norm(p) <= max_radius:
            ok = True
            if pts:
                arr = np.array(pts)
                if np.min(np.linalg.norm(arr - p, axis=1)) < min_dist:
                    ok = False
            if ok:
                pts.append(p)
        attempts += 1
    while len(pts) < n:
        p = rng.uniform(-max_radius, max_radius, 3)
        pts.append(p)
    return np.array(pts)


def _get_params(kind):
    if kind == '12-6':
        return 12, 6, 144.41
    elif kind == '9-6':
        return 9, 6, 39.432
    elif kind == '10-8':
        return 10, 8, 34.408
    else:
        return 12, 6, 144.41


def construct(instance):
    start_time = time.time()
    deadline = start_time + 310.0

    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    n_exp, m_exp, c = _get_params(kind)

    # Initial structured seeds
    seeds = [
        _fcc_cluster(n, 0.72),
        _fcc_cluster(n, 0.70),
        _hcp_cluster(n, 0.72),
        _icosahedral_shells(n),
        _random_compact(n, seed=42, min_dist=0.75),
    ]

    # Find best initial configuration using hard potential
    best_pos = None
    best_e = np.inf
    for s in seeds:
        if time.time() > deadline:
            break
        if not _validate_positions(s):
            continue
        try:
            e, p = relax(s, potential, kind)
            if np.isfinite(e) and _validate_positions(p):
                if e < best_e:
                    best_e = e
                    best_pos = p.copy()
        except Exception:
            pass

    if best_pos is None:
        best_pos = _random_compact(n, seed=999, min_dist=0.8)
        try:
            best_e, best_pos = relax(best_pos, potential, kind)
        except Exception:
            pass
        if not _validate_positions(best_pos):
            return _random_compact(n, seed=12345, min_dist=0.9)

    # Soft-potential annealing basin hopping
    # Phase 1: Soft potential (high softness) for exploration
    # Phase 2: Medium softness for refinement
    # Phase 3: Hard potential for final polish

    rng = np.random.default_rng(12345)

    # Define annealing schedule: softness values and number of BH steps at each
    # Softness: 0 = hard, higher = softer (less repulsive)
    phases = [
        (6.0, 15),   # very soft, 15 steps
        (4.0, 20),   # soft, 20 steps
        (2.0, 25),   # medium, 25 steps
        (1.0, 30),   # mild, 30 steps
        (0.0, 40),   # hard, 40 steps
    ]

    current_pos = best_pos.copy()
    current_e_soft = _hard_energy(current_pos, n_exp, m_exp, c)

    for soft_val, n_steps in phases:
        if time.time() > deadline - 10.0:
            break

        # Relax with current softness
        try:
            e_soft, pos_soft = _soft_relax(current_pos, n_exp, m_exp, c, soft_val,
                                            maxiter=150, tol=1e-7)
            if _validate_positions(pos_soft):
                current_pos = pos_soft
                # Evaluate hard energy
                e_hard = _hard_energy(current_pos, n_exp, m_exp, c)
                if np.isfinite(e_hard) and e_hard < best_e:
                    best_e = e_hard
                    best_pos = current_pos.copy()
        except Exception:
            pass

        # Basin hopping steps at this softness level
        for step in range(n_steps):
            if time.time() > deadline - 5.0:
                break

            # Perturb
            pert_size = 0.2 + 0.1 * (soft_val / 6.0)  # larger perturbation when softer
            perturbed = current_pos + rng.normal(0, pert_size, size=(n, 3))

            if not _validate_positions(perturbed):
                continue

            # Relax perturbed with soft potential
            try:
                e_new_soft, pos_new = _soft_relax(perturbed, n_exp, m_exp, c, soft_val,
                                                   maxiter=100, tol=1e-7)
                if not _validate_positions(pos_new):
                    continue

                e_new_hard = _hard_energy(pos_new, n_exp, m_exp, c)
                e_cur_hard = _hard_energy(current_pos, n_exp, m_exp, c)

                # Metropolis acceptance based on hard energy
                if e_new_hard < e_cur_hard:
                    current_pos = pos_new
                    if e_new_hard < best_e:
                        best_e = e_new_hard
                        best_pos = pos_new.copy()
                else:
                    # Accept with probability exp(-dE/T), T decreases with softness
                    T = 0.1 + 0.05 * (soft_val / 6.0)
                    dE = e_new_hard - e_cur_hard
                    if np.random.random() < np.exp(-dE / T):
                        current_pos = pos_new
            except Exception:
                pass

    # Final polish with hard potential
    if time.time() < deadline - 3.0:
        try:
            e_final, p_final = relax(best_pos, potential, kind)
            if np.isfinite(e_final) and _validate_positions(p_final):
                if e_final < best_e:
                    best_e = e_final
                    best_pos = p_final
        except Exception:
            pass

    # Additional random restarts with hard potential if time remains
    remaining = deadline - time.time()
    if remaining > 8.0:
        for i in range(10):
            if time.time() > deadline - 3.0:
                break
            # Perturb best
            s = best_pos + rng.normal(0, 0.3, size=(n, 3))
            if not _validate_positions(s):
                continue
            try:
                e, p = relax(s, potential, kind)
                if np.isfinite(e) and _validate_positions(p):
                    if e < best_e:
                        best_e = e
                        best_pos = p.copy()
            except Exception:
                pass

    if not _validate_positions(best_pos):
        return _random_compact(n, seed=77777, min_dist=0.9)

    return best_pos
# EVOLVE-BLOCK-END
