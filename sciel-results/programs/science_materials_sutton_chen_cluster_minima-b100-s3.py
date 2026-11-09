"""Combined: structure-based seeds + coarse-to-fine SC relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_KIND_PARAMS = {
    '12-6': (12, 6, 144.41),
    '9-6': (9, 6, 39.432),
    '10-8': (10, 8, 34.408),
}

_SPACING = {
    '12-6': 0.72,
    '9-6': 0.72,
    '10-8': 0.72,
}

_SOFT_PARAMS = {
    '12-6': {'n': 6.0, 'm': 3.0, 'c_scale': 0.5},
    '9-6': {'n': 4.5, 'm': 3.0, 'c_scale': 0.5},
    '10-8': {'n': 5.0, 'm': 4.0, 'c_scale': 0.5},
}


def _validate_positions(pos):
    if not np.all(np.isfinite(pos)):
        return False
    n = pos.shape[0]
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.35


def _fcc_cluster(n, spacing=0.72):
    if n <= 0:
        return np.empty((0, 3))
    basis = np.array([[0, 0, 0],
                      [0, 0.5, 0.5],
                      [0.5, 0, 0.5],
                      [0.5, 0.5, 0]], dtype=float)
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                for b in basis:
                    p = np.array([i, j, k], dtype=float) + b
                    sites.append(p)
    sites = np.array(sites) * spacing
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n]]


def _hcp_cluster(n, spacing=0.72):
    if n <= 0:
        return np.empty((0, 3))
    a = spacing
    c = spacing * np.sqrt(8.0 / 3.0)
    v1 = np.array([a, 0, 0])
    v2 = np.array([-a / 2, a * np.sqrt(3) / 2, 0])
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                base = i * v1 + j * v2
                z = k * c
                sites.append(base + np.array([0, 0, z]))
                sites.append(base + np.array([a / 2, a * np.sqrt(3) / 6, z + c / 2]))
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n]]


def _bcc_cluster(n, spacing=0.72):
    if n <= 0:
        return np.empty((0, 3))
    a = spacing
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                sites.append(np.array([i, j, k], dtype=float) * a)
                sites.append((np.array([i, j, k], dtype=float) + 0.5) * a)
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n]]


def _icosahedral_cluster(n, spacing=0.72):
    if n <= 0:
        return np.empty((0, 3))
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)

    points = [np.zeros(3)]
    if n <= 1:
        return np.array(points[:n])

    r1 = spacing
    for v in ico_verts:
        points.append(v * r1)
    if len(points) >= n:
        return np.array(points[:n])

    r2 = spacing * np.sqrt(3.0)
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            if 1.0 < d < 2.0:
                mid = (ico_verts[i] + ico_verts[j]) / 2.0
                mid = mid / np.linalg.norm(mid)
                points.append(mid * r2)
    if len(points) >= n:
        return np.array(points[:n])

    r3 = spacing * np.sqrt(5.0)
    remaining = n - len(points)
    if remaining > 0:
        for k in range(remaining):
            t = (k + 0.5) / remaining
            inc = np.pi * (3 - np.sqrt(5))
            z = 1 - 2 * t
            r = np.sqrt(1 - z * z)
            theta = inc * k
            p = np.array([r * np.cos(theta), r * np.sin(theta), z])
            points.append(p * r3)

    return np.array(points[:n])


def _random_compact(n, spacing=0.72, seed=42, min_dist_frac=0.7):
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    min_d = min_dist_frac * spacing
    points = []
    max_attempts = n * 500
    attempt = 0
    while len(points) < n and attempt < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < min_d:
                    ok = False
                    break
            if ok:
                points.append(p)
        attempt += 1
    while len(points) < n:
        if points:
            base = points[len(points) % len(points)]
        else:
            base = np.zeros(3)
        p = base + rng.normal(0, spacing * 0.3, 3)
        points.append(p)
    return np.array(points[:n])


def _soft_sc_energy_gradient(x, n, n_s, m_s, c_s):
    """
    Proper non-pairwise soft SC energy and gradient.
    E = sum_{i<j} r_ij^(-n_s) - c_s * sum_i sqrt(S_i)
    where S_i = sum_{j!=i} r_ij^(-m_s)
    """
    x = x.reshape(n, 3)
    i_idx, j_idx = np.triu_indices(n, k=1)

    diff = x[i_idx] - x[j_idx]
    r2 = np.sum(diff * diff, axis=1)
    r2 = np.maximum(r2, 1e-10)
    r = np.sqrt(r2)

    # Repulsive: sum_{i<j} r^(-n_s)
    r_inv_n = r ** (-n_s)
    E_rep = np.sum(r_inv_n)

    # S_i = sum_{j!=i} r_ij^(-m_s)
    r_inv_m = r ** (-m_s)
    S = np.zeros(n)
    np.add.at(S, i_idx, r_inv_m)
    np.add.at(S, j_idx, r_inv_m)

    sqrt_S = np.sqrt(np.maximum(S, 1e-30))
    E_att = -c_s * np.sum(sqrt_S)
    E = E_rep + E_att

    # Gradient
    grad = np.zeros((n, 3))

    # dE_rep/dx_i: d/dr = -n_s * r^(-n_s - 1), chain rule: * diff/r
    # dE_rep/dx_i = -n_s * r^(-n_s-1) * (x_i - x_j)/r = -n_s * r^(-n_s-2) * diff
    coeff_rep = -n_s * r ** (-n_s - 2)
    forces_rep = coeff_rep[:, np.newaxis] * diff
    np.add.at(grad, i_idx, forces_rep)
    np.add.at(grad, j_idx, -forces_rep)

    # Attractive: -c_s * sum_i sqrt(S_i)
    # d/dx_i [-c_s * sqrt(S_i)] = -c_s * (1/(2*sqrt(S_i))) * dS_i/dx_i
    # dS_i/dx_i = sum_{j!=i} d(r_ij^(-m_s))/dx_i = sum_{j!=i} (-m_s) * r_ij^(-m_s-1) * (x_i - x_j)/r_ij
    # = sum_{j!=i} (-m_s) * r_ij^(-m_s-2) * (x_i - x_j)
    # Similarly for dS_j/dx_i when i is in S_j's sum.
    #
    # For pair (i,j): contribution to grad from attractive term:
    #   d/dx_i [-c_s*(sqrt(S_i) + sqrt(S_j))]
    #   = -c_s * [1/(2*sqrt(S_i)) * dS_i/dx_i + 1/(2*sqrt(S_j)) * dS_j/dx_i]
    #   dS_i/dx_i includes term for pair (i,j): -m_s * r^(-m_s-2) * (x_i - x_j)
    #   dS_j/dx_i includes term for pair (j,i): -m_s * r^(-m_s-2) * (x_j - x_i) = m_s * r^(-m_s-2) * (x_i - x_j)
    #   Wait, S_j = sum_{k!=j} r_jk^(-m_s). dS_j/dx_i = d(r_ij^(-m_s))/dx_i = -m_s * r_ij^(-m_s-1) * (x_i - x_j)/r_ij
    #   = -m_s * r^(-m_s-2) * (x_i - x_j)
    #   And dS_i/dx_i for the pair (i,j) term: same form: -m_s * r^(-m_s-2) * (x_i - x_j)
    #
    # So for pair (i,j), contribution to grad[i]:
    #   -c_s * [1/(2*sqrt(S_i)) * (-m_s * r^(-m_s-2) * (x_i-x_j)) + 1/(2*sqrt(S_j)) * (-m_s * r^(-m_s-2) * (x_i-x_j))]
    #   = -c_s * (-m_s * r^(-m_s-2)) * (x_i-x_j) * [1/(2*sqrt(S_i)) + 1/(2*sqrt(S_j))]
    #   = c_s * m_s * r^(-m_s-2) * (x_i-x_j) * [1/(2*sqrt(S_i)) + 1/(2*sqrt(S_j))]
    #
    # Contribution to grad[j]:
    #   d/dx_j [-c_s*(sqrt(S_i)+sqrt(S_j))]
    #   = -c_s * [1/(2*sqrt(S_i)) * dS_i/dx_j + 1/(2*sqrt(S_j)) * dS_j/dx_j]
    #   dS_i/dx_j = d(r_ij^(-m_s))/dx_j = -m_s * r^(-m_s-1) * (x_j-x_i)/r = m_s * r^(-m_s-2) * (x_i-x_j)
    #   dS_j/dx_j for pair (j,i): -m_s * r^(-m_s-2) * (x_j-x_i) = m_s * r^(-m_s-2) * (x_i-x_j)
    #   So: -c_s * [1/(2*sqrt(S_i)) * m_s * r^(-m_s-2) * (x_i-x_j) + 1/(2*sqrt(S_j)) * m_s * r^(-m_s-2) * (x_i-x_j)]
    #   = -c_s * m_s * r^(-m_s-2) * (x_i-x_j) * [1/(2*sqrt(S_i)) + 1/(2*sqrt(S_j))]

    inv_2sqrt_S = np.zeros(n)
    mask = S > 1e-20
    inv_2sqrt_S[mask] = 1.0 / (2.0 * np.sqrt(S[mask]))

    coeff_att_base = c_s * m_s * r ** (-m_s - 2)
    weights = coeff_att_base * (inv_2sqrt_S[i_idx] + inv_2sqrt_S[j_idx])

    forces_att = weights[:, np.newaxis] * diff
    np.add.at(grad, i_idx, forces_att)
    np.add.at(grad, j_idx, -forces_att)

    return E, grad


def _try_relax(pos, potential, kind):
    try:
        if not _validate_positions(pos):
            return np.inf, None
        energy, relaxed = relax(pos, potential, kind)
        if np.isfinite(energy) and _validate_positions(relaxed):
            return energy, relaxed
    except Exception:
        pass
    return np.inf, None


def _soft_relax(pos, kind, time_limit=4.0):
    """Relax with softened SC potential using proper non-pairwise form."""
    start = time.time()
    n = len(pos)
    pos = pos.copy()

    sp = _SOFT_PARAMS.get(kind, {'n': 6.0, 'm': 3.0, 'c_scale': 0.5})
    n_s = sp['n']
    m_s = sp['m']
    _, _, c_true = _KIND_PARAMS[kind]
    c_s = c_true * sp['c_scale']

    def obj(x):
        if time.time() - start > time_limit:
            raise TimeoutError()
        E, g = _soft_sc_energy_gradient(x, n, n_s, m_s, c_s)
        return E, g.reshape(-1)

    try:
        E0, g0 = obj(pos.flatten())
        if not np.isfinite(E0):
            return np.inf, None

        res = minimize(
            fun=obj,
            x0=pos.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-5, 'maxfun': 600}
        )

        if np.isfinite(res.fun):
            relaxed = res.x.reshape(n, 3)
            if _validate_positions(relaxed):
                return res.fun, relaxed
    except (TimeoutError, Exception):
        pass
    return np.inf, None


def construct(instance):
    start_time = time.time()
    deadline = start_time + 290.0

    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    spacing = _SPACING.get(kind, 0.72)

    # Generate diverse structural seeds
    candidates = []
    candidates.append(_fcc_cluster(n, spacing))
    candidates.append(_fcc_cluster(n, spacing * 0.9))
    candidates.append(_fcc_cluster(n, spacing * 1.1))
    candidates.append(_hcp_cluster(n, spacing))
    candidates.append(_hcp_cluster(n, spacing * 0.9))
    candidates.append(_hcp_cluster(n, spacing * 1.1))
    candidates.append(_bcc_cluster(n, spacing))
    candidates.append(_bcc_cluster(n, spacing * 1.05))
    candidates.append(_icosahedral_cluster(n, spacing))
    candidates.append(_icosahedral_cluster(n, spacing * 1.1))
    candidates.append(_icosahedral_cluster(n, spacing * 0.9))

    # Random seeds with varied parameters
    for seed in range(1, 20):
        if time.time() > deadline - 15:
            break
        for md_frac in [0.65, 0.75, 0.85]:
            if time.time() > deadline - 15:
                break
            pos = _random_compact(n, spacing, seed=seed * 100 + int(md_frac * 100), min_dist_frac=md_frac)
            if _validate_positions(pos):
                candidates.append(pos)

    best_energy = np.inf
    best_positions = None

    for pos in candidates:
        if time.time() > deadline - 12:
            break

        # Phase 1: Soft relax
        soft_e, soft_pos = _soft_relax(pos, kind, time_limit=3.5)

        # Phase 2: True relax from soft-relaxed
        if soft_pos is not None and _validate_positions(soft_pos):
            true_e, true_pos = _try_relax(soft_pos, potential, kind)
            if true_e < best_energy:
                best_energy = true_e
                best_positions = true_pos.copy()
        else:
            # Direct true relax if soft failed
            true_e, true_pos = _try_relax(pos, potential, kind)
            if true_e < best_energy:
                best_energy = true_e
                best_positions = true_pos.copy()

    # Polish: perturb best and re-relax
    if best_positions is not None and time.time() < deadline - 8:
        rng = np.random.default_rng(12345)
        for trial in range(5):
            if time.time() > deadline - 8:
                break
            perturbed = best_positions + rng.normal(0, 0.05, best_positions.shape)
            e, p = _try_relax(perturbed, potential, kind)
            if e < best_energy:
                best_energy = e
                best_positions = p.copy()

    # Final polish
    if best_positions is not None and time.time() < deadline - 3:
        try:
            e2, p2 = relax(best_positions, potential, kind)
            if np.isfinite(e2) and _validate_positions(p2) and e2 < best_energy:
                best_energy = e2
                best_positions = p2.copy()
        except Exception:
            pass

    if best_positions is None:
        pos = _random_compact(n, spacing, seed=999)
        e, best_positions = _try_relax(pos, potential, kind)
        if best_positions is None:
            best_positions = pos

    return best_positions
# EVOLVE-BLOCK-END
