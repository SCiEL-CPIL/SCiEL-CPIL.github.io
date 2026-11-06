"""QAOA p=2 graph-dependent angle optimizer: vectorized simulation + multi-start Nelder-Mead."""
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE_MARGIN = 18.0


def _density(n, edges):
    if n < 2:
        return 0.0
    possible = n * (n - 1) / 2.0
    return len(edges) / possible


def _initial_angles(n, edges, p):
    d = _density(n, edges)
    d = min(max(d, 0.05), 1.0)

    # Heuristic based on graph density and known QAOA angle trends
    gamma1 = 0.20 + 0.60 * d
    beta1 = 0.35 - 0.15 * d
    gamma2 = 0.50 + 0.45 * d
    beta2 = 0.25 - 0.10 * d

    result = np.zeros((p, 2))
    if p >= 1:
        result[0, 0] = gamma1
        result[0, 1] = beta1
    if p >= 2:
        result[1, 0] = gamma2
        result[1, 1] = beta2
    for k in range(2, p):
        if k % 2 == 0:
            result[k, 0] = gamma2
            result[k, 1] = beta2
        else:
            result[k, 0] = gamma1
            result[k, 1] = beta1
    return result


def _precompute(n, edges):
    """Precompute c_diag and mixer index pairs for fast evaluation."""
    num_states = 1 << n

    # c_diag: cut value per basis state
    c_diag = np.zeros(num_states, dtype=float)
    for state in range(num_states):
        bits = [(state >> i) & 1 for i in range(n)]
        cut = 0
        for (u, v) in edges:
            if bits[u] != bits[v]:
                cut += 1
        c_diag[state] = cut

    # Mixer index pairs for vectorized mixer application
    mixer_pairs = []
    for i in range(n):
        mask = 1 << i
        idx0 = np.array([s0 for s0 in range(num_states) if (s0 & mask) == 0], dtype=np.int64)
        idx1 = idx0 | mask
        mixer_pairs.append((idx0, idx1))

    return c_diag, mixer_pairs


def _expected_cut_fast(n, edges, gammas, betas, c_diag, mixer_pairs):
    """Fast expected cut via vectorized state simulation."""
    num_states = 1 << n
    psi = np.full(num_states, 1.0 / np.sqrt(num_states), dtype=complex)

    for k in range(len(gammas)):
        gamma = gammas[k]
        beta = betas[k]

        # Cost layer: diagonal phase
        psi = psi * np.exp(-1j * gamma * c_diag)

        # Mixer layer: apply exp(-i*beta*sum X_i)
        c = np.cos(beta)
        s = np.sin(beta)
        for idx0, idx1 in mixer_pairs:
            v0 = psi[idx0].copy()
            v1 = psi[idx1].copy()
            psi[idx0] = c * v0 - 1j * s * v1
            psi[idx1] = -1j * s * v0 + c * v1

    probs = np.abs(psi) ** 2
    return float(np.sum(probs * c_diag))


def _nelder_mead_refine(params0, objective, maxfev=200, xatol=1e-8, fatol=1e-10):
    """Nelder-Mead refinement from a starting point."""
    try:
        res = minimize(
            objective,
            params0,
            method='Nelder-Mead',
            options={
                'maxiter': 800,
                'maxfev': maxfev,
                'xatol': xatol,
                'fatol': fatol,
                'adaptive': True,
            }
        )
        if res.fun < objective(params0):
            return np.clip(res.x, -np.pi, np.pi)
    except Exception:
        pass
    return np.clip(params0, -np.pi, np.pi)


def angles(n, edges, p):
    """Entry point. Returns (p, 2) array of (gamma, beta) in radians."""
    t_start = time.time()
    deadline = t_start + 150.0 - _DEADLINE_MARGIN

    try:
        init = _initial_angles(n, edges, p)
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))

    if n == 0 or len(edges) == 0:
        return init

    try:
        c_diag, mixer_pairs = _precompute(n, edges)
        opt_cut = int(np.max(c_diag)) if c_diag.size else 0
    except Exception:
        return init

    if opt_cut == 0:
        return init

    if n > 12:
        return init

    def objective(params):
        a = params.reshape(p, 2)
        a = np.clip(a, -np.pi, np.pi)
        try:
            v = _expected_cut_fast(n, edges, a[:, 0], a[:, 1], c_diag, mixer_pairs)
            return -v / opt_cut
        except Exception:
            return 1e10

    # Generate diverse starting points
    d = _density(n, edges)
    d = min(max(d, 0.05), 1.0)

    starts = [init.flatten().copy()]

    # Heuristic centers with wider exploration ranges
    g1_c, g1_r = 0.20 + 0.60 * d, 0.5
    b1_c, b1_r = 0.35 - 0.15 * d, 0.35
    g2_c, g2_r = 0.50 + 0.45 * d, 0.5
    b2_c, b2_r = 0.25 - 0.10 * d, 0.35

    rng = np.random.RandomState(713)
    n_extra_starts = 12
    for _ in range(n_extra_starts):
        g1 = rng.uniform(max(0.01, g1_c - g1_r), min(np.pi, g1_c + g1_r))
        b1 = rng.uniform(max(0.01, b1_c - b1_r), min(np.pi, b1_c + b1_r))
        g2 = rng.uniform(max(0.01, g2_c - g2_r), min(np.pi, g2_c + g2_r))
        b2 = rng.uniform(max(0.01, b2_c - b2_r), min(np.pi, b2_c + b2_r))
        s = np.array([g1, b1, g2, b2][:2 * p])
        starts.append(np.clip(s, -np.pi, np.pi))

    # Add some structured starts: symmetric, near-zero, near-pi/2
    structured = [
        np.array([0.3, 0.3, 0.5, 0.2][:2 * p]),
        np.array([0.1, 0.4, 0.6, 0.15][:2 * p]),
        np.array([0.5, 0.2, 0.3, 0.3][:2 * p]),
        np.array([np.pi/4, np.pi/6, np.pi/3, np.pi/8][:2 * p]),
    ]
    for s in structured:
        if len(s) == 2 * p:
            starts.append(np.clip(s, -np.pi, np.pi))

    # Evaluate all starts and keep the best few
    scored_starts = []
    for s in starts:
        if time.time() > deadline - _DEADLINE_MARGIN:
            break
        val = objective(s)
        scored_starts.append((val, s))

    scored_starts.sort(key=lambda x: x[0])  # ascending (most negative = best)

    # Refine top candidates
    best_params = scored_starts[0][1] if scored_starts else init.flatten()
    best_val = scored_starts[0][0] if scored_starts else objective(best_params)

    n_refine = min(4, len(scored_starts))
    for i in range(n_refine):
        if time.time() > deadline - _DEADLINE_MARGIN - 8.0:
            break
        val, s = scored_starts[i]
        refined = _nelder_mead_refine(s, objective, maxfev=200, xatol=1e-8, fatol=1e-10)
        r_val = objective(refined)
        if r_val < best_val:
            best_val = r_val
            best_params = refined

    # Final high-precision polish on the best
    if time.time() < deadline - _DEADLINE_MARGIN - 5.0:
        polished = _nelder_mead_refine(best_params, objective, maxfev=300, xatol=1e-10, fatol=1e-12)
        p_val = objective(polished)
        if p_val < best_val:
            best_params = polished

    return np.clip(best_params.reshape(p, 2), -np.pi, np.pi)
