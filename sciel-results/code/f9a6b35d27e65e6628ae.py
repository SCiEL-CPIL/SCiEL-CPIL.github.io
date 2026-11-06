"""QAOA p=2 graph-dependent angle optimizer: grid search + Nelder-Mead refinement."""
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE_MARGIN = 15.0  # seconds to spare before 150 s wall-clock kill


def _density(n, edges):
    if n < 2:
        return 0.0
    possible = n * (n - 1) / 2.0
    return len(edges) / possible


def _initial_angles(n, edges, p):
    """Direct explicit initial guess from graph structure."""
    d = _density(n, edges)
    d = min(max(d, 0.05), 1.0)

    gamma1 = 0.25 + 0.55 * d
    beta1 = 0.30 - 0.10 * d
    gamma2 = 0.45 + 0.60 * d
    beta2 = 0.22 - 0.08 * d

    result = np.zeros((p, 2))
    if p >= 1:
        result[0, 0] = gamma1
        result[0, 1] = beta1
    if p >= 2:
        result[1, 0] = gamma2
        result[1, 1] = beta2
    for k in range(2, p):
        if (k % 2 == 0):
            result[k, 0] = gamma2
            result[k, 1] = beta2
        else:
            result[k, 0] = gamma1
            result[k, 1] = beta1
    return result


def _expected_cut(n, edges, gammas, betas, c_diag):
    """Fast expected cut via exact state simulation (vectorized mixer)."""
    num_states = 1 << n
    psi = np.full(num_states, 1.0 / np.sqrt(num_states), dtype=complex)

    # Precompute mixer index pairs for each qubit
    mixer_pairs = []
    for i in range(n):
        mask = 1 << i
        idx0 = np.array([s0 for s0 in range(num_states) if (s0 & mask) == 0], dtype=np.int64)
        idx1 = idx0 | mask
        mixer_pairs.append((idx0, idx1))

    for k in range(len(gammas)):
        gamma = gammas[k]
        beta = betas[k]

        # Cost layer: diagonal
        psi = psi * np.exp(-1j * gamma * c_diag)

        # Mixer layer: product over i of (cos(beta) I - i sin(beta) X_i)
        c = np.cos(beta)
        s = np.sin(beta)
        for idx0, idx1 in mixer_pairs:
            v0 = psi[idx0].copy()
            v1 = psi[idx1].copy()
            psi[idx0] = c * v0 - 1j * s * v1
            psi[idx1] = -1j * s * v0 + c * v1

    probs = np.abs(psi) ** 2
    return float(np.sum(probs * c_diag))


def _c_diag(n, edges):
    """Precompute cut value per basis state."""
    num_states = 1 << n
    cd = np.zeros(num_states, dtype=float)
    for state in range(num_states):
        bits = [(state >> i) & 1 for i in range(n)]
        cut = 0
        for (u, v) in edges:
            if bits[u] != bits[v]:
                cut += 1
        cd[state] = cut
    return cd


def _exhaustive_maxcut(n, edges, c_diag):
    return int(np.max(c_diag)) if c_diag.size else 0


def _nelder_mead_refine(n, edges, p, angles0, c_diag, opt_cut, deadline):
    """Nelder-Mead refinement from a starting point."""
    if opt_cut == 0:
        return angles0

    def objective(params):
        """Negative ratio (we minimize)."""
        a = params.reshape(p, 2)
        a = np.clip(a, -np.pi, np.pi)
        try:
            v = _expected_cut(n, edges, a[:, 0], a[:, 1], c_diag)
            return -v / opt_cut
        except Exception:
            return 1e10

    x0 = angles0.flatten().copy()
    
    nm_options = {
        'maxiter': 500,
        'maxfev': 300,
        'xatol': 1e-6,
        'fatol': 1e-8,
        'adaptive': True,
    }
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options=nm_options)
        if res.success or res.fun < objective(x0):
            best_params = np.clip(res.x, -np.pi, np.pi).reshape(p, 2)
            return best_params
    except Exception:
        pass
    return angles0


def _grid_search(n, edges, p, c_diag, opt_cut, deadline):
    """Global grid search over a coarsely spaced, graph-scaled parameter space."""
    if opt_cut == 0:
        return _initial_angles(n, edges, p)

    d = _density(n, edges)
    d = min(max(d, 0.05), 1.0)

    # Define graph-scaled parameter ranges
    # gamma1: centered around 0.25+0.55*d, range ±0.4
    g1_center = 0.25 + 0.55 * d
    g1_range = 0.4
    # beta1: centered around 0.30-0.10*d, range ±0.25
    b1_center = 0.30 - 0.10 * d
    b1_range = 0.25
    # gamma2: centered around 0.45+0.60*d, range ±0.5
    g2_center = 0.45 + 0.60 * d
    g2_range = 0.5
    # beta2: centered around 0.22-0.08*d, range ±0.25
    b2_center = 0.22 - 0.08 * d
    b2_range = 0.25

    # Grid resolution: 7 points per parameter (49 total for p=2)
    n_grid = 7
    g1_vals = np.linspace(g1_center - g1_range, g1_center + g1_range, n_grid)
    b1_vals = np.linspace(b1_center - b1_range, b1_center + b1_range, n_grid)
    g2_vals = np.linspace(g2_center - g2_range, g2_center + g2_range, n_grid)
    b2_vals = np.linspace(b2_center - b2_range, b2_center + b2_range, n_grid)

    # Clip to valid range
    g1_vals = np.clip(g1_vals, 0.0, np.pi)
    b1_vals = np.clip(b1_vals, 0.0, np.pi)
    g2_vals = np.clip(g2_vals, 0.0, np.pi)
    b2_vals = np.clip(b2_vals, 0.0, np.pi)

    best = _initial_angles(n, edges, p)
    best_val = -1.0

    for g1 in g1_vals:
        if time.time() > deadline - _DEADLINE_MARGIN:
            break
        for b1 in b1_vals:
            if time.time() > deadline - _DEADLINE_MARGIN:
                break
            for g2 in g2_vals:
                if time.time() > deadline - _DEADLINE_MARGIN:
                    break
                for b2 in b2_vals:
                    if time.time() > deadline - _DEADLINE_MARGIN:
                        break
                    a = np.zeros((p, 2))
                    if p >= 1:
                        a[0, 0], a[0, 1] = g1, b1
                    if p >= 2:
                        a[1, 0], a[1, 1] = g2, b2
                    try:
                        v = _expected_cut(n, edges, a[:, 0], a[:, 1], c_diag)
                        ratio = v / opt_cut
                    except Exception:
                        continue
                    if ratio > best_val:
                        best_val = ratio
                        best = a.copy()
    return best


def angles(n, edges, p):
    """Entry point. Returns (p, 2) array of (gamma, beta) in radians."""
    t_start = time.time()
    deadline = t_start + 150.0 - _DEADLINE_MARGIN

    # Hard safety: if we somehow can't do anything, return the direct guess
    try:
        init = _initial_angles(n, edges, p)
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))

    if n == 0 or len(edges) == 0:
        return init

    # Precompute
    try:
        c_diag = _c_diag(n, edges)
        opt_cut = _exhaustive_maxcut(n, edges, c_diag)
    except Exception:
        return init

    if opt_cut == 0:
        return init

    # If n is large (state space too big), just return the direct guess.
    if n > 12:
        return init

    try:
        # Step 1: Global grid search to find a good starting point
        grid_best = _grid_search(n, edges, p, c_diag, opt_cut, deadline)
        
        # Step 2: Nelder-Mead refinement from the best grid point
        refined = _nelder_mead_refine(n, edges, p, grid_best, c_diag, opt_cut, deadline)
        
        # Step 3: Final high-precision polish with tighter tolerance
        nm_options_tight = {
            'maxiter': 300,
            'maxfev': 200,
            'xatol': 1e-8,
            'fatol': 1e-10,
            'adaptive': True,
        }
        
        def objective(params):
            a = params.reshape(p, 2)
            a = np.clip(a, -np.pi, np.pi)
            try:
                v = _expected_cut(n, edges, a[:, 0], a[:, 1], c_diag)
                return -v / opt_cut
            except Exception:
                return 1e10
        
        if time.time() < deadline - _DEADLINE_MARGIN - 5.0:
            try:
                res = minimize(objective, refined.flatten(), method='Nelder-Mead', options=nm_options_tight)
                if res.success or res.fun < objective(refined.flatten()):
                    refined = np.clip(res.x, -np.pi, np.pi).reshape(p, 2)
            except Exception:
                pass
        
        return np.clip(refined, -np.pi, np.pi)
    except Exception:
        return np.clip(init, -np.pi, np.pi)
