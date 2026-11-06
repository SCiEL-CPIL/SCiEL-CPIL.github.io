import numpy as np
import time
from scipy.optimize import minimize, least_squares

_H_cache = {}
_e0_cache = {}
_psig_cache = {}
_phase_cache = {}
_idx_cache = {}


def _build_H(n, coupling, field):
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n - 1):
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> (i + 1)) & 1
            zz_val = 1.0 if bi == bj else -1.0
            H[state, state] += -coupling * zz_val
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[state, flipped] += -field
    _H_cache[key] = H
    return H


def _exact_ground_energy(n, coupling, field):
    key = (n, coupling, field)
    if key in _e0_cache:
        return _e0_cache[key]
    H = _build_H(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    e0 = float(eigvals[0])
    _e0_cache[key] = e0
    return e0


def _exact_ground_state(n, coupling, field):
    key = (n, coupling, field)
    if key in _psig_cache:
        return _psig_cache[key]
    H = _build_H(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    idx_min = np.argmin(eigvals)
    psi_g = eigvecs[:, idx_min].real
    norm = np.linalg.norm(psi_g)
    psi_g = psi_g / norm
    for i in range(len(psi_g)):
        if abs(psi_g[i]) > 1e-10:
            if psi_g[i] < 0:
                psi_g = -psi_g
            break
    _psig_cache[key] = psi_g
    return psi_g


def _get_phases(n):
    key = n
    if key in _phase_cache:
        return _phase_cache[key]
    dim = 1 << n
    phases = []
    idx = np.arange(dim)
    for i in range(n - 1):
        mask1 = 1 << i
        mask2 = 1 << (i + 1)
        phase = np.ones(dim, dtype=np.float64)
        mask = ((idx & mask1) != 0) & ((idx & mask2) != 0)
        phase[mask] = -1.0
        phases.append(phase)
    _phase_cache[key] = phases
    return phases


def _get_idx_pairs(n):
    key = n
    if key in _idx_cache:
        return _idx_cache[key]
    dim = 1 << n
    idx_pairs = {}
    idx = np.arange(dim)
    for q in range(n):
        mask = 1 << q
        idx_0 = idx[(idx & mask) == 0]
        idx_1 = idx_0 + mask
        idx_pairs[q] = (idx_0, idx_1)
    _idx_cache[key] = idx_pairs
    return idx_pairs


def _circuit_state_vec(n, thetas, layers):
    dim = 1 << n
    psi = np.zeros(dim, dtype=np.float64)
    psi[0] = 1.0
    phases = _get_phases(n)
    idx_pairs = _get_idx_pairs(n)
    for layer in range(layers):
        for q in range(n):
            theta = float(thetas[layer, q])
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx_0, idx_1 = idx_pairs[q]
            v0 = psi[idx_0].copy()
            v1 = psi[idx_1].copy()
            psi[idx_0] = c * v0 - s * v1
            psi[idx_1] = s * v0 + c * v1
        if layer < layers - 1:
            for i in range(n - 1):
                psi *= phases[i]
    return psi


def _circuit_energy_vec(n, coupling, field, thetas, layers):
    H = _build_H(n, coupling, field)
    psi = _circuit_state_vec(n, thetas, layers)
    return float(psi @ (H @ psi))


def _symmap(params, n, layers):
    half_n = n // 2
    out = np.zeros((layers, n))
    for l in range(layers):
        for i in range(half_n):
            v = params[l * half_n + i]
            out[l, i] = v
            out[l, n - 1 - i] = v
    return out


def _obj_sym(params, n, coupling, field, layers, e0):
    th = _symmap(params, n, layers)
    e = _circuit_energy_vec(n, coupling, field, th, layers)
    return (e - e0) / n


def _residuals_sym(params, n, coupling, field, layers, psi_g):
    th = _symmap(params, n, layers)
    psi_c = _circuit_state_vec(n, th, layers)
    dot = np.dot(psi_c, psi_g)
    if dot < 0:
        return psi_c + psi_g
    else:
        return psi_c - psi_g


def _residuals_full(params, n, coupling, field, layers, psi_g):
    th = params.reshape(layers, n)
    psi_c = _circuit_state_vec(n, th, layers)
    dot = np.dot(psi_c, psi_g)
    if dot < 0:
        return psi_c + psi_g
    else:
        return psi_c - psi_g


def _meanfield_init(n, coupling, field, layers):
    """Variational mean-field: each spin in superposition with angle theta
    such that the single-spin ground state of -field*X - coupling*Z has
    the right mixing. For a single spin in field B_x with coupling J:
    the effective transverse field tilts the spin. Use arctan2(B, J_eff)."""
    # For open chain, edge spins feel only field; interior feel field + coupling
    # Simplest: use the ratio field/coupling for all
    if coupling == 0:
        theta_all = np.pi / 2
    else:
        # Mean-field: each spin sees effective field from neighbors
        # For a uniform guess, use arctan2(field, coupling)
        theta_all = np.arctan2(field, coupling)
    # Refine: edge spins have no neighbor, so pure transverse field
    # theta_edge = pi/2; interior theta = arctan2(field, coupling)
    half_n = n // 2
    init = np.zeros(layers * half_n)
    for l in range(layers):
        for i in range(half_n):
            if i == 0 or i == half_n - 1:
                # Edge spins: mostly transverse
                init[l * half_n + i] = 0.6 * theta_all + 0.4 * (np.pi / 2)
            else:
                init[l * half_n + i] = theta_all
    return init


def _heuristic_inits_sym(n, coupling, field, layers):
    half_n = n // 2
    inits = []
    inits.append(np.zeros(layers * half_n))
    inits.append(np.full(layers * half_n, np.pi / 2))
    inits.append(np.full(layers * half_n, np.pi))
    ratio = field / coupling if coupling != 0 else field
    theta_est = np.arctan(ratio)
    inits.append(np.full(layers * half_n, theta_est))
    inits.append(np.full(layers * half_n, theta_est * 0.8))
    inits.append(np.full(layers * half_n, theta_est * 1.2))
    inits.append(np.full(layers * half_n, 0.3))
    inits.append(np.full(layers * half_n, 0.7))
    inits.append(np.full(layers * half_n, 1.0))
    # Mean-field init
    inits.append(_meanfield_init(n, coupling, field, layers))
    # Alternating patterns
    alt1 = np.zeros(layers * half_n)
    for l in range(layers):
        for i in range(half_n):
            alt1[l * half_n + i] = theta_est if i % 2 == 0 else theta_est * 0.5
    inits.append(alt1)
    # Random inits
    rng = np.random.RandomState(42)
    for _ in range(12):
        inits.append(rng.uniform(-np.pi, np.pi, size=(layers * half_n)))
    # Layer-dependent
    for l_scale in [0.5, 0.8, 1.0, 1.2, 1.5]:
        arr = np.zeros(layers * half_n)
        for l in range(layers):
            arr[l * half_n:(l + 1) * half_n] = theta_est * l_scale
        inits.append(arr)
    return inits


def angles(n, coupling, field, layers):
    start = time.time()
    deadline = start + 145.0
    e0 = _exact_ground_energy(n, coupling, field)
    psi_g = _exact_ground_state(n, coupling, field)

    use_sym = (n % 2 == 0)
    if use_sym:
        dim_params = layers * (n // 2)
        obj = lambda p: _obj_sym(p, n, coupling, field, layers, e0)
        res_func = lambda p: _residuals_sym(p, n, coupling, field, layers, psi_g)
        inits = _heuristic_inits_sym(n, coupling, field, layers)
    else:
        dim_params = layers * n
        obj = lambda p: (lambda t: (_circuit_energy_vec(n, coupling, field, t, layers) - e0) / n)(p.reshape(layers, n))
        res_func = lambda p: _residuals_full(p, n, coupling, field, layers, psi_g)
        inits = [np.zeros(dim_params), np.full(dim_params, np.pi / 2)]
        rng = np.random.RandomState(11)
        for _ in range(10):
            inits.append(rng.uniform(-np.pi, np.pi, size=(dim_params)))

    best_x = inits[0].copy()
    best_val = obj(best_x)

    # Evaluate all initial points
    for x0 in inits:
        if time.time() > deadline:
            break
        v = obj(x0)
        if v < best_val:
            best_val = v
            best_x = x0.copy()

    # Sort inits by objective value
    scored_inits = []
    for x0 in inits:
        v = obj(x0)
        scored_inits.append((v, x0))
    scored_inits.sort(key=lambda t: t[0])

    # Phase 1: Multi-start Nelder-Mead on energy from top initial points
    n_starts = min(8, len(scored_inits))
    for idx in range(n_starts):
        if time.time() > deadline - 40.0:
            break
        x0 = scored_inits[idx][1].copy()
        try:
            res = minimize(
                obj,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-12,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Phase 2: Least-squares state-vector matching from best energy result
    if time.time() < deadline - 25.0:
        try:
            lsq_res = least_squares(
                res_func,
                best_x,
                method='trf',
                max_nfev=2000,
                xtol=1e-14,
                ftol=1e-14,
                gtol=1e-14
            )
            e_lsq = obj(lsq_res.x)
            if e_lsq < best_val:
                best_val = e_lsq
                best_x = lsq_res.x.copy()
        except Exception:
            pass

    # Phase 3: Multi-start least-squares from top initial points (different landscape)
    n_lsq_starts = min(5, len(scored_inits))
    for idx in range(n_lsq_starts):
        if time.time() > deadline - 15.0:
            break
        x0 = scored_inits[idx][1].copy()
        try:
            lsq_res = least_squares(
                res_func,
                x0,
                method='trf',
                max_nfev=1500,
                xtol=1e-13,
                ftol=1e-13,
                gtol=1e-13
            )
            e_lsq = obj(lsq_res.x)
            if e_lsq < best_val:
                best_val = e_lsq
                best_x = lsq_res.x.copy()
        except Exception:
            pass

    # Phase 4: L-BFGS-B with finite-difference gradient from best
    if time.time() < deadline - 10.0:
        def _grad_fd(p):
            eps = 1e-7
            g = np.zeros_like(p)
            for k in range(p.size):
                pp = p.copy()
                pm = p.copy()
                pp[k] += eps
                pm[k] -= eps
                g[k] = (obj(pp) - obj(pm)) / (2.0 * eps)
            return g

        try:
            res = minimize(
                obj,
                best_x,
                method='L-BFGS-B',
                jac=_grad_fd,
                options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 120}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Phase 5: Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            res = minimize(
                obj,
                best_x,
                method='L-BFGS-B',
                jac=_grad_fd,
                options={'maxiter': 2000, 'ftol': 1e-17, 'gtol': 1e-15, 'maxls': 200}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Phase 6: One more Nelder-Mead polish with very tight tolerance
    if time.time() < deadline - 2.0:
        try:
            res = minimize(
                obj,
                best_x,
                method='Nelder-Mead',
                options={
                    'maxiter': 20000,
                    'xatol': 1e-15,
                    'fatol': 1e-15,
                    'adaptive': True
                }
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    if use_sym:
        full = _symmap(best_x, n, layers)
    else:
        full = best_x.reshape((layers, n))

    if not np.all(np.isfinite(full)):
        return np.zeros((layers, n))
    return full
