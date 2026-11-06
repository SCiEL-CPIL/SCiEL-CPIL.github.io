"""Model-and-optimize: parameterize the 3-layer Ry/CZ circuit and directly
minimize the exact mean energy gap per spin against the exact ground state."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import scipy.optimize

# Explicit global wall-clock deadline guard (evaluator kills at 150 s).
_GLOBAL_DEADLINE = time.time() + 142.0
_CACHE = {}
_GATE_CACHE = {}
_HAM_CACHE = {}


def _get_ry_indices(n, qubit):
    """Pre-compute index arrays for Ry on qubit."""
    key = ('ry', n, qubit)
    if key not in _GATE_CACHE:
        dim = 2 ** n
        mask = 1 << qubit
        idx = np.arange(dim)
        zero_mask = (idx & mask) == 0
        zero_idx = idx[zero_mask]
        one_idx = zero_idx | mask
        _GATE_CACHE[key] = (zero_idx, one_idx)
    return _GATE_CACHE[key]


def _get_cz_indices(n, a, b):
    """Pre-compute index array for CZ on qubits (a, b)."""
    key = ('cz', n, a, b)
    if key not in _GATE_CACHE:
        dim = 2 ** n
        mask = (1 << a) | (1 << b)
        idx = np.arange(dim)
        both_one = (idx & mask) == mask
        _GATE_CACHE[key] = np.where(both_one)[0]
    return _GATE_CACHE[key]


def _build_hamiltonian(n, coupling, field):
    """Build the full Hamiltonian matrix and return (H, E0, psi0)."""
    key = (int(n), float(coupling), float(field))
    if key in _HAM_CACHE:
        return _HAM_CACHE[key]
    
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=float)
    # ZZ terms
    for i in range(n - 1):
        ma = 1 << i
        mb = 1 << (i + 1)
        for state in range(dim):
            za = 1.0 if ((state & ma) == 0) else -1.0
            zb = 1.0 if ((state & mb) == 0) else -1.0
            H[state, state] += -coupling * za * zb
    # X terms
    for i in range(n):
        m = 1 << i
        for state in range(dim):
            H[state, state ^ m] += -field
    evals, evecs = np.linalg.eigh(H)
    E0 = float(evals[0])
    psi0 = evecs[:, 0]
    _HAM_CACHE[key] = (H, E0, psi0)
    return H, E0, psi0


def _circuit_state(theta, n, layers):
    """Simulate the circuit and return the state vector (real)."""
    dim = 2 ** n
    psi = np.zeros(dim, dtype=float)
    psi[0] = 1.0
    for l in range(layers):
        for i in range(n):
            t = theta[l, i]
            c = np.cos(t * 0.5)
            s = np.sin(t * 0.5)
            zero_idx, one_idx = _get_ry_indices(n, i)
            a0 = psi[zero_idx]
            a1 = psi[one_idx]
            psi[zero_idx] = c * a0 - s * a1
            psi[one_idx] = s * a0 + c * a1
        if l < layers - 1:
            for i in range(n - 1):
                both_one_idx = _get_cz_indices(n, i, i + 1)
                psi[both_one_idx] = -psi[both_one_idx]
    return psi


def _energy_of_state(psi, H):
    return float(psi @ H @ psi)


def _mean_field_init(n, coupling, field, layers):
    """
    Variational mean-field initialization.
    For each site, estimate the optimal Ry angle from local field balance.
    """
    theta = np.zeros((layers, n))
    
    # For open chain, edge sites have 1 neighbor, bulk sites have 2
    # The optimal angle for Ry in a single-qubit Hamiltonian -h_x X - h_z Z
    # satisfies tan(theta/2) = h_x / h_z (approximately)
    
    # Estimate neighbor magnetization ~ 0 for initial guess
    ratio = field / max(coupling, 1e-10)
    theta_last = np.arctan(ratio)
    theta_last = np.clip(theta_last, 0, np.pi / 2)
    
    # Earlier layers can be slightly different to allow entanglement building
    theta[0] = 0.7 * theta_last
    theta[1] = 0.85 * theta_last
    theta[2] = theta_last
    
    return theta


def _optimize_case(n, coupling, field, layers, time_budget):
    H, E0, psi0 = _build_hamiltonian(n, coupling, field)
    dim = 2 ** n
    n_params = layers * n

    def energy_obj(theta_flat):
        theta = theta_flat.reshape(layers, n)
        psi = _circuit_state(theta, n, layers)
        return (_energy_of_state(psi, H) - E0) / n

    def state_mismatch(theta_flat):
        """Least-squares objective: match circuit state to ground state."""
        theta = theta_flat.reshape(layers, n)
        psi = _circuit_state(theta, n, layers)
        # Ensure global phase alignment: project onto psi0
        overlap = np.dot(psi, psi0)
        if abs(overlap) > 1e-15:
            phase = overlap / abs(overlap)
            psi_aligned = psi / phase
        else:
            psi_aligned = psi
        diff = psi_aligned - psi0
        return diff

    bounds = [(-np.pi, np.pi)] * n_params
    rng = np.random.default_rng(42)

    # Build diverse starting points
    starts = []
    # Mean-field init
    starts.append(_mean_field_init(n, coupling, field, layers).reshape(-1))
    # Zero init
    starts.append(np.zeros(n_params))
    # Pi/2 init (fully transverse)
    starts.append(np.full(n_params, np.pi / 2))
    # Random inits
    n_random = 40
    for _ in range(n_random):
        starts.append(rng.uniform(-np.pi, np.pi, size=n_params))

    best_theta = np.zeros(n_params)
    best_val = np.inf
    t0 = time.time()

    for s in starts:
        if time.time() - t0 > time_budget - 5.0:
            break
        try:
            # Use least_squares for state matching (more robust landscape)
            res = scipy.optimize.least_squares(
                state_mismatch, s,
                bounds=bounds,
                method='trf',
                max_nfev=300,
                xtol=1e-12,
                ftol=1e-12,
            )
            e_val = energy_obj(res.x)
            if np.isfinite(e_val) and e_val < best_val:
                best_val = e_val
                best_theta = res.x.copy()
        except Exception:
            pass

    # Polish with L-BFGS-B on energy for final precision
    remaining = time_budget - (time.time() - t0)
    if remaining > 3.0 and best_val < np.inf:
        try:
            res_polish = scipy.optimize.minimize(
                energy_obj, best_theta, method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-16, "gtol": 1e-14},
            )
            if np.isfinite(res_polish.fun) and res_polish.fun < best_val:
                best_val = res_polish.fun
                best_theta = res_polish.x.copy()
        except Exception:
            pass

    # Polish pass 2
    remaining = time_budget - (time.time() - t0)
    if remaining > 2.0 and best_val < np.inf:
        try:
            res_polish2 = scipy.optimize.minimize(
                energy_obj, best_theta, method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 300, "ftol": 1e-16, "gtol": 1e-15},
            )
            if np.isfinite(res_polish2.fun) and res_polish2.fun < best_val:
                best_val = res_polish2.fun
                best_theta = res_polish2.x.copy()
        except Exception:
            pass

    # Polish pass 3 (ultra-fine)
    remaining = time_budget - (time.time() - t0)
    if remaining > 1.5 and best_val < np.inf:
        try:
            res_polish3 = scipy.optimize.minimize(
                energy_obj, best_theta, method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 200, "ftol": 1e-17, "gtol": 1e-16},
            )
            if np.isfinite(res_polish3.fun) and res_polish3.fun < best_val:
                best_val = res_polish3.fun
                best_theta = res_polish3.x.copy()
        except Exception:
            pass

    if not np.all(np.isfinite(best_theta)):
        best_theta = np.zeros(n_params)
    return best_theta.reshape(layers, n)


def angles(n, coupling, field, layers):
    key = (int(n), float(coupling), float(field), int(layers))
    if key in _CACHE:
        return _CACHE[key].copy()

    remaining = _GLOBAL_DEADLINE - time.time()
    if remaining <= 5.0:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        _CACHE[key] = theta
        return theta.copy()

    # Use more of the available time for optimization
    budget = min(remaining * 0.92, 45.0)

    try:
        theta = _optimize_case(n, coupling, field, layers, budget)
    except Exception:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2

    if not np.all(np.isfinite(theta)):
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2

    _CACHE[key] = theta
    return theta.copy()
# EVOLVE-BLOCK-END
