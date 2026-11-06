"""Optimized program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_H_cache = {}
_e0_cache = {}
_idx_cache = {}


def _build_H(n, coupling, field):
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    # Z terms: -coupling * sum Z_i Z_{i+1}
    for i in range(n - 1):
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> (i + 1)) & 1
            zz_val = 1.0 if bi == bj else -1.0
            H[state, state] += -coupling * zz_val
    # X terms: -field * sum X_i
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


def _get_indices(n):
    """Pre-compute index arrays for each qubit's 0 and 1 subspaces, and CZ masks."""
    key = n
    if key in _idx_cache:
        return _idx_cache[key]
    dim = 1 << n
    qubit_indices = []
    for q in range(n):
        mask = 1 << q
        idx_0 = np.array([i for i in range(dim) if (i & mask) == 0], dtype=np.int64)
        idx_1 = idx_0 + mask
        qubit_indices.append((idx_0, idx_1))
    # Pre-compute CZ masks for adjacent pairs
    cz_masks = []
    all_idx = np.arange(dim)
    for i in range(n - 1):
        m1 = 1 << i
        m2 = 1 << (i + 1)
        both_set = ((all_idx & m1) != 0) & ((all_idx & m2) != 0)
        cz_masks.append(both_set)
    _idx_cache[n] = (qubit_indices, cz_masks)
    return _idx_cache[n]


def _circuit_state(n, thetas, layers):
    """
    Vectorized application of the Ry/CZ circuit.
    thetas: (layers, n) array of rotation angles.
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.float64)
    state[0] = 1.0

    qubit_indices, cz_masks = _get_indices(n)

    for layer in range(layers):
        # Apply Ry to each qubit
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx_0, idx_1 = qubit_indices[q]
            vals_0 = state[idx_0].copy()
            vals_1 = state[idx_1].copy()
            state[idx_0] = c * vals_0 - s * vals_1
            state[idx_1] = s * vals_0 + c * vals_1

        # Apply CZ between layers (except after the last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                state[cz_masks[i]] *= -1.0

    return state


def _circuit_energy(n, coupling, field, thetas, layers):
    H = _build_H(n, coupling, field)
    psi = _circuit_state(n, thetas, layers)
    return float(np.dot(psi, H @ psi))


def _objective(params, n, coupling, field, layers, e0):
    thetas = params.reshape((layers, n))
    e_circuit = _circuit_energy(n, coupling, field, thetas, layers)
    return (e_circuit - e0) / n


def _mean_field_init(n, coupling, field, layers):
    """
    Generate initial angles based on a simple Mean Field approximation.
    For TFIM, the ground state is a product state in the X basis with angle theta
    where tan(theta) = field/coupling (roughly).
    """
    # Mean field angle for a single spin in a field h and coupling J
    # The effective field is roughly field + coupling * <Z>
    # For a simple initialization, we can use arctan2(field, coupling)
    theta_mf = np.arctan2(field, coupling)
    # The circuit starts at |0> (Z basis +1). Ry(theta) rotates towards X.
    # To get a state aligned with the mean field direction, we might need pi/2 - theta_mf or similar.
    # Let's try a few variations.
    inits = []
    # Variation 1: All layers same angle
    inits.append(np.full((layers, n), np.pi/2 - theta_mf))
    # Variation 2: Alternate angles
    alt = np.zeros((layers, n))
    for l in range(layers):
        if l % 2 == 0:
            alt[l, :] = np.pi/2 - theta_mf
        else:
            alt[l, :] = - (np.pi/2 - theta_mf)
    inits.append(alt)
    return inits


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    e0 = _exact_ground_energy(n, coupling, field)

    best_x = np.zeros(layers * n)
    best_val = _objective(best_x, n, coupling, field, layers, e0)

    # Pre-compute indices for this n
    _get_indices(n)

    # Determine number of restarts based on n and time budget
    # Increase restarts for better exploration
    if n <= 4:
        num_restarts = 30
    else:
        num_restarts = 20

    # Strategy: use diverse initializations
    # For transverse-field Ising, angles near pi/2 are often good
    # Mix of random and structured initializations
    mf_inits = _mean_field_init(n, coupling, field, layers)
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break

        if i == 0:
            # Start with all zeros (already done, but keep for consistency)
            continue
        elif i == 1:
            # All pi/2 (common in TFIM ground states)
            x_init = np.full(layers * n, np.pi / 2)
        elif i == 2:
            # All -pi/2
            x_init = np.full(layers * n, -np.pi / 2)
        elif i == 3:
            # Alternating pi/2, -pi/2
            x_init = np.array([np.pi / 2 if (j % 2 == 0) else -np.pi / 2 for j in range(layers * n)])
        elif i == 4:
            # Layer-dependent: first layer pi/2, others 0
            x_init = np.zeros(layers * n)
            x_init[:n] = np.pi / 2
        elif i == 5:
            # Layer-dependent: last layer pi/2, others 0
            x_init = np.zeros(layers * n)
            x_init[-n:] = np.pi / 2
        elif i == 6:
            # Strong coupling: small angles (Z-dominated)
            x_init = np.full(layers * n, 0.1)
        elif i == 7:
            # Strong field: larger angles (X-dominated)
            x_init = np.full(layers * n, np.pi / 3)
        elif i == 8:
            # Gradient-like: increasing angles across layers
            x_init = np.zeros(layers * n)
            for l in range(layers):
                x_init[l * n:(l+1) * n] = (l + 1) * np.pi / (layers + 1)
        elif i == 9:
            # Alternating layers: pi/2, 0, pi/2
            x_init = np.zeros(layers * n)
            for l in range(layers):
                if l % 2 == 0:
                    x_init[l * n:(l+1) * n] = np.pi / 2
        elif i == 10:
            # Mean Field Init 1
            x_init = mf_inits[0].flatten()
        elif i == 11:
            # Mean Field Init 2
            x_init = mf_inits[1].flatten()
        else:
            # Random initializations with different seeds
            np.random.seed(1000 + i)
            x_init = np.random.uniform(-np.pi, np.pi, size=(layers * n))

        def obj(p):
            return _objective(p, n, coupling, field, layers, e0)

        try:
            res = minimize(obj, x_init, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 50})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Final polish with tighter tolerances if time remains
    if time.time() < deadline - 15.0:
        try:
            def obj(p):
                return _objective(p, n, coupling, field, layers, e0)
            res = minimize(obj, best_x, method='L-BFGS-B',
                          options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 100})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Second polish with different method if time remains
    if time.time() < deadline - 10.0:
        try:
            def obj(p):
                return _objective(p, n, coupling, field, layers, e0)
            res = minimize(obj, best_x, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Third polish: COBYLA for constraint-free local optimization
    if time.time() < deadline - 5.0:
        try:
            def obj(p):
                return _objective(p, n, coupling, field, layers, e0)
            res = minimize(obj, best_x, method='COBYLA',
                          options={'maxiter': 2000, 'rhobeg': 1e-3, 'tol': 1e-15})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Final ultra-precise polish
    if time.time() < deadline - 2.0:
        try:
            def obj(p):
                return _objective(p, n, coupling, field, layers, e0)
            res = minimize(obj, best_x, method='L-BFGS-B',
                          options={'maxiter': 10000, 'ftol': 1e-20, 'gtol': 1e-16, 'maxls': 200})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    return best_x.reshape((layers, n))
# EVOLVE-BLOCK-END
