"""Rewritten program: nonlinear least-squares state-matching with mean-field init."""
import numpy as np
import time
from scipy.optimize import minimize, least_squares

_H_cache = {}
_e0_cache = {}
_psi0_cache = {}

def _build_H(n, coupling, field):
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    # Z_i Z_{i+1} terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> (i + 1)) & 1
            zz_val = 1.0 if bi == bj else -1.0
            H[state, state] += -coupling * zz_val
    # X_i terms: -field * X_i
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[state, flipped] += -field
    _H_cache[key] = H
    return H

def _exact_ground_state(n, coupling, field):
    key = (n, coupling, field)
    if key in _e0_cache:
        return _e0_cache[key]
    H = _build_H(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    e0 = eigvals[0]
    psi0 = eigvecs[:, 0]
    # Make deterministic: ensure largest magnitude component is real and positive
    idx_max = np.argmax(np.abs(psi0))
    phase = psi0[idx_max] / np.abs(psi0[idx_max])
    psi0 = psi0 / phase
    _e0_cache[key] = (e0, psi0)
    return e0, psi0

def _circuit_state(n, thetas, layers):
    """Compute the circuit state vector efficiently using tensor operations."""
    dim = 1 << n
    # Start with |0...0>
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        # Apply Ry(theta_q) to each qubit q
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            # Ry = [[c, -s], [s, c]]
            # For each basis state, if qubit q is 0: new = c*old
            # If qubit q is 1: new = s*old, and we swap
            new_state = np.zeros(dim, dtype=np.complex128)
            for idx in range(dim):
                bit = (idx >> q) & 1
                if bit == 0:
                    new_state[idx] += c * state[idx]
                    new_state[idx ^ (1 << q)] += s * state[idx]
                else:
                    new_state[idx] += c * state[idx]
                    new_state[idx ^ (1 << q)] += s * state[idx]
            state = new_state
        
        # Apply CZ between adjacent pairs (if not last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                m1 = 1 << i
                m2 = 1 << (i + 1)
                for idx in range(dim):
                    if (idx & m1) and (idx & m2):
                        state[idx] *= -1.0
    
    return state

def _circuit_energy(n, coupling, field, thetas, layers):
    H = _build_H(n, coupling, field)
    psi = _circuit_state(n, thetas, layers)
    return np.real(np.conj(psi) @ H @ psi)

def _mean_field_init(n, coupling, field, layers):
    """
    Initialize angles using a variational mean-field / product-state approximation.
    For each qubit, we can think of the effective field as the transverse field
    plus the mean-field contribution from neighbors.
    
    For a product state |theta_1/2> ... |theta_n/2>, the energy is:
    E = sum_i [-field * sin(theta_i)] + sum_i -coupling * cos(theta_i) cos(theta_{i+1})
    (since <Z_i Z_{i+1}> = cos(theta_i) cos(theta_{i+1}) for Ry rotations from |0>)
    
    Actually, Ry(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>
    <Z> = cos^2(theta/2) - sin^2(theta/2) = cos(theta)
    <X> = 2 cos(theta/2) sin(theta/2) = sin(theta)
    
    So E = -field * sum sin(theta_i) - coupling * sum cos(theta_i) cos(theta_{i+1})
    
    We minimize this with respect to each theta_i.
    """
    thetas = np.zeros((layers, n))
    
    # Solve the mean-field problem: minimize E = -field*sum(sin) - coupling*sum(cos*cos)
    # dE/dtheta_i = -field*cos(theta_i) + coupling*sin(theta_i)*(cos(theta_{i-1}) + cos(theta_{i+1}))
    # (with boundary terms for i=0 and i=n-1)
    
    # Use simple iterative relaxation
    theta = np.zeros(n)
    for _ in range(1000):
        theta_old = theta.copy()
        for i in range(n):
            left_cos = np.cos(theta[i-1]) if i > 0 else 0.0
            right_cos = np.cos(theta[i+1]) if i < n-1 else 0.0
            # dE/dtheta_i = -field*cos(theta_i) + coupling*sin(theta_i)*(left_cos + right_cos) = 0
            # tan(theta_i) = field / (coupling*(left_cos + right_cos))
            denom = coupling * (left_cos + right_cos)
            if abs(denom) < 1e-12:
                theta[i] = np.pi / 2.0
            else:
                theta[i] = np.arctan2(field, denom)
        if np.max(np.abs(theta - theta_old)) < 1e-10:
            break
    
    # Distribute the mean-field angles across layers
    # A simple heuristic: put all the rotation in the first layer, or spread evenly
    # Better: for 3 layers, we can try to distribute. But since CZ gates mix things,
    # let's just put the mean-field angle in each layer divided by layers.
    for l in range(layers):
        thetas[l, :] = theta / layers
    
    return thetas

def _objective_energy(params, n, coupling, field, layers, e0):
    thetas = params.reshape((layers, n))
    e_circuit = _circuit_energy(n, coupling, field, thetas, layers)
    return (e_circuit - e0) / n

def _residuals_state(params, n, coupling, field, layers, psi0):
    """
    Return residuals for least_squares: match circuit state to ground state.
    We use the real and imaginary parts of (psi_circuit - psi0) as residuals.
    """
    thetas = params.reshape((layers, n))
    psi = _circuit_state(n, thetas, layers)
    # Return flattened real and imaginary parts of the difference
    diff = psi - psi0
    return np.concatenate([np.real(diff), np.imag(diff)])

def _objective_state(params, n, coupling, field, layers, psi0):
    """Objective for minimize: ||psi_circuit - psi0||^2 / n"""
    thetas = params.reshape((layers, n))
    psi = _circuit_state(n, thetas, layers)
    diff = psi - psi0
    return np.real(np.vdot(diff, diff)) / n

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0
    
    e0, psi0 = _exact_ground_state(n, coupling, field)
    total_params = layers * n
    
    # Generate diverse initial guesses
    inits = []
    
    # 1. Mean-field initialization
    mf_thetas = _mean_field_init(n, coupling, field, layers)
    inits.append(mf_thetas.flatten())
    
    # 2. Zero initialization
    inits.append(np.zeros(total_params))
    
    # 3. Small random initializations
    for seed in range(5):
        np.random.seed(seed * 1000 + 42)
        x = np.random.uniform(-1.0, 1.0, size=total_params)
        inits.append(x)
    
    # 4. Mean-field with different layer distributions
    mf_thetas2 = np.zeros((layers, n))
    theta_mf = mf_thetas[0, :]  # The converged mean-field angles
    # Put all in last layer
    mf_thetas2[layers-1, :] = theta_mf
    inits.append(mf_thetas2.flatten())
    
    # Put half in first, half in last
    mf_thetas3 = np.zeros((layers, n))
    mf_thetas3[0, :] = theta_mf / 2.0
    mf_thetas3[layers-1, :] = theta_mf / 2.0
    inits.append(mf_thetas3.flatten())
    
    best_val = np.inf
    best_x = inits[0].copy()
    
    # First pass: use least_squares for state matching (more robust landscape)
    for i, x_init in enumerate(inits):
        if time.time() > deadline:
            break
        try:
            res = least_squares(
                _residuals_state,
                x_init,
                args=(n, coupling, field, layers, psi0),
                method='trf',
                bounds=([-np.pi] * total_params, [np.pi] * total_params),
                max_nfev=2000,
                xtol=1e-14,
                ftol=1e-14,
                gtol=1e-14
            )
            # Evaluate actual energy-based objective
            val = _objective_energy(res.x, n, coupling, field, layers, e0)
            if val < best_val:
                best_val = val
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Second pass: polish with L-BFGS-B on energy objective
    for i, x_init in enumerate(inits[:3]):  # Use best few inits
        if time.time() > deadline:
            break
        # Also use the best found so far as an init
        if i == 0:
            x_init = best_x.copy()
        try:
            res = minimize(
                _objective_energy,
                x_init,
                args=(n, coupling, field, layers, e0),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Final polish from best_x
    if time.time() < deadline - 5.0:
        try:
            res = minimize(
                _objective_energy,
                best_x,
                args=(n, coupling, field, layers, e0),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-14}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x.reshape((layers, n))
