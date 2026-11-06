"""Optimize 3-layer Ry/CZ circuit parameters for transverse-field Ising chains."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Caches for Hamiltonian, ground energy, and CZ masks
_H_cache = {}
_e0_cache = {}
_cz_mask_cache = {}
_idx_cache = {}


def _build_hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian matrix for open chain with caching."""
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    
    # ZZ terms: -coupling * sum Z_i Z_(i+1)
    # For each basis state, compute the ZZ contribution
    for x in range(dim):
        zz_val = 0.0
        for i in range(n - 1):
            xi = (x >> i) & 1
            xj = (x >> (i + 1)) & 1
            if xi == xj:
                zz_val += 1.0
            else:
                zz_val -= 1.0
        H[x, x] -= coupling * zz_val
    
    # X terms: -field * sum X_i
    for i in range(n):
        for x in range(dim):
            xp = x ^ (1 << i)
            H[xp, x] -= field
    
    _H_cache[key] = H
    return H


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground energy using eigendecomposition with caching."""
    key = (n, coupling, field)
    if key in _e0_cache:
        return _e0_cache[key]
    
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    e0 = eigvals[0]
    _e0_cache[key] = e0
    return e0


def _get_cz_masks(n):
    """Pre-compute bitmasks for CZ operations."""
    key = n
    if key in _cz_mask_cache:
        return _cz_mask_cache[key]
    
    dim = 1 << n
    masks = []
    for i in range(n - 1):
        mask = np.ones(dim, dtype=np.float64)
        for x in range(dim):
            bi = (x >> i) & 1
            bj = (x >> (i + 1)) & 1
            if bi == 1 and bj == 1:
                mask[x] = -1.0
        masks.append(mask)
    
    _cz_mask_cache[key] = masks
    return masks


def _get_idx_cache(n):
    """Pre-compute index arrays for each qubit."""
    key = n
    if key in _idx_cache:
        return _idx_cache[key]
    
    dim = 1 << n
    idx_0 = {}
    idx_1 = {}
    for q in range(n):
        mask = 1 << q
        idx_0[q] = np.array([i for i in range(dim) if (i & mask) == 0], dtype=np.int32)
        idx_1[q] = idx_0[q] + mask
        idx_1[q] = idx_1[q].astype(np.int32)
    
    _idx_cache[key] = (idx_0, idx_1)
    return _idx_cache[key]


def _apply_circuit_state(theta, n, layers):
    """
    Apply the circuit to |0>^n and return the state vector.
    Uses vectorized operations for speed.
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    cz_masks = _get_cz_masks(n)
    idx_0, idx_1 = _get_idx_cache(n)
    
    for l in range(layers):
        # Apply Ry to every qubit
        for i in range(n):
            theta_i = theta[l, i]
            c = np.cos(theta_i / 2)
            s = np.sin(theta_i / 2)
            
            i0 = idx_0[i]
            i1 = idx_1[i]
            
            # For bit i = 0: |0> -> c|0> + s|1>
            # For bit i = 1: |1> -> -s|0> + c|1>
            # new_0 = c * old_0 - s * old_1
            # new_1 = s * old_0 + c * old_1
            
            old_0 = state[i0].copy()
            old_1 = state[i1].copy()
            
            state[i0] = c * old_0 - s * old_1
            state[i1] = s * old_0 + c * old_1
        
        # Apply CZ on adjacent pairs if not final layer
        if l < layers - 1:
            for i in range(n - 1):
                state *= cz_masks[i]
    
    return state


def _circuit_energy(theta, n, coupling, field, H):
    """Compute <psi|H|psi> where psi is the circuit state."""
    psi = _apply_circuit_state(theta, n, layers=3)
    energy = np.vdot(psi, H @ psi).real
    return energy


def _objective_flat(theta_flat, n, coupling, field, H, ground_energy):
    """Objective function: (circuit energy - ground energy) / n"""
    theta = theta_flat.reshape(3, n)
    energy = _circuit_energy(theta, n, coupling, field, H)
    return (energy - ground_energy) / n


def _fit_mps_init(n, coupling, field, layers, H, ground_energy, psi0):
    """
    Fit the circuit parameters by minimizing the energy, but initialize
    using a heuristic based on the ground state structure.
    
    This uses the ground state to inform initial angles:
    - For each qubit, estimate the local magnetization <X_i> and <Z_i>
    - Use these to set initial Ry angles
    """
    # Compute local expectation values from the ground state
    dim = 1 << n
    probs = np.abs(psi0) ** 2
    
    # Estimate local X and Z expectations
    local_x = np.zeros(n)
    local_z = np.zeros(n)
    
    for i in range(n):
        # <Z_i> = sum_x (-1)^x_i |psi_x|^2
        for x in range(dim):
            bit_i = (x >> i) & 1
            if bit_i == 0:
                local_z[i] += probs[x]
            else:
                local_z[i] -= probs[x]
        
        # <X_i> = sum_x,y psi_x^* psi_y <x|X_i|y>
        # X_i flips bit i
        for x in range(dim):
            y = x ^ (1 << i)
            local_x[i] += 2 * np.real(np.conj(psi0[x]) * psi0[y])
    
    # Convert to initial angles
    # Ry(theta) |0> = cos(theta/2)|0> + sin(theta/2)|1>
    # So <Z> = cos(theta) approximately for a single qubit
    # <X> = sin(theta) approximately
    # theta = atan2(<X>, <Z>)
    
    init_theta = np.zeros((layers, n))
    
    for i in range(n):
        # Clamp to avoid numerical issues
        z_val = np.clip(local_z[i], -1.0, 1.0)
        x_val = np.clip(local_x[i], -1.0, 1.0)
        
        theta_est = np.arctan2(x_val, z_val)
        # Ensure in [-pi, pi]
        theta_est = np.arctan2(np.sin(theta_est), np.cos(theta_est))
        
        # Distribute the angle across layers
        # First layer gets most of the rotation
        init_theta[0, i] = theta_est * 0.6
        init_theta[1, i] = theta_est * 0.25
        init_theta[2, i] = theta_est * 0.15
    
    return init_theta


def angles(n, coupling, field, layers):
    """
    Optimize parameters of a fixed three-layer Ry/CZ circuit to approximate 
    ground states of open transverse-field Ising chains.
    
    Args:
        n: number of qubits (4 or 6)
        coupling: Ising coupling (1.0)
        field: transverse field (0.45, 1.0, 1.65)
        layers: number of layers (3)
    
    Returns:
        (layers, n) array of angles in radians
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Build Hamiltonian and compute exact ground energy
    H = _build_hamiltonian(n, coupling, field)
    ground_energy = _exact_ground_energy(n, coupling, field)
    
    dim = 1 << n
    total_params = layers * n
    
    best_theta = None
    best_obj = np.inf
    
    # Get ground state for MPS-inspired initialization
    eigvals, eigvecs = np.linalg.eigh(H)
    psi0 = eigvecs[:, 0]
    
    # Generate multiple initial guesses
    initial_guesses = []
    
    # MPS-inspired initialization
    try:
        mps_init = _fit_mps_init(n, coupling, field, layers, H, ground_energy, psi0)
        initial_guesses.append(mps_init.flatten())
    except Exception:
        pass
    
    # Guess 1: zeros
    initial_guesses.append(np.zeros(total_params))
    
    # Guess 2: small random
    rng = np.random.RandomState(42)
    initial_guesses.append(rng.uniform(-np.pi/4, np.pi/4, total_params))
    
    # Guess 3: symmetric (same angle for all qubits in each layer)
    for l in range(layers):
        initial_guesses.append(np.array([0.1 * (l + 1)] * n).flatten())
    
    # Guess 4: Heuristic based on field strength
    for alpha in [0.3, 0.5, 0.7, 1.0]:
        guess = np.zeros(total_params)
        for l in range(layers):
            for i in range(n):
                guess[l * n + i] = alpha * field * (1.0 / (l + 1))
        initial_guesses.append(guess)
    
    # More diverse random inits
    for seed in [10, 20, 30, 40, 50]:
        rng_s = np.random.RandomState(seed)
        initial_guesses.append(rng_s.uniform(-np.pi, np.pi, total_params))
    
    # Try L-BFGS-B from each initial guess
    for init in initial_guesses:
        if time.time() > deadline - 30:
            break
        
        try:
            res = minimize(
                _objective_flat,
                init,
                args=(n, coupling, field, H, ground_energy),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_theta = res.x.reshape(layers, n)
        except Exception:
            pass
    
    # If we have time, do differential evolution for global search
    if time.time() < deadline - 45:
        bounds = [(-np.pi, np.pi)] * total_params
        
        try:
            de_result = differential_evolution(
                _objective_flat,
                bounds,
                args=(n, coupling, field, H, ground_energy),
                maxiter=100,
                popsize=12,
                tol=1e-9,
                seed=42,
                polish=False
            )
            if de_result.fun < best_obj:
                best_obj = de_result.fun
                best_theta = de_result.x.reshape(layers, n)
        except Exception:
            pass
    
    # Final high-precision polish with L-BFGS-B
    if best_theta is not None and time.time() < deadline - 10:
        try:
            res = minimize(
                _objective_flat,
                best_theta.flatten(),
                args=(n, coupling, field, H, ground_energy),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_theta = res.x.reshape(layers, n)
        except Exception:
            pass
    
    # Second polish pass if time allows
    if best_theta is not None and time.time() < deadline - 5:
        try:
            res = minimize(
                _objective_flat,
                best_theta.flatten(),
                args=(n, coupling, field, H, ground_energy),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_theta = res.x.reshape(layers, n)
        except Exception:
            pass
    
    # If no good solution found, return a reasonable default
    if best_theta is None:
        best_theta = np.zeros((layers, n))
    
    return best_theta
