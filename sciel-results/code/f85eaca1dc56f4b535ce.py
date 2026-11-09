"""Optimized shallow-circuit Ising ground state approximation."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _apply_ry(state, i, theta, n):
    """Apply Ry(theta) to qubit i of state (length 2^n)."""
    # Reshape to isolate qubit i
    # Qubit i is bit i.
    # Shape: (2^i, 2, 2^(n-i-1))
    shape = (2**i, 2, 2**(n-i-1))
    state_reshaped = state.reshape(shape)
    
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    
    # Ry = [[c, -s], [s, c]]
    # new_0 = c * old_0 - s * old_1
    # new_1 = s * old_0 + c * old_1
    
    old_0 = state_reshaped[:, 0, :]
    old_1 = state_reshaped[:, 1, :]
    
    new_0 = c * old_0 - s * old_1
    new_1 = s * old_0 + c * old_1
    
    state_reshaped[:, 0, :] = new_0
    state_reshaped[:, 1, :] = new_1
    
    return state_reshaped.reshape(2**n)

def _apply_cz(state, i, n):
    """Apply CZ to qubits i and i+1."""
    idx = np.arange(2**n)
    cond_i = (idx >> i) & 1
    cond_ip1 = (idx >> (i+1)) & 1
    
    mask = np.where((cond_i == 1) & (cond_ip1 == 1), -1, 1).astype(complex)
    
    return state * mask

def _build_state(theta, n, layers, coupling=None):
    """
    Build the quantum state from the circuit angles.
    """
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1.0
    
    for layer in range(layers):
        for i in range(n):
            state = _apply_ry(state, i, theta[layer, i], n)
        
        if layer < layers - 1:
            for i in range(n - 1):
                state = _apply_cz(state, i, n)
                
    return state

def _compute_energy(state, n, coupling, field):
    """
    Compute expected energy <psi|H|psi> where
    H = -coupling * sum_i Z_i Z_{i+1} - field * sum_i X_i
    """
    prob = np.abs(state)**2
    energy = 0.0
    idx = np.arange(2**n)
    
    # ZZ terms
    for i in range(n - 1):
        z_i = np.where((idx >> i) & 1 == 1, -1, 1)
        z_ip1 = np.where((idx >> (i+1)) & 1 == 1, -1, 1)
        zz_expect = np.sum(prob * z_i * z_ip1)
        energy -= coupling * zz_expect
        
    # X terms
    for i in range(n):
        shape = (2**i, 2, 2**(n-i-1))
        state_reshaped = state.reshape(shape)
        psi_0 = state_reshaped[:, 0, :]
        psi_1 = state_reshaped[:, 1, :]
        
        x_expect = 2 * np.real(np.sum(np.conj(psi_0) * psi_1))
        energy -= field * x_expect
        
    return np.real(energy)

def _exact_ground_energy(n, coupling, field):
    """
    Compute exact ground state energy by diagonalizing H.
    """
    dim = 2**n
    H = np.zeros((dim, dim), dtype=complex)
    idx = np.arange(dim)
    
    for i in range(n - 1):
        z_i = np.where((idx >> i) & 1 == 1, -1, 1)
        z_ip1 = np.where((idx >> (i+1)) & 1 == 1, -1, 1)
        H[np.arange(dim), np.arange(dim)] += -coupling * z_i * z_ip1
        
    for i in range(n):
        for k in range(dim):
            k_flip = k ^ (1 << i)
            H[k, k_flip] += -field
            
    eigenvalues = np.linalg.eigvalsh(H)
    return eigenvalues[0]

def angles(n, coupling, field, layers):
    """
    Return optimal circuit angles for the given parameters.
    """
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    theta0 = np.zeros((layers, n))
    
    def objective(theta_flat):
        theta = theta_flat.reshape(layers, n)
        state = _build_state(theta, n, layers)
        return _compute_energy(state, n, coupling, field)
        
    best_theta = theta0.copy()
    best_energy = objective(theta0.ravel())
    
    n_restarts = 5
    
    for r in range(n_restarts):
        if time.time() > deadline - 20:
            break
            
        if r == 0:
            theta_init = theta0.copy()
        else:
            theta_init = np.random.uniform(-np.pi, np.pi, (layers, n))
            
        try:
            result = minimize(
                objective,
                theta_init.ravel(),
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * (layers * n),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_energy:
                best_energy = result.fun
                best_theta = result.x.reshape(layers, n)
        except Exception:
            pass
            
    if time.time() < deadline - 10:
        try:
            result_nm = minimize(
                objective,
                best_theta.ravel(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-12}
            )
            if result_nm.fun < best_energy:
                best_energy = result_nm.fun
                best_theta = result_nm.x.reshape(layers, n)
        except Exception:
            pass
            
    return best_theta
# EVOLVE-BLOCK-END
