"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _build_hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian H = -J sum Z_i Z_{i+1} - h sum X_i."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=float)
    
    I = np.eye(2, dtype=float)
    X = np.array([[0, 1], [1, 0]], dtype=float)
    Z = np.array([[1, 0], [0, -1]], dtype=float)
    
    def kron_list(ops):
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    for i in range(n - 1):
        ops = [I] * n
        ops[i] = Z
        ops[i+1] = Z
        H -= coupling * kron_list(ops)
        
    for i in range(n):
        ops = [I] * n
        ops[i] = X
        H -= field * kron_list(ops)
        
    return H


def _apply_cz_vectorized(state, n):
    """Apply CZ gates on adjacent pairs in ascending order. Modifies state in place."""
    dim = 1 << n
    indices = np.arange(dim)
    for i in range(n - 1):
        mask_i = 1 << i
        mask_ip1 = 1 << (i + 1)
        both_set = (indices & mask_i) != 0
        both_set &= (indices & mask_ip1) != 0
        state[both_set] *= -1.0
    return state


def _circuit_state_vectorized(theta, n, layers):
    """Compute the statevector of the circuit using vectorized bit-index arithmetic."""
    dim = 1 << n
    state = np.zeros(dim, dtype=float)
    state[0] = 1.0
    
    for l in range(layers):
        for i in range(n):
            t = theta[l, i]
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            mask = 1 << i
            idx0 = np.arange(dim)
            idx0 = idx0[(idx0 & mask) == 0]
            idx1 = idx0 | mask
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            state[idx0] = c * old0 - s * old1
            state[idx1] = s * old0 + c * old1
        
        if l < layers - 1:
            _apply_cz_vectorized(state, n)
            
    return state


def _energy_from_state(state, H):
    """Compute <H> for a real state vector using matrix multiplication."""
    return float(state @ (H @ state))


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy by diagonalizing H."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return np.real(eigvals[0])


def _energy_vectorized(thetas, n, H, layers):
    """Compute circuit energy using vectorized state evolution and matrix multiplication."""
    state = _circuit_state_vectorized(thetas, n, layers)
    return _energy_from_state(state, H)


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin for 150s limit

    # Build Hamiltonian once
    H = _build_hamiltonian(n, coupling, field)
    
    # Precompute exact ground energy
    e0 = _exact_ground_energy(n, coupling, field)

    def obj(params):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e10
        thetas = params.reshape(layers, n)
        e = _energy_vectorized(thetas, n, H, layers)
        return e

    # Generate diverse initializations
    n_params = layers * n
    inits = []
    
    # Mean-field like initialization: pi/2 for all
    inits.append(np.full(n_params, np.pi / 2.0))
    
    # All zeros
    inits.append(np.zeros(n_params))
    
    # Random inits with fixed seed for reproducibility
    np.random.seed(42)
    for _ in range(10):
        inits.append(np.random.uniform(-np.pi, np.pi, size=n_params))
    
    # Symmetric inits: same angle for all qubits in a layer
    for _ in range(5):
        layer_angles = np.random.uniform(-np.pi, np.pi, size=layers)
        inits.append(np.repeat(layer_angles, n))
    
    # Alternating inits
    for _ in range(3):
        a = np.random.uniform(-np.pi, np.pi)
        b = np.random.uniform(-np.pi, np.pi)
        for l in range(layers):
            for i in range(n):
                inits[-1][l * n + i] = a if i % 2 == 0 else b
    
    # Sort by objective value
    init_objs = []
    for x in inits:
        init_objs.append((obj(x), x))
    init_objs.sort(key=lambda t: t[0])
    
    best_params = init_objs[0][1]
    best_val = init_objs[0][0]
    
    # Optimize from best initializations
    n_restarts = 8
    for r in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 20.0:
            break
        
        if r < len(init_objs):
            x_start = init_objs[r][1]
        else:
            x_start = init_objs[0][1] + np.random.randn(n_params) * 0.3
        
        try:
            res = minimize(
                obj,
                x_start,
                method='BFGS',
                options={'maxiter': 400, 'gtol': 1e-9}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 10.0:
        try:
            res = minimize(
                obj,
                best_params,
                method='BFGS',
                options={'maxiter': 500, 'gtol': 1e-11}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Second polish if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0:
        try:
            res = minimize(
                obj,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    return best_params.reshape(layers, n)


# EVOLVE-BLOCK-END
