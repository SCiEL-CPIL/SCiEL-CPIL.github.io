import numpy as np
from scipy.optimize import minimize
import time


def _build_circuit_state(theta, n, layers):
    """Build the state vector from circuit parameters using tensor operations."""
    dim = 2 ** n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0  # |0>^n
    shape = [2] * n
    
    for layer in range(layers):
        # Apply Ry to each qubit
        for q in range(n):
            t = theta[layer, q]
            if abs(t) < 1e-15:
                continue
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            gate = np.array([[c, -s], [s, c]], dtype=np.complex128)
            state_r = state.reshape(shape)
            state_moved = np.moveaxis(state_r, q, 0)
            state_out = np.tensordot(gate, state_moved, axes=([1], [0]))
            state = state_out.reshape(dim)
        
        # Apply CZ between this layer and next (if not last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                # CZ: multiply by -1 when both qubits q and q+1 are 1
                state_r = state.reshape(shape)
                # Create phase array: -1 where both bits are 1
                phase = np.ones(shape, dtype=np.complex128)
                # Set phase to -1 where qubit q=1 AND qubit q+1=1
                idx_q = [0] * n
                idx_q[q] = 1
                idx_qp1 = [0] * n
                idx_qp1[q] = 1
                idx_qp1[q + 1] = 1
                phase[tuple(idx_q)] = -1.0
                # But we need it for ALL combinations where both are 1
                # Better: iterate over the two qubit axes
                state_moved = np.moveaxis(state_r, [q, q + 1], [0, 1])
                # state_moved has shape (2, 2, 2, ..., 2) with first two axes being q and q+1
                # CZ applies -1 to element [1,1,...]
                state_moved[1, 1] *= -1.0
                state = np.moveaxis(state_moved, [0, 1], [q, q + 1]).reshape(dim)
    
    return state


def _precompute_hamiltonian(n, coupling, field):
    """Precompute the Hamiltonian matrix and its ground energy."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # Z_i Z_(i+1) terms (diagonal)
    for i in range(n - 1):
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            zi = 1.0 if bit_i == 0 else -1.0
            zip1 = 1.0 if bit_ip1 == 0 else -1.0
            H[idx, idx] += -coupling * zi * zip1
    
    # X_i terms (off-diagonal)
    for i in range(n):
        for idx in range(dim):
            flipped = idx ^ (1 << i)
            H[idx, flipped] += -field
    
    eigvals = np.linalg.eigvalsh(H)
    ground_energy = np.min(eigvals)
    return H, ground_energy


def _compute_energy(state, n, coupling, field, H=None):
    """Compute expectation value <psi|H|psi>."""
    if H is not None:
        return np.real(np.conj(state) @ H @ state)
    
    dim = 2 ** n
    energy = 0.0
    
    # Z_i Z_(i+1) terms
    for i in range(n - 1):
        zz = 0.0
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            zi = 1.0 if bit_i == 0 else -1.0
            zip1 = 1.0 if bit_ip1 == 0 else -1.0
            zz += (np.abs(state[idx]) ** 2) * zi * zip1
        energy += -coupling * zz
    
    # X_i terms
    for i in range(n):
        x_val = 0.0 + 0.0j
        for idx in range(dim):
            flipped = idx ^ (1 << i)
            x_val += np.conj(state[idx]) * state[flipped]
        energy += -field * np.real(x_val)
    
    return np.real(energy)


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Precompute Hamiltonian and ground energy
    H, ground_energy = _precompute_hamiltonian(n, coupling, field)
    
    def objective(params):
        theta = params.reshape((layers, n))
        state = _build_circuit_state(theta, n, layers)
        energy = _compute_energy(state, n, coupling, field, H)
        return (energy - ground_energy) / n
    
    best_result = None
    best_val = np.inf
    
    # Generate multiple starting points
    starts = []
    # All zeros
    starts.append(np.zeros(layers * n))
    # pi/2 in last layer
    x0 = np.zeros(layers * n)
    x0[(layers - 1) * n:layers * n] = np.pi / 2
    starts.append(x0)
    # Small angles
    starts.append(np.random.uniform(-0.5, 0.5, size=layers * n))
    # Several random starts
    np.random.seed(42)
    for _ in range(4):
        starts.append(np.random.uniform(-np.pi, np.pi, size=layers * n))
    
    for x0 in starts:
        if time.time() > deadline - 15.0:
            break
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                          bounds=[(-np.pi, np.pi)] * (layers * n),
                          options={'maxiter': 3000, 'maxfun': 10000,
                                   'ftol': 1e-14, 'gtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    # Final polish pass with tighter tolerance if time permits
    if best_result is not None and time.time() < deadline - 5.0:
        try:
            res = minimize(objective, best_result, method='L-BFGS-B',
                          bounds=[(-np.pi, np.pi)] * (layers * n),
                          options={'maxiter': 5000, 'maxfun': 20000,
                                   'ftol': 1e-16, 'gtol': 1e-16})
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    if best_result is None:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    return best_result.reshape((layers, n))
