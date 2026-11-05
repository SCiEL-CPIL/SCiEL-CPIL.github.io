import numpy as np
import time
from scipy.optimize import minimize

# Pauli matrices
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
I2 = np.eye(2, dtype=np.complex128)

# CZ gate
CZ = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, -1]], dtype=np.complex128)

_H_cache = {}

def _build_hamiltonian(n, coupling, field):
    H = np.zeros((2**n, 2**n), dtype=np.complex128)
    def kron_all(ops):
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res
    for i in range(n - 1):
        ops = [I2] * n
        ops[i] = Z
        ops[i + 1] = Z
        H -= coupling * kron_all(ops)
    for i in range(n):
        ops = [I2] * n
        ops[i] = X
        H -= field * kron_all(ops)
    return H

def _get_hamiltonian(n, coupling, field):
    key = (n, coupling, field)
    if key not in _H_cache:
        _H_cache[key] = _build_hamiltonian(n, coupling, field)
    return _H_cache[key]

def _apply_single_qubit_gate_batch(states, gate, qubit, n):
    """
    Apply a 2x2 gate to a specific qubit in a batch of state vectors.
    states: (2^n, B)
    """
    B = states.shape[1]
    states_tensor = states.reshape([2] * n + [B])
    
    perm_list = [i for i in range(n) if i != qubit]
    perm_list.insert(n - 1, qubit)
    perm_list.append(n)
    
    states_perm = np.transpose(states_tensor, perm_list)
    
    new_shape = [-1, 2, B]
    states_3d = states_perm.reshape(new_shape)
    
    new_states_3d = np.einsum('ab,ijb->ija', gate, states_3d)
    
    new_states_perm = new_states_3d.reshape(states_perm.shape)
    
    inv_perm = [0] * (n + 1)
    for i, p in enumerate(perm_list):
        inv_perm[p] = i
        
    new_states_tensor = np.transpose(new_states_perm, inv_perm)
    
    return new_states_tensor.reshape(2**n, B)

def _apply_cz_gate_batch(states, i, j, n):
    """
    Apply CZ gate between qubits i and j (i < j) in a batch of state vectors.
    states: (2^n, B)
    """
    B = states.shape[1]
    states_tensor = states.reshape([2] * n + [B])
    
    perm_list = list(range(n))
    perm_list[-1], perm_list[j] = perm_list[j], perm_list[-1]
    perm_list[-2], perm_list[i] = perm_list[i], perm_list[-2]
    perm_list.append(n)
    
    states_perm = np.transpose(states_tensor, perm_list)
    
    # Reshape last two qubit dims (now at n-2, n-1) and B
    # We want to apply CZ to the 4-dim space formed by qubits i and j.
    # In permuted tensor, i is at n-2, j is at n-1.
    # So we reshape to (-1, 4, B)
    new_shape = [-1, 4, B]
    states_3d = states_perm.reshape(new_shape)
    
    new_states_3d = np.einsum('ab,ijb->ija', CZ, states_3d)
    
    new_states_perm = new_states_3d.reshape(states_perm.shape)
    
    inv_perm = [0] * (n + 1)
    for idx, p in enumerate(perm_list):
        inv_perm[p] = idx
        
    new_states_tensor = np.transpose(new_states_perm, inv_perm)
    
    return new_states_tensor.reshape(2**n, B)

def _forward_pass_and_grads(theta, n, layers):
    psi = np.zeros(2**n, dtype=np.complex128)
    psi[0] = 1.0
    
    num_params = layers * n
    # M: (2^n, 1 + num_params)
    # Column 0: psi
    # Column k+1: d_psi_k
    M = np.zeros((2**n, 1 + num_params), dtype=np.complex128)
    M[:, 0] = psi
    
    param_idx = 0
    for layer in range(layers):
        for q in range(n):
            theta_val = theta[layer, q]
            c = np.cos(theta_val / 2)
            s = np.sin(theta_val / 2)
            
            ry = np.array([[c, -s], [s, c]], dtype=np.complex128)
            ry_prime = np.array([[-0.5 * s, -0.5 * c],
                                 [0.5 * c, -0.5 * s]], dtype=np.complex128)
            
            # Apply Ry to all columns
            M = _apply_single_qubit_gate_batch(M, ry, q, n)
            
            # Update derivative column for current param
            # d_new_k = ry_prime * psi_old + ry * d_old_k
            # M[:, k+1] currently holds ry * d_old_k
            # We need to add ry_prime * psi_old
            
            # psi_old is the original psi before this gate
            # We didn't keep it separately, but we can compute ry_prime * psi_old
            # Wait, M[:, 0] is now the NEW psi.
            # We need psi_old.
            # Let's store psi_old before the update.
            
            # Correction: Store psi_old
            # This requires a copy or re-calculation.
            # Better: Keep psi_old separately.
            
            # Let's refactor to keep psi_old.
            
            pass # Placeholder, logic needs fix
        if layer < layers - 1:
            for i in range(n - 1):
                M = _apply_cz_gate_batch(M, i, i + 1, n)
    
    return M[:, 0], M[:, 1:]

# Fix _forward_pass_and_grads
def _forward_pass_and_grads(theta, n, layers):
    psi_old = np.zeros(2**n, dtype=np.complex128)
    psi_old[0] = 1.0
    
    num_params = layers * n
    M = np.zeros((2**n, 1 + num_params), dtype=np.complex128)
    M[:, 0] = psi_old
    
    param_idx = 0
    for layer in range(layers):
        for q in range(n):
            theta_val = theta[layer, q]
            c = np.cos(theta_val / 2)
            s = np.sin(theta_val / 2)
            
            ry = np.array([[c, -s], [s, c]], dtype=np.complex128)
            ry_prime = np.array([[-0.5 * s, -0.5 * c],
                                 [0.5 * c, -0.5 * s]], dtype=np.complex128)
            
            # Apply Ry to all columns
            M = _apply_single_qubit_gate_batch(M, ry, q, n)
            
            # Update derivative column for current param
            # d_new_k = ry_prime * psi_old + ry * d_old_k
            # M[:, k+1] currently holds ry * d_old_k (because the batched apply applied ry to all cols)
            # We need to add ry_prime * psi_old
            
            term1 = _apply_single_qubit_gate_batch(psi_old.reshape(-1, 1), ry_prime, q, n).flatten()
            M[:, param_idx + 1] += term1
            
            # Update psi_old to new psi
            psi_old = M[:, 0].copy()
            
            param_idx += 1
        
        if layer < layers - 1:
            for i in range(n - 1):
                M = _apply_cz_gate_batch(M, i, i + 1, n)
                # CZ doesn't change the derivative logic, just applies to all cols
                # psi_old is updated implicitly by M[:, 0]
                psi_old = M[:, 0].copy()
    
    return M[:, 0], M[:, 1:]

def _energy_and_grad(theta_flat, n, coupling, field, layers, H):
    theta = theta_flat.reshape(layers, n)
    psi, d_psi = _forward_pass_and_grads(theta, n, layers)
    
    H_psi = H @ psi
    E = np.real(np.vdot(psi, H_psi))
    
    grad = np.zeros_like(theta_flat)
    # d_psi is (2^n, num_params)
    # We need <d_psi_k | H | psi> for each k
    # This is np.vdot(d_psi[:, k], H_psi)
    # Vectorized: np.einsum('ij,j->i', d_psi.conj(), H_psi)
    grads = np.einsum('ij,j->i', d_psi.conj(), H_psi)
    grad = 2.0 * np.real(grads)
    
    return E, grad / n

def _heuristic_init(n, field):
    theta = np.zeros((3, n))
    theta[2] = np.pi / 2
    return theta

def _optimize_one(n, coupling, field, layers, H, deadline):
    bounds = [(-np.pi, np.pi)] * (layers * n)
    best_theta = None
    best_val = np.inf
    
    theta_init = _heuristic_init(n, field).flatten()
    theta_plus = np.zeros(layers * n)
    theta_plus[-n:] = np.pi / 2
    theta_zero = np.zeros(layers * n)
    
    initial_points = [theta_init, theta_plus, theta_zero]
    
    rng = np.random.RandomState(42)
    for _ in range(5):
        initial_points.append(rng.uniform(-np.pi, np.pi, size=(layers, n)).flatten())
    
    for theta_start in initial_points:
        if time.time() > deadline - 5:
            break
        try:
            def obj(x):
                return _energy_and_grad(x, n, coupling, field, layers, H)[0]
            def jac(x):
                return _energy_and_grad(x, n, coupling, field, layers, H)[1]
            
            res = minimize(
                obj,
                theta_start,
                jac=jac,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'maxfun': 200, 'gtol': 1e-8}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_theta = res.x.copy()
        except Exception:
            pass
    
    if best_theta is None:
        best_theta = theta_init
    return best_theta.reshape(layers, n)

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145
    H = _get_hamiltonian(n, coupling, field)
    
    if time.time() > deadline - 10:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    theta = _optimize_one(n, coupling, field, layers, H, deadline)
    theta = np.where(np.isfinite(theta), theta, 0.0)
    return theta
