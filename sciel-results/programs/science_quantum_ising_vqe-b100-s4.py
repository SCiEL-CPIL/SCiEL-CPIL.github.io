import numpy as np
import scipy.optimize as sco
import time

# Pre-computed gate matrices for CZ
_CZ = np.diag([1, 1, 1, -1]).astype(np.complex128)

def ry_gate(theta):
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)

def embed_2qubit_gate(state, gate, q1, q2, n):
    """Apply a 2-qubit gate to qubits q1, q2 in an n-qubit state.
    qubit i is bit i (little-endian: qubit 0 is LSB)."""
    dim = 2 ** n
    # Reshape to 2 x 2 x ... x 2
    # In little-endian, qubit i corresponds to axis i in the tensor
    # State tensor: state[q0, q1, ..., q_{n-1}]
    tensor = state.reshape([2] * n)
    
    # Permute axes to bring q1, q2 to front
    other_axes = [i for i in range(n) if i not in [q1, q2]]
    # New axis order: q1, q2, then others
    perm = [q1, q2] + other_axes
    tensor_perm = np.transpose(tensor, perm)
    
    # Reshape to (4, -1)
    shape_rest = tensor_perm.shape[2:]
    flat = tensor_perm.reshape(4, -1)
    
    # Apply gate
    flat_new = gate @ flat
    
    # Reshape back
    tensor_perm_new = flat_new.reshape([2, 2] + list(shape_rest))
    
    # Inverse permutation
    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i
    tensor_new = np.transpose(tensor_perm_new, inv_perm)
    
    return tensor_new.reshape(dim)

def embed_1qubit_gate(state, gate, q, n):
    """Apply a 1-qubit gate to qubit q in an n-qubit state."""
    dim = 2 ** n
    tensor = state.reshape([2] * n)
    
    # Reshape to (2, -1) along axis q
    # Move axis q to front
    other_axes = [i for i in range(n) if i != q]
    perm = [q] + other_axes
    tensor_perm = np.transpose(tensor, perm)
    
    flat = tensor_perm.reshape(2, -1)
    flat_new = gate @ flat
    tensor_perm_new = flat_new.reshape([2] + [2] * (n - 1))
    
    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i
    tensor_new = np.transpose(tensor_perm_new, inv_perm)
    
    return tensor_new.reshape(dim)

def compute_energy(state, H):
    return np.real(np.vdot(state, H @ state))

def build_hamiltonian(n, coupling, field):
    """Build H = -coupling * sum Z_i Z_{i+1} - field * sum X_i for open chain."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        basis = np.zeros(dim, dtype=complex)
        for state_idx in range(dim):
            bit_i = (state_idx >> i) & 1
            bit_ip1 = (state_idx >> (i + 1)) & 1
            # Z eigenvalue: +1 if bit=0, -1 if bit=1
            zi = 1 if bit_i == 0 else -1
            zip1 = 1 if bit_ip1 == 0 else -1
            basis[state_idx] = zi * zip1
        H += -coupling * np.diag(basis)
    
    # X_i terms
    for i in range(n):
        X = np.zeros((dim, dim), dtype=complex)
        for state_idx in range(dim):
            flip_idx = state_idx ^ (1 << i)
            X[flip_idx, state_idx] = 1
        H += -field * X
    
    return H

def exact_ground_energy(n, coupling, field):
    H = build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]

def circuit_state(theta, n, layers):
    """Build the circuit state given theta of shape (layers, n)."""
    dim = 2 ** n
    # Start in |0>^n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    for l in range(layers):
        # Apply Ry(theta[l, i]) to each qubit i
        for i in range(n):
            g = ry_gate(theta[l, i])
            state = embed_1qubit_gate(state, g, i, n)
        
        # Apply CZ on adjacent pairs (except after final layer)
        if l < layers - 1:
            for i in range(n - 1):
                state = embed_2qubit_gate(state, _CZ, i, i + 1, n)
    
    return state

def objective(params, n, coupling, field, layers, H, ground_energy):
    theta = params.reshape(layers, n)
    state = circuit_state(theta, n, layers)
    energy = compute_energy(state, H)
    # Mean energy gap per spin relative to ground state
    return (energy - ground_energy) / n

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Build Hamiltonian and get ground energy
    H = build_hamiltonian(n, coupling, field)
    ground_energy = exact_ground_energy(n, coupling, field)
    
    dim = 2 ** n
    n_params = layers * n
    
    # Initial guess: all zeros, then perturb
    best_params = np.zeros(n_params)
    best_val = np.inf
    
    # Try multiple initializations
    np.random.seed(42)
    initializations = [
        np.zeros(n_params),
        np.full(n_params, np.pi/4),
        np.random.uniform(-np.pi/2, np.pi/2, n_params),
        np.random.uniform(0, np.pi, n_params),
        np.random.uniform(-np.pi, np.pi, n_params),
    ]
    
    for init in initializations:
        if time.time() > deadline:
            break
        try:
            result = sco.minimize(
                objective, init,
                args=(n, coupling, field, layers, H, ground_energy),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Try L-BFGS-B with finite differences if time permits
    if time.time() < deadline - 20:
        try:
            result = sco.minimize(
                objective, best_params,
                args=(n, coupling, field, layers, H, ground_energy),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    return best_params.reshape(layers, n)
