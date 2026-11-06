"""Quantum information: shallow-circuit Ising ground states."""
import numpy as np
from scipy.optimize import minimize
import time

# Precompute Pauli matrices
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def _matmul_kron(*mats):
    """Compute Kronecker product of matrices."""
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result


def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian H = -coupling*sum Z_i Z_{i+1} - field*sum X_i for open chain."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        ops = [I2] * n
        ops[i] = Z
        ops[i + 1] = Z
        H -= coupling * _matmul_kron(*ops)
    
    # X terms: -field * X_i
    for i in range(n):
        ops = [I2] * n
        ops[i] = X
        H -= field * _matmul_kron(*ops)
    
    return H


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy via diagonalization."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return float(eigvals[0])


def _apply_ry(state, theta, qubit):
    """Apply Ry(theta) to qubit in the given state vector."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    # Ry = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]
    gate = np.array([[c, -s], [s, c]], dtype=complex)
    
    n = int(np.log2(len(state)))
    dim = 2 ** n
    q = qubit
    
    # Reshape state to apply gate on qubit q
    # Qubit ordering: qubit i is bit i, so qubit 0 is least significant
    # We need to be careful about the ordering
    
    # For qubit q, the reshape is:
    # shape: (2^q, 2^(n-q-1), 2, 2^(q+1)) -> apply gate on axis 2
    # Actually, let's think about this differently.
    # State index: sum of bits. Qubit q is bit q.
    # If we reshape to (2^q, 2, 2^(n-q-1)), then axis 1 is qubit q.
    
    state_reshaped = state.reshape([2 ** q, 2, 2 ** (n - q - 1)])
    # Apply gate on axis 1
    state_reshaped = np.einsum('ab,...b->...a', gate, state_reshaped)
    return state_reshaped.reshape(dim)


def _apply_cz(state, q1, q2):
    """Apply CZ gate between qubits q1 and q2."""
    n = int(np.log2(len(state)))
    dim = 2 ** n
    
    # CZ = diag(1, 1, 1, -1) in the computational basis of the two qubits
    # For qubits q1 and q2, we need to apply -1 when both are 1.
    
    # Iterate over all states and flip sign when both qubits are 1
    # More efficient: use the fact that CZ only affects states where both bits are 1
    # State index has bit q1 and bit q2 set -> multiply by -1
    
    mask = np.zeros(dim, dtype=bool)
    for idx in range(dim):
        if (idx >> q1) & 1 and (idx >> q2) & 1:
            mask[idx] = True
    
    state[mask] *= -1
    return state


def _circuit_state(theta, n):
    """Build the circuit state from angles theta of shape (layers, n)."""
    layers = theta.shape[0]
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0  # |0>^n
    
    for layer in range(layers):
        # Apply Ry(theta[layer, i]) to qubit i
        for i in range(n):
            state = _apply_ry(state, theta[layer, i], i)
        
        # Apply CZ between adjacent pairs (if not the last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                state = _apply_cz(state, i, i + 1)
    
    return state


def _circuit_energy(theta, n, coupling, field, H):
    """Compute the energy of the circuit state."""
    state = _circuit_state(theta, n)
    # Energy = <psi|H|psi>
    energy = np.vdot(state, H @ state).real
    return float(energy)


def _objective(theta_flat, n, coupling, field, H, ground_energy):
    """Objective function: mean energy gap per spin."""
    theta = theta_flat.reshape(-1, n)
    energy = _circuit_energy(theta, n, coupling, field, H)
    gap = energy - ground_energy
    return gap / n


def angles(n, coupling, field, layers):
    """Optimize parameters of a fixed three-layer Ry/CZ circuit."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Precompute Hamiltonian and ground energy
    H = _build_hamiltonian(n, coupling, field)
    ground_energy = _exact_ground_energy(n, coupling, field)
    
    # Initial guess: try several starting points
    best_theta = None
    best_score = np.inf
    
    # Try multiple initializations
    inits = []
    
    # 1. All zeros (product |0> state)
    inits.append(np.zeros((layers, n)))
    
    # 2. All pi/2 (product |+> state)
    init_plus = np.zeros((layers, n))
    init_plus[-1] = np.pi / 2
    inits.append(init_plus)
    
    # 3. Small random perturbations around pi/2
    np.random.seed(42)
    for _ in range(10):
        init = np.zeros((layers, n))
        init[-1] = np.pi / 2 + np.random.uniform(-0.5, 0.5, n)
        init[:-1] = np.random.uniform(-0.5, 0.5, (layers - 1, n))
        inits.append(init)
    
    # 4. Try to match the expected structure: in transverse field Ising,
    #    the ground state has some correlation. Try symmetric inits.
    for val in [0.3, 0.5, 0.7, 1.0, 1.2]:
        init = np.full((layers, n), val)
        inits.append(init)
    
    # 5. Antisymmetric-like patterns
    for val in [0.4, 0.6, 0.8]:
        init = np.zeros((layers, n))
        for i in range(n):
            init[:, i] = val if i % 2 == 0 else -val
        inits.append(init)
    
    for init in inits:
        if time.time() > deadline:
            break
        
        theta_flat = init.flatten()
        
        try:
            result = minimize(
                _objective,
                theta_flat,
                args=(n, coupling, field, H, ground_energy),
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'maxls': 50
                }
            )
            
            if result.success or result.fun < best_score:
                score = result.fun
                if score < best_score:
                    best_score = score
                    best_theta = result.x.reshape(layers, n)
                    
        except Exception:
            continue
    
    # If no optimization succeeded, fall back to the initial seed behavior
    if best_theta is None:
        best_theta = np.zeros((layers, n))
        best_theta[-1] = np.pi / 2
    
    return best_theta
