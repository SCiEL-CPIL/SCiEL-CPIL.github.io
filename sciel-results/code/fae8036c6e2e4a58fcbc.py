import numpy as np
from scipy.optimize import basinhopping, minimize
import time

# Cache for exact ground energies and Hamiltonians
_hamiltonian_cache = {}
_ground_energy_cache = {}

def _get_hamiltonian(n, coupling, field):
    key = (n, coupling, field)
    if key in _hamiltonian_cache:
        return _hamiltonian_cache[key]
    # Build H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    # Z = diag(1, -1), X = [[0,1],[1,0]]
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        # Z on qubit i and qubit i+1
        # Qubit i is bit i (LSB convention: qubit i corresponds to bit i)
        # We build operator by kron
        op = np.array([[1]], dtype=complex)
        for q in range(n - 1, -1, -1):
            if q == i or q == i + 1:
                op = np.kron(op, Z)
            else:
                op = np.kron(op, np.eye(2, dtype=complex))
        H -= coupling * op
    
    # X_i terms
    for i in range(n):
        op = np.array([[1]], dtype=complex)
        for q in range(n - 1, -1, -1):
            if q == i:
                op = np.kron(op, X)
            else:
                op = np.kron(op, np.eye(2, dtype=complex))
        H -= field * op
    
    _hamiltonian_cache[key] = H
    return H


def _get_ground_energy(n, coupling, field):
    key = (n, coupling, field)
    if key in _ground_energy_cache:
        return _ground_energy_cache[key]
    H = _get_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    e0 = float(eigvals[0])
    _ground_energy_cache[key] = e0
    return e0


def _simulate_circuit(theta, n):
    """Simulate the 3-layer Ry/CZ circuit. theta shape (layers, n).
    Returns statevector of length 2^n."""
    # Start |0>^n
    dim = 1 << n
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    
    layers = theta.shape[0]
    
    for layer in range(layers):
        # Apply Ry(theta[layer, i]) to qubit i for all i
        # Ry(theta) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
        # We need to apply to each qubit. For a statevector, applying a single-qubit gate
        # to qubit q means: for each basis state, we modify the amplitudes.
        # Efficient approach: reshape and use einsum or manual index manipulation.
        
        # Apply Ry to all qubits simultaneously (they commute since they act on different qubits)
        for i in range(n):
            t = theta[layer, i]
            c = np.cos(t / 2)
            s = np.sin(t / 2)
            # Ry = [[c, -s], [s, c]]
            # Qubit i is bit i. We need to apply this gate.
            # Reshape psi into tensor of shape (2,)*n where axis 0 is qubit n-1 (MSB), axis n-1 is qubit 0 (LSB)
            # Actually, let's use a direct approach with indexing.
            
            # For qubit i (bit i), we split the state into blocks of size 2^i,
            # and within each block of size 2^(i+1), the first 2^i entries have bit i = 0,
            # the next 2^i have bit i = 1.
            block = 1 << i
            for base in range(0, dim, 2 * block):
                # Indices with bit i = 0: base + [0, block)
                # Indices with bit i = 1: base + [block, 2*block)
                idx0 = slice(base, base + block)
                idx1 = slice(base + block, base + 2 * block)
                psi0 = psi[idx0].copy()
                psi1 = psi[idx1].copy()
                psi[idx0] = c * psi0 - s * psi1
                psi[idx1] = s * psi0 + c * psi1
        
        # Apply CZ gates on adjacent pairs (i, i+1) for i = 0..n-2, in ascending order
        if layer < layers - 1:
            for i in range(n - 1):
                # CZ: |11> -> -|11>, others unchanged
                # Qubits i and i+1 are both 1
                block_low = 1 << i
                block_high = 1 << (i + 1)
                # For each combination of other bits, find where both bits i and i+1 are 1
                # Bits i and i+1: the value contributed is (1<<i) + (1<<(i+1))
                # We iterate over all other bit patterns
                for base in range(0, dim, 1 << (i + 2)):
                    # Within this super-block of size 2^(i+2),
                    # the offset for bits i and i+1 both being 1 is:
                    offset = (1 << i) | (1 << (i + 1))
                    # But we need to be careful: the super-block structure
                    # Actually, let's just find all indices where both bit i and bit i+1 are 1
                    pass
                
                # Simpler: iterate over all basis states and check
                # For n<=6, dim<=64, this is fine
                for idx in range(dim):
                    bit_i = (idx >> i) & 1
                    bit_ip1 = (idx >> (i + 1)) & 1
                    if bit_i == 1 and bit_ip1 == 1:
                        psi[idx] *= -1
    
    return psi


def _energy(psi, H, n):
    """Compute <psi|H|psi>."""
    return float(np.real(np.vdot(psi, H @ psi)))


def _objective(params, n, coupling, field, H, e0):
    """Objective: mean energy gap per spin."""
    theta = params.reshape(3, n)
    psi = _simulate_circuit(theta, n)
    e = _energy(psi, H, n)
    return (e - e0) / n


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s kill
    
    H = _get_hamiltonian(n, coupling, field)
    e0 = _get_ground_energy(n, coupling, field)
    
    # Initial guess: all zeros (product |0> state) or small random
    best_params = np.zeros(3 * n)
    best_obj = _objective(best_params, n, coupling, field, H, e0)
    
    # Try several random initializations with basin hopping
    rng = np.random.default_rng(42)
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        if restart == 0:
            x0 = np.zeros(3 * n)
        else:
            x0 = rng.uniform(-np.pi, np.pi, size=3 * n)
        
        try:
            result = basinhopping(
                _objective,
                x0,
                args=(n, coupling, field, H, e0),
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 50}},
                niter=3,
                T=0.5,
                stepsize=0.5,
                seed=restart,
                callback=None
            )
            
            obj_val = result.fun
            if obj_val < best_obj:
                best_obj = obj_val
                best_params = result.x.copy()
        except Exception:
            continue
    
    # Final polish with tighter tolerance if time permits
    if time.time() < deadline - 5:
        try:
            result = minimize(
                _objective,
                best_params,
                args=(n, coupling, field, H, e0),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    theta = best_params.reshape(3, n)
    # Ensure finite
    theta = np.where(np.isfinite(theta), theta, 0.0)
    return theta
