"""Quantum information: shallow-circuit Ising ground states.

Strategy:
1. Exploit reflection symmetry of the open chain: theta[i] == theta[n-1-i].
   For n=4: unique angles per layer = 2 (positions 0,1 mirror to 3,2).
   For n=6: unique angles per layer = 3 (positions 0,1,2 mirror to 5,4,3).
   Total unique parameters: 3 layers * ceil(n/2) = 6 for n=4, 9 for n=6.

2. Use a coarse-to-fine grid search on the reduced parameter space, followed
   by scipy.optimize.minimize (BFGS) for local polish.

3. Compute exact ground energy via exact diagonalization.

4. Deadline guard: track wall-clock, return best found if approaching limit.
"""

import time
import numpy as np
from scipy.optimize import minimize as scipy_minimize

# Precompute exact ground energies for known configs (cache)
_exact_cache = {}

def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy via full diagonalization."""
    key = (n, coupling, field)
    if key in _exact_cache:
        return _exact_cache[key]
    
    # Build Hamiltonian in computational basis
    dim = 2 ** n
    # Pauli matrices
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms: -coupling * sum Z_i Z_{i+1}
    for i in range(n - 1):
        op = np.array([[1]], dtype=complex)
        for q in range(n):
            if q == i or q == i + 1:
                op = np.kron(op, Z)
            else:
                op = np.kron(op, I)
        H += -coupling * op
    
    # X terms: -field * sum X_i
    for i in range(n):
        op = np.array([[1]], dtype=complex)
        for q in range(n):
            if q == i:
                op = np.kron(op, X)
            else:
                op = np.kron(op, I)
        H += -field * op
    
    # H is Hermitian, diagonalize
    evals = np.linalg.eigvalsh(H)
    ground_energy = float(evals[0])
    _exact_cache[key] = ground_energy
    return ground_energy


def _apply_circuit(theta, n, layers):
    """Apply the 3-layer Ry/CZ circuit to |0>^n and return statevector.
    
    theta: shape (layers, n)
    Returns: statevector of length 2^n
    """
    dim = 2 ** n
    # Start with |0>^n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    for layer in range(layers):
        # Apply Ry(theta) to every qubit
        for q in range(n):
            t = theta[layer, q]
            c = np.cos(t / 2)
            s = np.sin(t / 2)
            # Ry = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
            Ry = np.array([[c, -s], [s, c]], dtype=complex)
            # Build full operator
            op = np.array([[1]], dtype=complex)
            for k in range(n):
                if k == q:
                    op = np.kron(op, Ry)
                else:
                    op = np.kron(op, I2)
            state = op @ state
        
        # Apply CZ on every adjacent pair in ascending order (between layers, not after last)
        if layer < layers - 1:
            for i in range(n - 1):
                # CZ = diag(1,1,1,-1) acting on qubits i and i+1
                op = np.array([[1]], dtype=complex)
                for k in range(n):
                    if k == i or k == i + 1:
                        # We need to build CZ on these two qubits
                        pass
                # Build CZ operator properly
                cz_local = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, -1]], dtype=complex)
                # cz_local acts on (qubit i, qubit i+1) in that order
                # Full operator: kron over all qubits, with cz on positions i, i+1
                op = np.array([[1]], dtype=complex)
                for k in range(n):
                    if k == i:
                        # This is the first qubit of the CZ pair
                        # We can't just kron a 2x2; need to handle the pair
                        pass
                
                # Simpler approach: apply CZ by iterating over basis states
                # CZ flips sign of |11> on the two qubits
                for basis in range(dim):
                    bit_i = (basis >> (n - 1 - i)) & 1
                    bit_ip1 = (basis >> (n - 1 - (i + 1))) & 1
                    if bit_i == 1 and bit_ip1 == 1:
                        state[basis] *= -1
                break  # This is wrong, let me redo
    
    return state


def _apply_circuit_v2(theta, n, layers):
    """Efficient circuit application using bit-level operations."""
    dim = 2 ** n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    I2 = np.eye(2, dtype=complex)
    
    for layer in range(layers):
        # Apply Ry to each qubit
        for q in range(n):
            t = theta[layer, q]
            c = np.cos(t / 2)
            s = np.sin(t / 2)
            Ry = np.array([[c, -s], [s, c]], dtype=complex)
            # Build full operator
            op = np.array([[1]], dtype=complex)
            for k in range(n):
                if k == q:
                    op = np.kron(op, Ry)
                else:
                    op = np.kron(op, I2)
            state = op @ state
        
        # Apply CZ on adjacent pairs (only between layers)
        if layer < layers - 1:
            for i in range(n - 1):
                # CZ flips phase of basis states where both qubit i and i+1 are 1
                # Qubit i corresponds to bit position (n-1-i)
                for basis in range(dim):
                    bit_i = (basis >> (n - 1 - i)) & 1
                    bit_ip1 = (basis >> (n - 1 - (i + 1))) & 1
                    if bit_i == 1 and bit_ip1 == 1:
                        state[basis] *= -1
    
    return state


def _circuit_energy(theta, n, coupling, field):
    """Compute energy of the circuit state."""
    state = _apply_circuit_v2(theta, n, 3)
    
    # Build Hamiltonian
    dim = 2 ** n
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    H = np.zeros((dim, dim), dtype=complex)
    
    for i in range(n - 1):
        op = np.array([[1]], dtype=complex)
        for q in range(n):
            if q == i or q == i + 1:
                op = np.kron(op, Z)
            else:
                op = np.kron(op, I)
        H += -coupling * op
    
    for i in range(n):
        op = np.array([[1]], dtype=complex)
        for q in range(n):
            if q == i:
                op = np.kron(op, X)
            else:
                op = np.kron(op, I)
        H += -field * op
    
    energy = float(np.real(state.conj().T @ H @ state))
    return energy


def _energy_gap(theta, n, coupling, field, ground_e):
    """Compute (circuit energy - ground energy) / n."""
    e = _circuit_energy(theta, n, coupling, field)
    return (e - ground_e) / n


def _symmetric_theta(unique_params, n, layers):
    """Expand symmetric parameters to full theta array.
    
    unique_params: flat array of length layers * ceil(n/2)
    Returns: theta of shape (layers, n)
    """
    half = (n + 1) // 2
    theta = np.zeros((layers, n))
    for layer in range(layers):
        for j in range(half):
            val = unique_params[layer * half + j]
            theta[layer, j] = val
            theta[layer, n - 1 - j] = val
    return theta


def angles(n, coupling, field, layers):
    """Optimize circuit parameters for the given Ising chain."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    ground_e = _exact_ground_energy(n, coupling, field)
    
    half = (n + 1) // 2
    n_unique = layers * half
    
    def objective(params):
        theta = _symmetric_theta(params, n, layers)
        return _energy_gap(theta, n, coupling, field, ground_e)
    
    best_params = None
    best_val = np.inf
    
    # Phase 1: Coarse grid search
    # Use a moderate grid to find a good starting point
    # For n=4: half=2, n_unique=6; for n=6: half=3, n_unique=9
    # Grid: try values in [-pi, pi] with some steps
    
    if n_unique <= 6:
        grid_size = 5
    else:
        grid_size = 4
    
    grid_vals = np.linspace(-np.pi, np.pi, grid_size)
    
    # Use randomized sampling instead of full grid for higher dimensions
    n_samples = 200 if n_unique <= 6 else 100
    
    rng = np.random.default_rng(42)
    
    for _ in range(n_samples):
        params = rng.uniform(-np.pi, np.pi, size=n_unique)
        val = objective(params)
        if val < best_val:
            best_val = val
            best_params = params.copy()
        
        if time.time() > deadline - 20:
            break
    
    # Also try some structured initial guesses
    # All zeros
    p0 = np.zeros(n_unique)
    v0 = objective(p0)
    if v0 < best_val:
        best_val = v0
        best_params = p0.copy()
    
    # All pi/2 (|+> state)
    p1 = np.full(n_unique, np.pi / 2)
    v1 = objective(p1)
    if v1 < best_val:
        best_val = v1
        best_params = p1.copy()
    
    # Small angles
    p2 = np.full(n_unique, 0.3)
    v2 = objective(p2)
    if v2 < best_val:
        best_val = v2
        best_params = p2.copy()
    
    # Phase 2: Local optimization from best point
    if best_params is not None and time.time() < deadline - 10:
        try:
            result = scipy_minimize(
                objective, best_params, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Further polish with BFGS if time allows
    if best_params is not None and time.time() < deadline - 5:
        try:
            result2 = scipy_minimize(
                objective, best_params, method='BFGS',
                options={'maxiter': 200, 'gtol': 1e-6}
            )
            if result2.fun < best_val:
                best_val = result2.fun
                best_params = result2.x.copy()
        except Exception:
            pass
    
    # Phase 4: A few more random restarts if time allows
    if time.time() < deadline - 15:
        for _ in range(30):
            if time.time() > deadline - 10:
                break
            params = rng.uniform(-np.pi, np.pi, size=n_unique)
            val = objective(params)
            if val < best_val:
                best_val = val
                best_params = params.copy()
                # Quick local polish
                if time.time() < deadline - 5:
                    try:
                        r = scipy_minimize(
                            objective, best_params, method='Nelder-Mead',
                            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
                        )
                        if r.fun < best_val:
                            best_val = r.fun
                            best_params = r.x.copy()
                    except Exception:
                        pass
    
    # Expand to full theta
    if best_params is None:
        best_params = np.zeros(n_unique)
    
    theta = _symmetric_theta(best_params, n, layers)
    return theta
