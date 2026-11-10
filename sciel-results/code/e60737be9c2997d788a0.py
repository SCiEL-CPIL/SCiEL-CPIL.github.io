"""Quantum circuit synthesis with CNOT budget constraints."""
# EVOLVE-BLOCK-START
import math
import time
import random
import numpy as np

_DEADLINE = None
_TIME_MARGIN = 20.0  # seconds of safety margin before 270s kill


def _time_remaining():
    """Return seconds remaining before the deadline."""
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 270.0 - _TIME_MARGIN
    return max(0.0, _DEADLINE - time.time())


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            index = np.arange(2 ** qubits)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _cnot_matrix(qubits, control, target):
    mat = np.eye(2 ** qubits, dtype=complex)
    index = np.arange(2 ** qubits)
    bit = (index >> (qubits - 1 - control)) & 1
    mat = mat[index ^ (bit << (qubits - 1 - target)), :]
    return mat


def _apply_u3_to_state(state, qubits, qubit, theta, phi, lam):
    """Apply u3 to a specific qubit of a state vector (faster than full matrix mult)."""
    n = 2 ** qubits
    single = _u3_matrix(theta, phi, lam)
    # Reshape state to tensor
    state_tensor = state.reshape([2] * qubits)
    # Apply single-qubit gate to the specified qubit
    # Move the target qubit to the last position, apply, move back
    perm = list(range(qubits))
    perm[qubit], perm[-1] = perm[-1], perm[qubit]
    state_tensor = np.transpose(state_tensor, perm)
    # Apply gate to last qubit
    state_tensor = np.tensordot(single, state_tensor, axes=([1], [qubits - 1]))
    # Move back
    inv_perm = [0] * qubits
    for i, p in enumerate(perm):
        inv_perm[p] = i
    state_tensor = np.transpose(state_tensor, inv_perm)
    return state_tensor.reshape(n)


def _apply_cnot_to_state(state, qubits, control, target):
    """Apply CNOT to a state vector."""
    n = 2 ** qubits
    index = np.arange(n)
    bit = (index >> (qubits - 1 - control)) & 1
    return state[index ^ (bit << (qubits - 1 - target))]


def _circuit_to_state(gates, qubits):
    """Compute U|0...0> by applying gates to the initial state (much faster)."""
    state = np.zeros(2 ** qubits, dtype=complex)
    state[0] = 1.0
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            state = _apply_u3_to_state(state, qubits, q, theta, phi, lam)
        else:
            _, control, target = gate
            state = _apply_cnot_to_state(state, qubits, control, target)
    return state


def _circuit_to_unitary_from_columns(gates, qubits):
    """Build the full unitary by computing U|e_j> for each basis state."""
    n = 2 ** qubits
    cols = []
    for j in range(n):
        state = np.zeros(n, dtype=complex)
        state[j] = 1.0
        for gate in gates:
            if gate[0] == "u3":
                _, q, theta, phi, lam = gate
                state = _apply_u3_to_state(state, qubits, q, theta, phi, lam)
            else:
                _, control, target = gate
                state = _apply_cnot_to_state(state, qubits, control, target)
        cols.append(state)
    return np.column_stack(cols)


def _fid_from_states(target, gates, qubits):
    """Compute fidelity |<target|U|0..0>|^2 using state evolution only.
    Actually we need |tr(target^dagger U)|/n, which requires full U.
    But for the first column: (target^dagger U)|0> = target^dagger (U|0>)
    The trace is sum_j <j|target^dagger U|j> = sum_j <U|j>^* target^dagger |j> ... 
    Actually tr(target^dagger U) = sum_j <j| target^dagger U |j> = sum_j (target^dagger U)_{jj}
    = sum_j <j| target^dagger U |j> = sum_j conj(target_{j,j}) ... no.
    (target^dagger U)_{jj} = sum_k conj(target_{k,j}) U_{k,j}
    So tr = sum_j sum_k conj(target_{k,j}) U_{k,j}
    This requires all columns of U.
    
    Alternative: tr(target^dagger U) = <target, U> where <A,B> = tr(A^dagger B)
    = sum_{i,j} conj(target_{i,j}) U_{i,j}
    
    Let me just build the full unitary but use the faster state-based method.
    """
    # For efficiency, let's use the column-by-column approach
    target_mat = target
    n = 2 ** qubits
    trace_val = 0.0
    for j in range(n):
        state = np.zeros(n, dtype=complex)
        state[j] = 1.0
        for gate in gates:
            if gate[0] == "u3":
                _, q, theta, phi, lam = gate
                state = _apply_u3_to_state(state, qubits, q, theta, phi, lam)
            else:
                _, control, target_q = gate
                state = _apply_cnot_to_state(state, qubits, control, target_q)
        # Column j of U is 'state'
        # Contribution to trace: sum_i conj(target[i,j]) * state[i]
        trace_val += np.vdot(target_mat[:, j], state)
    return 1.0 - min(1.0, abs(trace_val) / n)


def synthesize(problem):
    """Synthesize a quantum circuit within CNOT budget using state-vector optimization."""
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 270.0 - _TIME_MARGIN
    
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    n = 2 ** qubits
    
    # Determine CNOT pairs to use
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    # Build layout: layers of u3 followed by CNOTs
    # For 2 qubits: budget CNOTs
    # For 3 qubits: budget CNOTs
    layout_cx = []
    for i in range(budget):
        layout_cx.append(list(pairs[i % len(pairs)]))
    
    # Layout: u3 layer, cx, u3 layer, cx, ..., u3 layer
    # Number of u3 layers = budget + 1
    n_u3_layers = budget + 1
    n_params = n_u3_layers * qubits * 3  # theta, phi, lam per qubit per layer
    
    def make_circuit(params):
        gates = []
        idx = 0
        for layer in range(n_u3_layers):
            for q in range(qubits):
                theta = params[idx]
                phi = params[idx + 1]
                lam = params[idx + 2]
                gates.append(["u3", q, theta, phi, lam])
                idx += 3
            if layer < budget:
                gates.append(["cx", layout_cx[layer][0], layout_cx[layer][1]])
        return gates
    
    # Initial parameters: try to match target using a simple heuristic
    # Start with zeros, then optimize
    rng = random.Random(12345 + qubits * 100 + budget)
    
    best_params = [0.0] * n_params
    best_dist = _fid_from_states(target, make_circuit(best_params), qubits)
    
    # Coordinate descent with adaptive step sizes
    step_sizes = [1.0] * n_params
    max_sweeps = 50
    
    for sweep in range(max_sweeps):
        if _time_remaining() < 15.0:
            break
        
        step = 1.0 / (1 + sweep * 0.5)
        improved = False
        
        for i in range(n_params):
            if _time_remaining() < 15.0:
                break
            
            current = best_params[i]
            # Try several perturbations
            candidates = [
                current + step,
                current - step,
                current + rng.uniform(-step * 2, step * 2),
                current + rng.uniform(-math.pi, math.pi) * 0.1,
            ]
            
            for cand in candidates:
                new_params = best_params[:]
                new_params[i] = cand
                dist = _fid_from_states(target, make_circuit(new_params), qubits)
                if dist < best_dist:
                    best_params = new_params
                    best_dist = dist
                    improved = True
                    step_sizes[i] = step
                    break
        
        if not improved:
            # Reduce step size globally
            step_sizes = [s * 0.5 for s in step_sizes]
            if step < 1e-6:
                break
    
    # Final polish with smaller steps
    for sweep in range(20):
        if _time_remaining() < 15.0:
            break
        
        step = 0.1 / (1 + sweep * 0.3)
        for i in range(n_params):
            if _time_remaining() < 15.0:
                break
            current = best_params[i]
            for delta in (step, -step, rng.uniform(-step, step)):
                new_params = best_params[:]
                new_params[i] = current + delta
                dist = _fid_from_states(target, make_circuit(new_params), qubits)
                if dist < best_dist:
                    best_params = new_params
                    best_dist = dist
                    break
    
    return make_circuit(best_params)
# EVOLVE-BLOCK-END
