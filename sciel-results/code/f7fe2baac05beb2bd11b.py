"""Rewritten program for exact quantum circuit synthesis within CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

# Constants
TIME_LIMIT = 250  # Safety margin for 270s limit
START_TIME = time.time()


def _time_remaining():
    return TIME_LIMIT - (time.time() - START_TIME)


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
    return np.array([[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _apply_single(matrix, qubit, single, qubits):
    """Apply a single-qubit gate to a specific qubit in the full matrix."""
    eye = np.eye(2, dtype=complex)
    operator = np.array([[1]], dtype=complex)
    for k in range(qubits):
        operator = np.kron(operator, single if k == qubit else eye)
    return operator @ matrix


def _apply_cnot(matrix, control, target, qubits):
    """Apply a CNOT gate to the full matrix."""
    index = np.arange(2 ** qubits)
    bit = (index >> (qubits - 1 - control)) & 1
    return matrix[index ^ (bit << (qubits - 1 - target)), :]


def _try_standard_gates(problem):
    """Try to identify and synthesize standard gates exactly."""
    qubits = problem["qubits"]
    target = _target(problem)
    n = 2 ** qubits
    
    # Check if target is identity
    if np.allclose(target, np.eye(n), atol=1e-10):
        return []
    
    # Check for CNOT
    if qubits == 2:
        # CNOT with control=0, target=1
        cx_01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
        if np.allclose(target, cx_01, atol=1e-10):
            return [["cx", 0, 1]]
        # CNOT with control=1, target=0
        cx_10 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
        if np.allclose(target, cx_10, atol=1e-10):
            return [["cx", 1, 0]]
        # SWAP
        swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
        if np.allclose(target, swap, atol=1e-10):
            return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
        # iSWAP
        iswap = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex)
        if np.allclose(target, iswap, atol=1e-10):
            # iSWAP can be decomposed as:
            # H on both qubits, then CNOT(0,1), H on both, then phases
            # Actually: iSWAP = (H⊗H) CNOT(0,1) (H⊗H) * diag(1, 1j, 1j, -1)
            # Let's use a known decomposition
            gates = [
                ["u3", 0, 0, 0, math.pi/2],  # S gate on qubit 0
                ["u3", 1, 0, 0, math.pi/2],  # S gate on qubit 1
                ["u3", 0, math.pi/2, 0, 0],  # Sdg? No, let's use a standard decomposition
            ]
            # Standard iSWAP decomposition:
            # q0: H --•-- H --Sdg--
            # q1: H --X-- H --S--
            # Wait, let me use a cleaner approach.
            # iSWAP = (1/sqrt(2)) * [[1, 0, 0, 0], [0, 0, i, 0], [0, i, 0, 0], [0, 0, 0, 1]]
            # Actually the standard iSWAP is: [[1,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,1]]
            # Decomposition:
            # q0: --H--•--H--Sdg--
            # q1: --H--X--H--S--
            gates = [
                ["u3", 0, math.pi/2, 0, 0],  # H on q0
                ["u3", 1, math.pi/2, 0, 0],  # H on q1
                ["cx", 0, 1],
                ["u3", 0, math.pi/2, 0, 0],  # H on q0
                ["u3", 1, math.pi/2, 0, 0],  # H on q1
                ["u3", 0, 0, 0, -math.pi/2],  # Sdg on q0
                ["u3", 1, 0, 0, math.pi/2],   # S on q1
            ]
            # Verify
            u = _build(gates, 2)
            if _distance(target, u) < 1e-6:
                return gates
    
    # Check for Toffoli
    if qubits == 3:
        toffoli = np.eye(8, dtype=complex)
        toffoli[6, 6] = 0
        toffoli[6, 7] = 1
        toffoli[7, 6] = 1
        toffoli[7, 7] = 0
        if np.allclose(target, toffoli, atol=1e-10):
            # Toffoli decomposition using 6 CNOTs
            # Standard decomposition:
            # q0: --H--•----•----•--
            # q1: --H--X--H--X--H--
            # q2: --H--X--H--X--H--
            # Wait, that's not right. Let me use the standard 6-CNOT Toffoli.
            # Actually, the standard Toffoli uses 6 CNOTs:
            gates = [
                ["u3", 2, math.pi/2, 0, 0],  # H on q2
                ["cx", 1, 2],
                ["u3", 2, -math.pi/4, 0, 0], # Tdg on q2
                ["cx", 0, 2],
                ["u3", 2, math.pi/4, 0, 0],  # T on q2
                ["cx", 1, 2],
                ["u3", 2, math.pi/2, 0, 0],  # H on q2
                ["u3", 1, math.pi/2, 0, 0],  # H on q1
                ["cx", 0, 1],
                ["u3", 1, -math.pi/4, 0, 0], # Tdg on q1
                ["cx", 0, 1],
                ["u3", 1, math.pi/4, 0, 0],  # T on q1
                ["u3", 1, math.pi/2, 0, 0],  # H on q1
            ]
            u = _build(gates, 3)
            if _distance(target, u) < 1e-6:
                return gates
    
    return None


def _optimize_angles(target, qubits, layout, initial_values, max_time=10):
    """Optimize single-qubit angles using coordinate descent."""
    n_params = len(initial_values)
    best = initial_values[:]
    best_distance = _distance(target, _build(_circuit_from_values(layout, best), qubits))
    
    start = time.time()
    sweep = 0
    while time.time() - start < max_time:
        step = 1.0 / (1 + sweep)
        improved = False
        for i in range(n_params):
            for trial in (best[i] + step, best[i] - step):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build(_circuit_from_values(layout, values), qubits))
                if value < best_distance:
                    best, best_distance = values, value
                    improved = True
        sweep += 1
        if not improved and sweep > 5:
            break
    
    return best, best_distance


def _circuit_from_values(layout, values):
    """Create circuit from layout and parameter values."""
    gates, k = [], 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
            k += 3
        else:
            gates.append(list(gate))
    return gates


def _make_layout(qubits, budget):
    """Create a circuit layout with CNOTs and single-qubit gates."""
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    return layout


def synthesize(problem):
    """Synthesize a circuit for the target unitary within the CNOT budget."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Try standard gates first
    std_gates = _try_standard_gates(problem)
    if std_gates is not None:
        return std_gates
    
    # Create layout
    layout = _make_layout(qubits, budget)
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    # Initialize with zeros
    initial_values = [0.0] * n_params
    
    # Optimize
    best, best_distance = _optimize_angles(target, qubits, layout, initial_values, max_time=_time_remaining() - 10)
    
    return _circuit_from_values(layout, best)
# EVOLVE-BLOCK-END
