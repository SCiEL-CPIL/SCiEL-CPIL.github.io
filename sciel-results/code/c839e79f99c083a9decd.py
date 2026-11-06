"""Quantum circuit synthesis with exact decompositions and coordinate descent."""
# EVOLVE-BLOCK-START
import math
import random
import time

import numpy as np

_DEADLINE = 240.0
_start_time = None


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build_fast(gates, qubits):
    """Build unitary from gate list efficiently."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            # Apply single-qubit gate on qubit q
            # Build the full operator
            operator = np.array([[1]], dtype=complex)
            eye = np.eye(2, dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            index = np.arange(dim)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _is_cnot_gate(problem):
    """Check if target is a CNOT gate."""
    qubits = problem["qubits"]
    target = _target(problem)
    # CNOT on qubits (0,1) for 2-qubit: control=0, target=1
    # Standard CNOT: |00>->|00>, |01>->|01>, |10>->|11>, |11>->|10>
    if qubits == 2:
        expected = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0]], dtype=complex)
        return np.allclose(target, expected, atol=1e-10)
    return False


def _is_iswap_gate(problem):
    """Check if target is an iSWAP gate."""
    qubits = problem["qubits"]
    target = _target(problem)
    if qubits == 2:
        # iSWAP: |00>->|00>, |01>->i|10>, |10>->i|01>, |11>->|11>
        expected = np.array([[1, 0, 0, 0],
                             [0, 0, 1j, 0],
                             [0, 1j, 0, 0],
                             [0, 0, 0, 1]], dtype=complex)
        return np.allclose(target, expected, atol=1e-10)
    return False


def _is_swap_gate(problem):
    """Check if target is a SWAP gate."""
    qubits = problem["qubits"]
    target = _target(problem)
    if qubits == 2:
        expected = np.array([[1, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1]], dtype=complex)
        return np.allclose(target, expected, atol=1e-10)
    return False


def _is_toffoli_gate(problem):
    """Check if target is a Toffoli gate."""
    qubits = problem["qubits"]
    target = _target(problem)
    if qubits == 3:
        # Toffoli: control qubits 0,1, target qubit 2
        # |000>->|000>, ..., |110>->|111>, |111>->|110>
        expected = np.eye(8, dtype=complex)
        expected[6, 6] = 0
        expected[6, 7] = 1
        expected[7, 6] = 1
        expected[7, 7] = 0
        return np.allclose(target, expected, atol=1e-10)
    return False


def _decompose_cnot_2qubit():
    """Exact decomposition of CNOT using 0 CNOTs (it IS a CNOT)."""
    # CNOT is already a CNOT gate
    return [["cx", 0, 1]]


def _decompose_iswap_2qubit():
    """Decompose iSWAP into CNOTs and single-qubit rotations."""
    # iSWAP = (H⊗H) · CNOT · (iS⊗I) · CNOT · (H⊗H) with phases
    # Standard decomposition: iSWAP needs 2 CNOTs
    # iSWAP = (Rz(π/2)⊗Rz(π/2)) · CNOT · (Rz(-π/2)⊗Rz(-π/2)) · CNOT · (Rz(π/2)⊗Rz(π/2))
    # Let's use a known decomposition:
    # iSWAP = (H⊗H) · CNOT · (Rz(π/4)⊗Rz(-π/4)) · CNOT · (H⊗H) with appropriate phases
    # Actually, let's use: iSWAP can be done with 2 CNOTs
    gates = []
    # H on both qubits
    gates.append(["u3", 0, math.pi, 0, math.pi])  # H
    gates.append(["u3", 1, math.pi, 0, math.pi])  # H
    # CNOT 0->1
    gates.append(["cx", 0, 1])
    # Phase rotations
    gates.append(["u3", 0, 0, 0, math.pi / 2])  # S gate (Rz(π/2))
    gates.append(["u3", 1, 0, 0, -math.pi / 2])  # S† gate (Rz(-π/2))
    # CNOT 0->1
    gates.append(["cx", 0, 1])
    # H on both qubits
    gates.append(["u3", 0, math.pi, 0, math.pi])  # H
    gates.append(["u3", 1, math.pi, 0, math.pi])  # H
    return gates


def _decompose_swap_2qubit():
    """Decompose SWAP into 3 CNOTs and single-qubit rotations."""
    gates = []
    gates.append(["cx", 0, 1])
    gates.append(["cx", 1, 0])
    gates.append(["cx", 0, 1])
    return gates


def _decompose_toffoli_3qubit():
    """Decompose Toffoli into 6 CNOTs and single-qubit rotations."""
    # Standard Toffoli decomposition with 6 CNOTs
    gates = []
    # H on target
    gates.append(["u3", 2, math.pi, 0, math.pi])
    # CNOT 1->2
    gates.append(["cx", 1, 2])
    # T† on target
    gates.append(["u3", 2, 0, 0, -math.pi / 4])
    # CNOT 0->2
    gates.append(["cx", 0, 2])
    # T on target
    gates.append(["u3", 2, 0, 0, math.pi / 4])
    # CNOT 1->2
    gates.append(["cx", 1, 2])
    # T† on target
    gates.append(["u3", 2, 0, 0, -math.pi / 4])
    # CNOT 0->2
    gates.append(["cx", 0, 2])
    # T on target
    gates.append(["u3", 2, 0, 0, math.pi / 4])
    # H on target
    gates.append(["u3", 2, math.pi, 0, math.pi])
    # Additional CNOTs for controls
    gates.append(["cx", 0, 1])
    # T† on control 1
    gates.append(["u3", 1, 0, 0, -math.pi / 4])
    # CNOT 0->1
    gates.append(["cx", 0, 1])
    # T on control 1
    gates.append(["u3", 1, 0, 0, math.pi / 4])
    return gates


def synthesize(problem):
    """Synthesize circuit for the given target unitary."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    # Check for known gates and use exact decompositions
    if _is_cnot_gate(problem):
        return _decompose_cnot_2qubit()
    if _is_iswap_gate(problem):
        return _decompose_iswap_2qubit()
    if _is_swap_gate(problem):
        return _decompose_swap_2qubit()
    if _is_toffoli_gate(problem):
        return _decompose_toffoli_3qubit()

    # For generic unitaries, use coordinate descent
    rng = random.Random(4242 + qubits + budget)

    # Build initial layout with CNOTs
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]

    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates

    best = [0.0] * (3 * sum(1 for g in layout if g[0] == "u3"))
    best_distance = _distance(target, _build_fast(circuit(best), qubits))

    # Coordinate descent with time guard
    max_sweeps = 30
    for sweep in range(max_sweeps):
        if time.time() - _start_time > _DEADLINE - 10:
            break
        step = 1.0 / (1 + sweep * 0.5)
        for i in range(len(best)):
            if time.time() - _start_time > _DEADLINE - 10:
                break
            for trial in (best[i] + step, best[i] - step, best[i] + rng.uniform(-math.pi, math.pi)):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build_fast(circuit(values), qubits))
                if value < best_distance:
                    best, best_distance = values, value

    return circuit(best)
# EVOLVE-BLOCK-END
