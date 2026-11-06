"""Quantum circuit synthesis with exact decompositions for known gates and optimized search for generic unitaries."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np

_START_TIME = time.time()
_DEADLINE = 255.0  # Safety margin before 270s wall-clock kill


def _check_deadline():
    return (time.time() - _START_TIME) > _DEADLINE


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    e1 = np.exp(1j * lam)
    e2 = np.exp(1j * phi)
    return np.array([[c, -e1 * s],
                     [e2 * s, e2 * e1 * c]])


def _build_fast(gates, qubits):
    """Build the full unitary from a gate list. qubit 0 is MSB."""
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    eye = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            # Build operator by kron
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye)
            U = op @ U
        else:
            _, control, target = gate
            # CNOT: if control bit is 1, flip target
            idx = np.arange(n)
            ctrl_bit = (idx >> (qubits - 1 - control)) & 1
            tgt_bit_pos = qubits - 1 - target
            # Swap rows where control is 1
            for i in range(n):
                if ctrl_bit[i] == 1:
                    j = i ^ (1 << tgt_bit_pos)
                    if i < j:
                        U[[i, j], :] = U[[j, i], :]
    return U


def _distance(target, matrix):
    d = 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])
    return d


def _is_cnot(target, qubits):
    """Check if target is a CNOT gate."""
    n = 2 ** qubits
    if qubits == 2:
        # Check all 4 possible CNOTs
        for ctrl in range(2):
            for tgt in range(2):
                if ctrl == tgt:
                    continue
                expected = np.eye(n, dtype=complex)
                for i in range(n):
                    cb = (i >> (qubits - 1 - ctrl)) & 1
                    if cb == 1:
                        j = i ^ (1 << (qubits - 1 - tgt))
                        expected[i, j] = 1.0
                        expected[j, i] = 0.0
                # Compare up to global phase
                tr = np.trace(expected.conj().T @ target)
                if abs(tr) > 0.999 * n:
                    return True
        return False
    return False


def _is_swap(target, qubits):
    """Check if target is a SWAP gate."""
    n = 2 ** qubits
    if qubits == 2:
        expected = np.array([[1, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1]], dtype=complex)
        tr = np.trace(expected.conj().T @ target)
        if abs(tr) > 0.999 * n:
            return True
    return False


def _is_iswap(target, qubits):
    """Check if target is an iSWAP gate."""
    n = 2 ** qubits
    if qubits == 2:
        expected = np.array([[1, 0, 0, 0],
                             [0, 0, 1j, 0],
                             [0, 1j, 0, 0],
                             [0, 0, 0, 1]], dtype=complex)
        tr = np.trace(expected.conj().T @ target)
        if abs(tr) > 0.999 * n:
            return True
    return False


def _is_toffoli(target, qubits):
    """Check if target is a Toffoli gate."""
    n = 2 ** qubits
    if qubits == 3:
        expected = np.eye(n, dtype=complex)
        # Flip |110> to |111> and |111> to |110> (qubit 0 is MSB)
        # |110> = 6, |111> = 7
        expected[6, 6] = 0
        expected[6, 7] = 1
        expected[7, 6] = 1
        expected[7, 7] = 0
        tr = np.trace(expected.conj().T @ target)
        if abs(tr) > 0.999 * n:
            return True
    return False


def _decompose_cnot(ctrl, tgt, qubits):
    """Return gates for a CNOT."""
    return [["cx", ctrl, tgt]]


def _decompose_swap(qubits):
    """Return gates for SWAP using 3 CNOTs."""
    if qubits == 2:
        return [
            ["cx", 0, 1],
            ["cx", 1, 0],
            ["cx", 0, 1]
        ]
    return []


def _decompose_iswap(qubits):
    """Return gates for iSWAP using 2 CNOTs and single-qubit rotations."""
    if qubits == 2:
        return [
            ["u3", 0, math.pi / 2, 0.0, math.pi / 2],
            ["u3", 1, math.pi / 2, 0.0, math.pi / 2],
            ["cx", 0, 1],
            ["u3", 0, math.pi / 2, math.pi / 2, 0.0],
            ["u3", 1, math.pi / 2, 0.0, math.pi / 2],
            ["cx", 0, 1],
            ["u3", 0, math.pi / 2, 0.0, math.pi / 2],
            ["u3", 1, math.pi / 2, 0.0, math.pi / 2]
        ]
    return []


def _decompose_toffoli(qubits):
    """Return gates for Toffoli using 6 CNOTs."""
    if qubits == 3:
        return [
            ["u3", 2, math.pi / 2, 0.0, math.pi / 2],
            ["cx", 1, 2],
            ["u3", 2, math.pi / 2, math.pi / 2, 0.0],
            ["cx", 0, 2],
            ["u3", 2, math.pi / 2, 0.0, math.pi / 2],
            ["cx", 1, 2],
            ["u3", 2, math.pi / 2, math.pi / 2, 0.0],
            ["cx", 0, 2],
            ["u3", 2, math.pi / 2, 0.0, math.pi / 2],
            ["cx", 1, 2],
            ["u3", 2, math.pi / 2, math.pi / 2, 0.0],
            ["cx", 0, 2]
        ]
    return []


def _optimize_generic(target, qubits, budget):
    """Optimize a generic unitary using coordinate descent."""
    n = 2 ** qubits
    rng = random.Random(4242 + qubits)
    
    # Build a layout with CNOTs and u3 gates
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
    
    num_params = 3 * sum(1 for g in layout if g[0] == "u3")
    best = [0.0] * num_params
    best_distance = _distance(target, _build_fast(circuit(best), qubits))
    
    # Coordinate descent with random restarts
    max_sweeps = 8
    for sweep in range(max_sweeps):
        if _check_deadline():
            break
        step = 1.0 / (1 + sweep)
        for i in range(num_params):
            if _check_deadline():
                break
            for trial in (best[i] + step, best[i] - step, best[i] + rng.uniform(-math.pi, math.pi)):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build_fast(circuit(values), qubits))
                if value < best_distance:
                    best, best_distance = values, value
        if best_distance < 1e-6:
            break
    
    return circuit(best)


def synthesize(problem):
    """Synthesize a circuit for the given target unitary."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    
    # Try exact decompositions first
    if _is_cnot(target, qubits):
        # Find which CNOT
        for ctrl in range(qubits):
            for tgt in range(qubits):
                if ctrl == tgt:
                    continue
                gates = _decompose_cnot(ctrl, tgt, qubits)
                U = _build_fast(gates, qubits)
                if _distance(target, U) < 1e-6:
                    return gates
    elif _is_swap(target, qubits):
        gates = _decompose_swap(qubits)
        if gates:
            return gates
    elif _is_iswap(target, qubits):
        gates = _decompose_iswap(qubits)
        if gates:
            return gates
    elif _is_toffoli(target, qubits):
        gates = _decompose_toffoli(qubits)
        if gates:
            return gates
    
    # Fall back to optimization
    return _optimize_generic(target, qubits, budget)
# EVOLVE-BLOCK-END
