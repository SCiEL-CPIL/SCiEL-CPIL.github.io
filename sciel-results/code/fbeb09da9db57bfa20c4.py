"""Exact quantum circuit synthesis within published CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 210.0
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    eye = np.eye(2, dtype=complex)
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
            index = np.arange(dim)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _u3(q, theta, phi, lam):
    return ["u3", int(q), float(theta), float(phi), float(lam)]


def _cx(c, t):
    return ["cx", int(c), int(t)]


def _exact_2qubit(target, qubits=2):
    """Try to decompose a 2-qubit unitary exactly using standard gate sequences."""
    # Check if it's a simple gate
    # CNOT: |00><00| + |01><01| + |10><11| + |11><10|
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    iswap = np.array([[0, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 0]], dtype=complex)
    swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)

    for ref_name, ref_mat, gates in [
        ("cnot", cnot, [_cx(0, 1)]),
        ("iswap", iswap, None),
        ("swap", swap, [_cx(0, 1), _cx(1, 0), _cx(0, 1)]),
    ]:
        if np.allclose(target, ref_mat, atol=1e-10):
            if ref_name == "iswap":
                # iSWAP = H(0) H(1) CNOT(0,1) CNOT(1,0) H(0) H(1) * S(0) S(1) * phase
                # Actually iSWAP = diag(1,1,-1,-1) * iSWAP_nophs
                # iSWAP = (S ⊗ S) · (H ⊗ H) · CNOT(0,1) · CNOT(1,0) · (H ⊗ H)
                # Let's use a known decomposition: iSWAP = (iS ⊗ iS) · (H ⊗ H) · CNOT(0,1) · CNOT(1,0) · (H ⊗ H)
                # Where S = diag(1, i)
                # Actually simpler: iSWAP = (1/√2) [[0,0,0,0],[0,0,1,i],[0,1,i,0],[0,0,0,0]]
                # Standard decomposition uses 2 CNOTs
                gates = [
                    _u3(0, 0, 0, math.pi / 2),  # S on q0
                    _u3(1, 0, 0, math.pi / 2),  # S on q1
                    _u3(0, math.pi / 2, 0, 0),  # H on q0
                    _u3(1, math.pi / 2, 0, 0),  # H on q1
                    _cx(0, 1),
                    _cx(1, 0),
                    _u3(0, math.pi / 2, 0, 0),  # H on q0
                    _u3(1, math.pi / 2, 0, 0),  # H on q1
                    _u3(0, 0, 0, -math.pi / 2),  # S† on q0
                    _u3(1, 0, 0, -math.pi / 2),  # S† on q1
                ]
            return gates
    return None


def _exact_toffoli(target, qubits=3):
    """Check if target is Toffoli gate."""
    # Toffoli: |110> -> |111>, others unchanged
    toffoli = np.eye(8, dtype=complex)
    toffoli[6, 6] = 0
    toffoli[7, 7] = 0
    toffoli[6, 7] = 1
    toffoli[7, 6] = 1
    if np.allclose(target, toffoli, atol=1e-10):
        # Toffoli decomposition: 6 CNOTs
        gates = [
            _u3(2, math.pi / 2, 0, 0),  # H on target
            _cx(1, 2),
            _u3(2, 0, 0, -math.pi / 4),  # T†
            _cx(0, 2),
            _u3(2, 0, 0, math.pi / 4),  # T
            _cx(1, 2),
            _u3(2, 0, 0, -math.pi / 4),  # T†
            _cx(0, 2),
            _u3(2, 0, 0, math.pi / 4),  # T
            _u3(2, math.pi / 2, 0, 0),  # H on target
        ]
        return gates
    return None


def _decompose_3qubit_kak(target, qubits=3):
    """
    Decompose a 3-qubit unitary using CNOT ladder with single-qubit rotations.
    Uses a fixed topology and optimizes the single-qubit parameters.
    """
    budget = 6  # typical CNOT budget for 3-qubit
    # Fixed CNOT topology for 3 qubits
    pairs = [(0, 1), (0, 2), (1, 2), (0, 1), (0, 2), (1, 2)]
    
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", pairs[layer % len(pairs)][0], pairs[layer % len(pairs)][1]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    best = np.random.uniform(-math.pi, math.pi, n_params)
    best_dist = _distance(target, _build(_make_gates(layout, best), qubits))
    
    # Coordinate descent with adaptive steps
    step = 0.5
    for iteration in range(500):
        if _elapsed() > _DEADLINE - 30:
            break
        improved = False
        for i in range(n_params):
            for trial in [best[i] + step, best[i] - step]:
                trial = ((trial + math.pi) % (2 * math.pi)) - math.pi
                new_best = best.copy()
                new_best[i] = trial
                new_dist = _distance(target, _build(_make_gates(layout, new_best), qubits))
                if new_dist < best_dist:
                    best, best_dist = new_best, new_dist
                    improved = True
        if not improved:
            step *= 0.7
            if step < 1e-6:
                break
    return _make_gates(layout, best), best_dist


def _make_gates(layout, params):
    gates = []
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(_u3(gate[1], params[k], params[k + 1], params[k + 2]))
            k += 3
        else:
            gates.append(_cx(gate[1], gate[2]))
    return gates


def synthesize(problem):
    """Fixed CNOT ladder with single-qubit rotations tuned by coordinate descent."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    
    # Try exact decompositions first
    if qubits == 2:
        exact = _exact_2qubit(target, qubits)
        if exact is not None:
            return exact
    elif qubits == 3:
        exact = _exact_toffoli(target, qubits)
        if exact is not None:
            return exact
    
    # For generic unitaries, use optimization
    if qubits == 2:
        # 2-qubit: use up to 3 CNOTs
        pairs = [(0, 1), (1, 0), (0, 1)]
        layout = []
        for layer in range(min(budget, 3)):
            layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
            layout.append(["cx", pairs[layer % len(pairs)][0], pairs[layer % len(pairs)][1]])
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    else:
        # 3-qubit: use budget CNOTs
        pairs = [(0, 1), (0, 2), (1, 2), (0, 1), (0, 2), (1, 2), (0, 1), (1, 0), (2, 0), (2, 1)]
        layout = []
        for layer in range(budget):
            layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
            layout.append(["cx", pairs[layer % len(pairs)][0], pairs[layer % len(pairs)][1]])
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    rng = np.random.RandomState(42)
    best = rng.uniform(-math.pi, math.pi, n_params)
    best_dist = _distance(target, _build(_make_gates(layout, best), qubits))
    
    # Coordinate descent
    step = 1.0
    for sweep in range(30):
        if _elapsed() > _DEADLINE - 20:
            break
        for i in range(n_params):
            for trial in [best[i] + step, best[i] - step, rng.uniform(-math.pi, math.pi)]:
                trial = ((trial + math.pi) % (2 * math.pi)) - math.pi
                new_best = best.copy()
                new_best[i] = trial
                new_dist = _distance(target, _build(_make_gates(layout, new_best), qubits))
                if new_dist < best_dist:
                    best, best_dist = new_best, new_dist
        step *= 0.8
        if step < 1e-4:
            break
    
    return _make_gates(layout, best)
# EVOLVE-BLOCK-END
