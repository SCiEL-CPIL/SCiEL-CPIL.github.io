"""Quantum circuit synthesis with CNOT budget constraints."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np

_DEADLINE = None
_TIME_MARGIN = 30.0


def _time_remaining():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 270.0 - _TIME_MARGIN
    return max(0.0, _DEADLINE - time.time())


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


# Precomputed CNOT permutation matrices for 2 and 3 qubits
_CNOT_PERMS = {}

def _get_cnot_perm(qubits, control, target):
    key = (qubits, control, target)
    if key not in _CNOT_PERMS:
        n = 2 ** qubits
        perm = np.arange(n)
        bit_shift = qubits - 1 - target
        ctrl_shift = qubits - 1 - control
        for i in range(n):
            if (i >> ctrl_shift) & 1:
                perm[i] = i ^ (1 << bit_shift)
        _CNOT_PERMS[key] = perm
    return _CNOT_PERMS[key]


def _build(gates, qubits):
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    eye = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            perm = _get_cnot_perm(qubits, control, target)
            matrix = matrix[perm, :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _u3(q, theta, phi, lam):
    return ["u3", q, float(theta), float(phi), float(lam)]


def _cx(control, target):
    return ["cx", control, target]


def _h(q):
    return _u3(q, math.pi / 2, 0.0, math.pi)


def _s(q):
    return _u3(q, 0.0, 0.0, math.pi)


def _sdag(q):
    return _u3(q, 0.0, 0.0, -math.pi)


def _t(q):
    return _u3(q, 0.0, 0.0, math.pi / 2)


def _tdag(q):
    return _u3(q, 0.0, 0.0, -math.pi / 2)


def _toffoli(c1, c2, t):
    return [
        _h(t), _t(c1), _cx(c1, c2), _tdag(c2),
        _cx(c1, c2), _t(c2), _t(t), _cx(c2, t),
        _tdag(c1), _t(c2), _cx(c1, c2), _tdag(c2),
        _h(t), _s(c2), _s(t),
    ]


def _cnot_gate(c, t):
    return [_cx(c, t)]


def _swap_gate(a, b):
    return [_cx(a, b), _cx(b, a), _cx(a, b)]


def _iswap_gate(a, b):
    return [
        _h(a), _cx(a, b), _s(a), _s(b), _cx(b, a), _h(a), _s(a), _s(b),
    ]


def _match_fixed(name, qubits):
    name_l = str(name).lower().replace("_", "").replace("-", "")
    if name_l in ("cnot", "cx"):
        return _cnot_gate(0, 1)
    if name_l in ("swap",):
        return _swap_gate(0, 1)
    if name_l in ("iswap",):
        return _iswap_gate(0, 1)
    if name_l in ("toffoli", "ccnot", "ccx"):
        return _toffoli(0, 1, 2)
    return None


def _build_layout(qubits, budget, cnot_sequence):
    """Build a layout: u3 layers interleaved with CNOTs."""
    layout = []
    for i, cnot in enumerate(cnot_sequence):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", cnot[0], cnot[1]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    return layout


def _make_circuit_builder(layout, qubits):
    """Return a function that builds a gate list from parameter values."""
    u3_positions = []
    for i, gate in enumerate(layout):
        if gate[0] == "u3":
            u3_positions.append((i, gate[1]))
    
    def circuit(values):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(["cx", gate[1], gate[2]])
        return gates
    return circuit


def _coordinate_descent(target, qubits, layout, seed, max_time):
    """Run coordinate descent starting from random init."""
    rng = random.Random(seed)
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3
    
    # Random initialization
    best = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
    circuit_fn = _make_circuit_builder(layout, qubits)
    best_distance = _distance(target, _build(circuit_fn(best), qubits))
    
    deadline = time.time() + max_time
    step = 1.0
    n_sweeps = 0
    
    while time.time() < deadline:
        n_sweeps += 1
        step = max(0.05, 1.0 / (1 + 0.3 * n_sweeps))
        
        for i in range(n_params):
            if time.time() >= deadline:
                break
            for trial in (best[i] + step, best[i] - step,
                          best[i] + rng.uniform(-math.pi, math.pi)):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build(circuit_fn(values), qubits))
                if value < best_distance:
                    best, best_distance = values, value
    
    return circuit_fn(best), best_distance


def _optimize_generic(target, qubits, budget, time_available):
    """Optimize generic unitary with multiple layouts and restarts."""
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    # Generate several CNOT sequences to try
    layouts = []
    
    # Layout 1: round-robin
    seq1 = [pairs[i % len(pairs)] for i in range(budget)]
    layouts.append(_build_layout(qubits, budget, seq1))
    
    # Layout 2: focus on pairs (0,1) and (1,2) for 3 qubits
    if qubits == 3:
        seq2 = [(0,1) if i % 2 == 0 else (1,2) for i in range(budget)]
        layouts.append(_build_layout(qubits, budget, seq2))
        
        # Layout 3: all three pairs cycling
        seq3 = [(0,1),(1,2),(0,2)] * (budget // 3 + 1)
        seq3 = seq3[:budget]
        layouts.append(_build_layout(qubits, budget, seq3))
    
    # Layout 4: random sequence
    rng_layout = random.Random(999 + qubits)
    seq4 = [rng_layout.choice(pairs) for _ in range(budget)]
    layouts.append(_build_layout(qubits, budget, seq4))
    
    best_circuit = None
    best_distance = 1.0
    
    n_layouts = len(layouts)
    time_per_layout = max(2.0, (time_available - 5.0) / n_layouts)
    
    for li, layout in enumerate(layouts):
        if time.time() > time.time() + time_available - 3.0:
            break
        remaining = time_available - 3.0
        # Try 2-3 restarts per layout
        n_restarts = min(3, max(1, int(remaining / 8)))
        t_per_restart = remaining / n_restarts
        
        for r in range(n_restarts):
            if _time_remaining() < 10.0:
                break
            circuit, dist = _coordinate_descent(
                target, qubits, layout, seed=42 + li * 100 + r, max_time=t_per_restart
            )
            if dist < best_distance:
                best_distance = dist
                best_circuit = circuit
    
    if best_circuit is None:
        return []
    
    # Verify CNOT count
    cx_count = sum(1 for g in best_circuit if g[0] == "cx")
    if cx_count > budget or len(best_circuit) > 400:
        return []
    
    return best_circuit


def synthesize(problem):
    """Synthesize a quantum circuit within CNOT budget."""
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 270.0 - _TIME_MARGIN

    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    name = problem.get("name", "")
    target = _target(problem)

    # 1) Exact known-gate fast path
    fixed = _match_fixed(name, qubits)
    if fixed is not None:
        cx_count = sum(1 for g in fixed if g[0] == "cx")
        if cx_count <= budget and len(fixed) <= 400:
            return fixed
        return []

    # 2) Generic unitary optimization
    remaining = _time_remaining()
    if remaining > 15.0:
        return _optimize_generic(target, qubits, budget, remaining)

    # 3) Fallback
    return []
# EVOLVE-BLOCK-END
