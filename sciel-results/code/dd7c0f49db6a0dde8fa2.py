"""Exact quantum circuit synthesis within published CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np

_START = time.time()
_DEADLINE = _START + 235.0  # safety margin before 270s kill


def _time_left():
    return _DEADLINE - time.time()


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


# ---------- fast circuit building ----------
def _cnot_perm(qubits, control, target):
    """Return index permutation such that matrix = P @ matrix where P rows = index^(bit<<tb)."""
    n = 2 ** qubits
    idx = np.arange(n)
    cb = (idx >> (qubits - 1 - control)) & 1
    tb = qubits - 1 - target
    return idx ^ (cb << tb)


def _u3_mat(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build(gates, qubits, perms=None):
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_mat(theta, phi, lam)
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye2)
            U = op @ U
        else:
            _, c, t = gate
            if perms is None:
                p = _cnot_perm(qubits, c, t)
            else:
                p = perms[(c, t)]
            U = U[p, :]
    return U


# ---------- distance ----------
def _distance(target, U):
    tr = np.trace(target.conj().T @ U)
    return 1.0 - min(1.0, abs(tr) / target.shape[0])


# ---------- known gate templates ----------
def _phase_inv_dist(A, B):
    d = abs(np.trace(A.conj().T @ B)) / A.shape[0]
    return 1.0 - min(1.0, d)


def _all_qubit_unitaries(qubits):
    """Yield (name, gate_spec) for known gates on every qubit subset."""
    from itertools import permutations, combinations

    def cx_mat(qubits, c, t):
        n = 2 ** qubits
        m = np.eye(n, dtype=complex)
        idx = np.arange(n)
        cb = (idx >> (qubits - 1 - c)) & 1
        m = m[idx ^ (cb << (qubits - 1 - t)), :]
        return m

    def swap_mat(qubits, a, b):
        n = 2 ** qubits
        m = np.eye(n, dtype=complex)
        idx = np.arange(n)
        p = idx.copy()
        ab = (idx >> (qubits - 1 - a)) & 1
        bb = (idx >> (qubits - 1 - b)) & 1
        p = (idx ^ ((ab ^ bb) << (qubits - 1 - a)) ^ ((ab ^ bb) << (qubits - 1 - b)))
        m = m[p, :]
        return m

    def iswap_mat(qubits, a, b):
        n = 2 ** qubits
        m = np.eye(n, dtype=complex)
        # iSWAP: |01><10| and |10><01| with factor i
        for i in range(n):
            ba = (i >> (qubits - 1 - a)) & 1
            bb = (i >> (qubits - 1 - b)) & 1
            if ba != bb:
                j = i ^ ((1) << (qubits - 1 - a)) ^ ((1) << (qubits - 1 - b))
                m[j, i] = 1j
        return m

    def toffoli_mat(qubits, c1, c2, t):
        n = 2 ** qubits
        m = np.eye(n, dtype=complex)
        idx = np.arange(n)
        b1 = (idx >> (qubits - 1 - c1)) & 1
        b2 = (idx >> (qubits - 1 - c2)) & 1
        both = b1 & b2
        m = m[idx ^ (both << (qubits - 1 - t)), :]
        return m

    yield "CNOT", lambda c, t: (cx_mat(qubits, c, t), [["cx", c, t]])
    yield "SWAP", lambda a, b: (swap_mat(qubits, a, b),
                                 [["cx", a, b], ["cx", b, a], ["cx", a, b]])
    yield "iSWAP", lambda a, b: (iswap_mat(qubits, a, b), _iswap_circuit(qubits, a, b))
    yield "TOFFOLI", lambda c1, c2, t: (toffoli_mat(qubits, c1, c2, t), _toffoli_circuit(qubits, c1, c2, t))


def _iswap_circuit(qubits, a, b):
    # iSWAP via 2 CNOTs + single-qubit rotations
    # Standard: (iSWAP) = single qubit gates + CX + single + CX + single
    # Use known decomposition:
    # H on both, then ... we use a robust 2-CNOT form:
    # iSWAP = [S† H on a, S† H on b] CX(a,b) [S H on a, S H on b] CX(a,b) [S† H on a, S† H on b]
    gates = []
    # u3 params: S = u3(pi/2, 0, pi/2); H = u3(pi/2, 0, pi); Sdag = u3(-pi/2, 0, -pi/2)
    def S(q): return ["u3", q, math.pi / 2, 0.0, math.pi / 2]
    def H(q): return ["u3", q, math.pi / 2, 0.0, math.pi]
    def Sdag(q): return ["u3", q, -math.pi / 2, 0.0, -math.pi / 2]
    gates += [Sdag(a), Sdag(b)]
    gates.append(["cx", a, b])
    gates += [S(a), S(b)]
    gates.append(["cx", a, b])
    gates += [Sdag(a), Sdag(b)]
    return gates


def _toffoli_circuit(qubits, c1, c2, t):
    # 6-CNOT Toffoli (no ancilla) standard decomposition
    pi = math.pi
    def u3(q, th, ph, la): return ["u3", q, th, ph, la]
    gates = []
    gates.append(u3(t, pi / 4, 0, 0))
    gates.append(["cx", c1, t])
    gates.append(u3(t, -pi / 4, 0, 0))
    gates.append(["cx", c2, t])
    gates.append(u3(t, pi / 4, 0, 0))
    gates.append(["cx", c1, t])
    gates.append(u3(t, -pi / 4, 0, 0))
    gates.append(u3(c2, 0, 0, pi / 4))
    gates.append(["cx", t, c2])
    gates.append(u3(t, pi / 8, 0, 0))
    gates.append(u3(c2, 0, 0, -pi / 8))
    gates.append(["cx", c1, t])
    gates.append(u3(t, -pi / 8, 0, 0))
    gates.append(u3(c2, 0, 0, pi / 8))
    gates.append(["cx", t, c2])
    gates.append(u3(c2, 0, 0, -pi / 4))
    gates.append(["cx", c1, t])
    gates.append(u3(t, pi / 4, 0, 0))
    gates.append(["cx", c2, t])
    gates.append(u3(t, -pi / 4, 0, 0))
    return gates


def _try_known(problem, target, qubits, budget):
    """Try to match target to a known gate template; return circuit or None."""
    best = None
    for name, fn in _all_qubit_unitaries(qubits):
        if name in ("CNOT", "SWAP", "iSWAP"):
            pairs = [(a, b) for a in range(qubits) for b in range(qubits) if a != b]
        else:  # TOFFOLI
            pairs = [(a, b, c) for a in range(qubits) for b in range(qubits) for c in range(qubits)
                     if len({a, b, c}) == 3]
        for spec in pairs:
            if name == "TOFFOLI":
                c1, c2, t = spec
                m, circuit = fn(c1, c2, t)
            else:
                a, b = spec
                m, circuit = fn(a, b)
            if sum(1 for g in circuit if g[0] == "cx") > budget:
                continue
            d = _phase_inv_dist(target, m)
            if best is None or d < best[0]:
                best = (d, circuit)
    if best is not None and best[0] < 1e-9:
        return best[1]
    return None


# ---------- generic synthesis via 2-qubit block decomposition ----------
def _generic_synthesize(problem, target, qubits, budget):
    """Decompose a generic unitary using a sequence of 2-qubit rotations + singles,
    then optimize single-qubit params with coordinate descent. Fast build."""
    from itertools import permutations

    n = qubits
    # choose a CNOT layout within budget: cycle through pairs
    pairs = [(c, t) for c in range(n) for t in range(n) if c != t]
    # Build a ladder: alternate CNOTs with single-qubit layers
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(n)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(n)]

    # precompute CNOT perms
    perms = {}
    for (c, t) in set((g[1], g[2]) for g in layout if g[0] == "cx"):
        perms[(c, t)] = _cnot_perm(n, c, t)

    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates

    nparams = 3 * sum(1 for g in layout if g[0] == "u3")
    best = np.zeros(nparams)
    best_d = _distance(target, _build(circuit(best), n, perms))

    # coordinate descent with adaptive step
    step = 1.0
    for sweep in range(200):
        if _time_left() < 5.0:
            break
        improved = False
        for i in range(nparams):
            if _time_left() < 5.0:
                break
            for trial in (best[i] + step, best[i] - step):
                v = best.copy()
                v[i] = trial
                d = _distance(target, _build(circuit(v), n, perms))
                if d < best_d:
                    best, best_d = v, d
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
    return circuit(best)


def synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    # 1) try known exact gates
    known = _try_known(problem, target, qubits, budget)
    if known is not None:
        # verify CNOT count and gate count
        if sum(1 for g in known if g[0] == "cx") <= budget and len(known) <= 400:
            return known

    # 2) generic synthesis
    return _generic_synthesize(problem, target, qubits, budget)
# EVOLVE-BLOCK-END
