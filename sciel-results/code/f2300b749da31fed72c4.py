"""Synthesize fixed two- and three-qubit unitaries within CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = time.time() + 240.0


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
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
            index = np.arange(2 ** qubits)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _apply_u3(gates, q, theta, phi, lam):
    gates.append(["u3", q, theta, phi, lam])


def _cnot(gates, control, target):
    gates.append(["cx", control, target])


def _decompose_cnot():
    return [["cx", 0, 1]]


def _decompose_swap():
    gates = []
    _cnot(gates, 0, 1)
    _cnot(gates, 1, 0)
    _cnot(gates, 0, 1)
    return gates


def _decompose_iswap():
    gates = []
    _apply_u3(gates, 0, math.pi / 2, 0, math.pi)
    _apply_u3(gates, 1, math.pi / 2, 0, math.pi)
    _cnot(gates, 0, 1)
    _apply_u3(gates, 0, 0.0, 0.0, -math.pi / 2)
    _apply_u3(gates, 1, 0.0, 0.0, math.pi / 2)
    _cnot(gates, 0, 1)
    _apply_u3(gates, 0, math.pi / 2, 0, math.pi)
    _apply_u3(gates, 1, math.pi / 2, 0, math.pi)
    return gates


def _decompose_toffoli():
    gates = []
    T = math.pi / 4
    _apply_u3(gates, 2, math.pi / 2, 0, math.pi)
    _apply_u3(gates, 2, 0, 0, T)
    _cnot(gates, 1, 2)
    _apply_u3(gates, 2, 0, 0, -T)
    _cnot(gates, 0, 2)
    _apply_u3(gates, 1, 0, 0, T)
    _apply_u3(gates, 2, 0, 0, -T)
    _cnot(gates, 0, 1)
    _apply_u3(gates, 2, 0, 0, T)
    _apply_u3(gates, 1, 0, 0, -T)
    _cnot(gates, 0, 1)
    _apply_u3(gates, 2, 0, 0, -T)
    _apply_u3(gates, 1, 0, 0, T)
    _apply_u3(gates, 2, math.pi / 2, 0, math.pi)
    return gates


def _is_named_gate(target, qubits):
    """Check if target matches a named gate, return (gates, name) or (None, None)."""
    if qubits == 2:
        # CNOT
        cnot_ref = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        if _distance(cnot_ref, target) < 1e-10:
            return _decompose_cnot(), "CNOT"
        # SWAP
        swap_ref = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if _distance(swap_ref, target) < 1e-10:
            return _decompose_swap(), "SWAP"
        # iSWAP
        iswap_ref = np.array([[0,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,0]], dtype=complex)
        if _distance(iswap_ref, target) < 1e-10:
            return _decompose_iswap(), "iSWAP"
    elif qubits == 3:
        toffoli_ref = np.identity(8, dtype=complex)
        toffoli_ref[6, 7] = 1
        toffoli_ref[7, 6] = 1
        toffoli_ref[6, 6] = 0
        toffoli_ref[7, 7] = 0
        if _distance(toffoli_ref, target) < 1e-10:
            return _decompose_toffoli(), "Toffoli"
    return None, None


def _optimize_2q(target, qubits=2, budget=3):
    """Optimize a 2-qubit circuit with 3 CNOTs using coordinate descent."""
    
    def make_circuit(params):
        gates = []
        k = 0
        for layer in range(4):
            for q in range(2):
                _apply_u3(gates, q, params[k], params[k+1], params[k+2]); k += 3
            if layer < 3:
                if layer % 2 == 0:
                    _cnot(gates, 0, 1)
                else:
                    _cnot(gates, 1, 0)
        return gates
    
    n_params = 24
    best = np.zeros(n_params)
    best_dist = _distance(target, _build(make_circuit(best), qubits))
    
    deadline = time.time() + 25.0
    sweep = 0
    while time.time() < deadline and sweep < 40:
        step = 2.0 / (1 + sweep * 0.3)
        improved = False
        for i in range(n_params):
            if time.time() > deadline:
                break
            for trial in (best[i] + step, best[i] - step):
                val = best.copy()
                val[i] = trial
                dist = _distance(target, _build(make_circuit(val), qubits))
                if dist < best_dist:
                    best, best_dist = val, dist
                    improved = True
        sweep += 1
        if not improved and sweep > 8:
            break
    
    return make_circuit(best)


def _optimize_3q(target, qubits=3, budget=6):
    """Optimize a 3-qubit circuit with given CNOT budget using coordinate descent."""
    pairs = [(0, 1), (1, 2), (0, 2)]
    
    def make_circuit(params):
        gates = []
        k = 0
        for layer in range(budget + 1):
            for q in range(3):
                _apply_u3(gates, q, params[k], params[k+1], params[k+2]); k += 3
            if layer < budget:
                c, t = pairs[layer % len(pairs)]
                _cnot(gates, c, t)
        return gates
    
    n_params = (budget + 1) * 9
    best = np.zeros(n_params)
    best_dist = _distance(target, _build(make_circuit(best), qubits))
    
    deadline = time.time() + 25.0
    sweep = 0
    while time.time() < deadline and sweep < 25:
        step = 2.0 / (1 + sweep * 0.3)
        improved = False
        for i in range(n_params):
            if time.time() > deadline:
                break
            for trial in (best[i] + step, best[i] - step):
                val = best.copy()
                val[i] = trial
                dist = _distance(target, _build(make_circuit(val), qubits))
                if dist < best_dist:
                    best, best_dist = val, dist
                    improved = True
        sweep += 1
        if not improved and sweep > 8:
            break
    
    return make_circuit(best)


def synthesize(problem):
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    if time.time() > _DEADLINE - 5:
        return []
    
    # Try named gates first
    gates, _ = _is_named_gate(target, qubits)
    if gates is not None:
        cnot_count = len([g for g in gates if g[0] == "cx"])
        if cnot_count <= budget:
            return gates
    
    # Generic: use optimization
    if qubits == 2:
        return _optimize_2q(target, qubits, budget)
    elif qubits == 3:
        return _optimize_3q(target, qubits, budget)
    
    return []
# EVOLVE-BLOCK-END

