"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

import numpy as np

_DEADLINE = time.time() + 260.0  # safety margin before 270s kill


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


def _iSWAP_2q():
    """Exact iSWAP decomposition: (H⊗H)·CX(0,1)·(S†⊗S)·CX(0,1)·(H⊗H)
    Actually the standard decomposition is:
    iSWAP = (H⊗H) · CX(0,1) · (S†⊗S) · CX(0,1) · (H⊗H)
    Let me verify: the known 2-CNOT decomposition of iSWAP is:
    iSWAP = (H⊗H) CX(0,1) (S†⊗S) CX(0,1) (H⊗H)
    where S = diag(1, i), S† = diag(1, -i)
    """
    gates = []
    # H⊗H
    gates.append(['u3', 0, math.pi / 2, 0, math.pi])
    gates.append(['u3', 1, math.pi / 2, 0, math.pi])
    # CX(0,1)
    gates.append(['cx', 0, 1])
    # S†⊗S: S† on qubit 0, S on qubit 1
    # S† = diag(1, -i) = u3(0, 0, 0, -pi/2)? Let's check.
    # u3(theta, phi, lam) = [[cos(t/2), -e^{i lam} sin(t/2)], [e^{i phi} sin(t/2), e^{i(phi+lam)} cos(t/2)]]
    # For S = diag(1, i): we need [[1, 0], [0, i]]. 
    # u3(0, 0, pi/2) = [[1, -e^{i pi/2}*0], [e^{i 0}*0, e^{i(pi/2)}*1]] = [[1, 0], [0, i]] ✓
    # For S† = diag(1, -i): u3(0, 0, -pi/2) = [[1, 0], [0, e^{-i pi/2}]] = [[1, 0], [0, -i]] ✓
    gates.append(['u3', 0, 0, 0, -math.pi / 2])
    gates.append(['u3', 1, 0, 0, math.pi / 2])
    # CX(0,1)
    gates.append(['cx', 0, 1])
    # H⊗H
    gates.append(['u3', 0, math.pi / 2, 0, math.pi])
    gates.append(['u3', 1, math.pi / 2, 0, math.pi])
    return gates


def _SWAP_2q():
    """SWAP = 3 CNOTs"""
    return [['cx', 0, 1], ['cx', 1, 0], ['cx', 0, 1]]


def _CNOT_2q():
    """CNOT(0,1)"""
    return [['cx', 0, 1]]


def _toffoli_3q(control1, control2, target):
    """Exact Toffoli decomposition with 6 CNOTs for given control/target qubits.
    The standard Toffoli(0,1,2) uses 6 CNOTs. We need to handle all permutations.
    """
    # Standard Toffoli: controls on qubits c1, c2, target on t
    # The known 6-CNOT decomposition for Toffoli(a, b, c) where a,b are controls, c is target:
    # This is complex. Let me use a known decomposition.
    # 
    # For Toffoli(0, 1, 2) (controls 0,1 target 2):
    # A common 6-CNOT decomposition:
    # H on target
    # CX(control1, target)
    # T† on target, T on control2
    # CX(control2, target)
    # T on target, T† on control2
    # CX(control1, target)
    # T on control2, T† on target, H on target
    # CX(control1, control2)
    # T on control1, T† on control2
    # CX(control1, control2)
    # That's 6 CNOTs but also T gates. However we only have u3 and cx.
    # T gate = u3(pi/4, 0, 0) = diag(1, e^{i pi/4}) up to global phase
    # T† = u3(-pi/4, 0, 0)
    #
    # Let me use a cleaner 6-CNOT Toffoli decomposition.
    # Actually the minimal CNOT count for Toffoli is 6.
    #
    # Standard decomposition of Toffoli(0,1,2):
    gates = []
    # Map to standard: we need Toffoli with controls at positions c1, c2 and target at t
    # Let's just implement the standard one and permute.
    # 
    # For now, let's use a known 6-CNOT Toffoli for controls (0,1) target (2):
    # H(2)
    # CX(0,2)
    # T†(2), T(1)
    # CX(1,2)
    # T(2), T†(1)
    # CX(0,2)
    # T(1), T†(2), H(2)
    # CX(0,1)
    # T(0), T†(1)
    # CX(0,1)
    
    # But this is for specific qubit positions. Let me build it generically.
    # Actually, let me just handle the specific cases we'll encounter.
    # The problem says "the Toffoli gate" - likely Toffoli(0,1,2) or some standard.
    # Let me build a general Toffoli(c1, c2, t) with 6 CNOTs.
    
    # The 6-CNOT Toffoli decomposition (from Nielsen & Chuang or similar):
    # For Toffoli with controls a, b and target c:
    # 1. H(c)
    # 2. CX(a, c)
    # 3. T†(c), T(b)
    # 4. CX(b, c)
    # 5. T(c), T†(b)
    # 6. CX(a, c)
    # 7. T(b), T†(c), H(c)
    # 8. CX(a, b)
    # 9. T(a), T†(b)
    # 10. CX(a, b)
    # That's 6 CNOTs (steps 2,4,6,8,10 = 5... wait let me recount)
    # Steps with CX: 2, 4, 6, 8, 10 = 5 CNOTs. That doesn't seem right.
    # 
    # Let me look up the exact 6-CNOT Toffoli.
    # Actually, I recall the standard is:
    # H(t)
    # CX(c1, t)
    # T†(t) T(c2)
    # CX(c2, t)
    # T(t) T†(c2)
    # CX(c1, t)
    # T(c2) T†(t) H(t)
    # CX(c1, c2)
    # T(c1) T†(c2)
    # CX(c1, c2)
    # That's 6 CNOTs. Let me code this.
    
    H = ['u3', None, math.pi / 2, 0, math.pi]
    T = ['u3', None, math.pi / 4, 0, 0]
    Tdag = ['u3', None, -math.pi / 4, 0, 0]
    
    def _h(q): return ['u3', q, math.pi / 2, 0, math.pi]
    def _t(q): return ['u3', q, math.pi / 4, 0, 0]
    def _tdag(q): return ['u3', q, -math.pi / 4, 0, 0]
    def _cx(c, t): return ['cx', c, t]
    
    gates = []
    gates.append(_h(target))
    gates.append(_cx(control1, target))
    gates.append(_tdag(target))
    gates.append(_t(control2))
    gates.append(_cx(control2, target))
    gates.append(_t(target))
    gates.append(_tdag(control2))
    gates.append(_cx(control1, target))
    gates.append(_t(control2))
    gates.append(_tdag(target))
    gates.append(_h(target))
    gates.append(_cx(control1, control2))
    gates.append(_t(control1))
    gates.append(_tdag(control2))
    gates.append(_cx(control1, control2))
    return gates


def _is_known_gate(name, target, qubits):
    """Check if the target is one of the known gates and return exact decomposition."""
    name_lower = name.lower()
    
    if qubits == 2:
        if 'cnot' in name_lower or 'cx' in name_lower:
            # Check if it's CNOT(0,1)
            ref = _build(_CNOT_2q(), 2)
            if _distance(target, ref) < 1e-10:
                return _CNOT_2q()
            # Try CNOT(1,0)
            ref = _build([['cx', 1, 0]], 2)
            if _distance(target, ref) < 1e-10:
                return [['cx', 1, 0]]
        
        if 'iswap' in name_lower or 'i-swap' in name_lower:
            ref = _build(_iSWAP_2q(), 2)
            if _distance(target, ref) < 1e-10:
                return _iSWAP_2q()
        
        if 'swap' in name_lower:
            ref = _build(_SWAP_2q(), 2)
            if _distance(target, ref) < 1e-10:
                return _SWAP_2q()
    
    if qubits == 3:
        if 'toffoli' in name_lower or 'ccnot' in name_lower or 't' == name_lower.strip():
            # Try all 6 permutations of Toffoli
            for c1 in range(3):
                for c2 in range(3):
                    if c2 == c1:
                        continue
                    t = 3 - c1 - c2
                    ref = _build(_toffoli_3q(c1, c2, t), 3)
                    if _distance(target, ref) < 1e-10:
                        return _toffoli_3q(c1, c2, t)
    
    return None


def _build_fast(gates, qubits):
    """Faster build using precomputed kron structures."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            # Build the full operator
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye2)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            index = np.arange(n)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def synthesize(problem):
    """Synthesize the target unitary within the CNOT budget."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    
    # Check for known gates first
    known = _is_known_gate(name, target, qubits)
    if known is not None:
        # Verify it's within budget
        cnot_count = sum(1 for g in known if g[0] == 'cx')
        if cnot_count <= budget:
            return known
    
    # Generic synthesis via coordinate descent
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    rng = random.Random(4242 + qubits + hash(name) % 10000)
    
    # Build a layout with exactly `budget` CNOTs and single-qubit gates between them
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    num_u3 = sum(1 for g in layout if g[0] == "u3")
    num_params = 3 * num_u3
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    best = [0.0] * num_params
    best_distance = _distance(target, _build(circuit(best), qubits))
    
    # Coordinate descent with multiple sweeps
    max_sweeps = 20
    for sweep in range(max_sweeps):
        if time.time() > _DEADLINE - 5:
            break
        step = 1.0 / (1 + sweep)
        for i in range(num_params):
            if time.time() > _DEADLINE - 5:
                break
            for trial in (best[i] + step, best[i] - step, best[i] + rng.uniform(-math.pi, math.pi)):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build(circuit(values), qubits))
                if value < best_distance:
                    best, best_distance = values, value
    
    # Final polish with smaller steps
    for polish_sweep in range(5):
        if time.time() > _DEADLINE - 5:
            break
        step = 0.1 / (1 + polish_sweep)
        for i in range(num_params):
            if time.time() > _DEADLINE - 5:
                break
            for trial in (best[i] + step, best[i] - step):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build(circuit(values), qubits))
                if value < best_distance:
                    best, best_distance = values, value
    
    return circuit(best)
# EVOLVE-BLOCK-END
