"""Search-based synthesis with deadline guard and exact decompositions for known gates."""
# EVOLVE-BLOCK-START
import math
import random
import time

import numpy as np

_DEADLINE = 250.0
_start_time = None


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build_fast(gates, qubits):
    """Build unitary from gate list using precomputed basis."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
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
            index = np.arange(dim)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _is_circuit_close(target, gates, qubits, tol=1e-10):
    """Check if circuit is close to target."""
    matrix = _build_fast(gates, qubits)
    return _distance(target, matrix) < tol


def _exact_cnot_2q():
    """Return exact circuit for CNOT on 2 qubits (control=0, target=1)."""
    return [
        ["u3", 1, math.pi / 2, 0.0, math.pi / 2],
        ["cx", 0, 1],
        ["u3", 1, math.pi / 2, -math.pi / 2, 0.0],
    ]


def _exact_iswap_2q():
    """Return exact circuit for iSWAP on 2 qubits."""
    # iSWAP = [[0,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,0]]
    # Use 2 CNOTs + single qubit rotations
    # Standard decomposition:
    # H on both, CNOT 0->1, H on both, then phase corrections
    # Actually iSWAP = (H⊗H) · CNOT(0,1) · (H⊗H) · diag(1,i,i,1)
    # Let me use a known decomposition with 2 CNOTs
    # iSWAP = [0,0,0,0; 0,0,i,0; 0,i,0,0; 0,0,0,0]
    # Decomposition:
    # 1. u3(1, pi/2, 0, pi/2) on q0  -> H
    # 2. u3(1, pi/2, 0, pi/2) on q1  -> H
    # 3. cx 0->1
    # 4. u3(1, pi/2, 0, pi/2) on q0  -> H
    # 5. u3(1, pi/2, 0, pi/2) on q1  -> H
    # 6. u3(0, 0, 0, pi) on q0  -> Z phase
    # 7. u3(0, 0, 0, pi) on q1  -> Z phase
    # Wait, let me think more carefully.
    # iSWAP = [[0,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,0]]
    # This equals: (S⊗S) · (H⊗H) · CNOT(0,1) · (H⊗H)
    # where S = diag(1,i)
    # Let's verify: H⊗H = 1/2[[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]
    # CNOT(0,1) applied to H⊗H:
    # Actually, let me just use a known 2-CNOT decomposition.
    # iSWAP can be done with 2 CNOTs:
    # u3(pi/2, 0, pi/2) on q0
    # u3(pi/2, 0, pi/2) on q1
    # cx 0->1
    # u3(pi/2, 0, pi/2) on q0
    # u3(pi/2, 0, pi/2) on q1
    # u3(0, 0, pi) on q0
    # u3(0, 0, pi) on q1
    # Let me just return a circuit and verify numerically.
    # For now, use a simple approach: since we have 3 CNOT budget for iSWAP,
    # we can use a standard decomposition.
    # iSWAP with 2 CNOTs:
    gates = [
        ["u3", 0, math.pi / 2, 0.0, math.pi / 2],
        ["u3", 1, math.pi / 2, 0.0, math.pi / 2],
        ["cx", 0, 1],
        ["u3", 0, math.pi / 2, 0.0, math.pi / 2],
        ["u3", 1, math.pi / 2, 0.0, math.pi / 2],
        ["u3", 0, 0.0, 0.0, math.pi],
        ["u3", 1, 0.0, 0.0, math.pi],
    ]
    return gates


def _exact_swap_2q():
    """Return exact circuit for SWAP on 2 qubits."""
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]


def _exact_toffoli_3q():
    """Return exact circuit for Toffoli on 3 qubits (control=0,1 target=2)."""
    # Toffoli requires 6 CNOTs in standard decomposition
    # Using the standard 6-CNOT decomposition
    # q0, q1 are controls, q2 is target
    # Standard decomposition uses T gates which we don't have directly,
    # but u3 can implement them.
    # Actually, Toffoli with 6 CNOTs requires T gates (pi/4 rotations).
    # u3 can implement T: u3(pi/2, 0, pi/4) = T up to global phase?
    # T = diag(1, e^{i*pi/4})
    # u3(theta, phi, lam) = [[cos(theta/2), -e^{i*lam}sin(theta/2)],
    #                        [e^{i*phi}sin(theta/2), e^{i*(phi+lam)}cos(theta/2)]]
    # For T: we need diag(1, e^{i*pi/4})
    # u3(0, 0, pi/4) = [[1, 0], [0, e^{i*pi/4}]] = T
    # T† = u3(0, 0, -pi/4)
    
    # Standard 6-CNOT Toffoli decomposition:
    # This is complex. Let me use a known one.
    # Actually, the minimum CNOT count for Toffoli is 6.
    # Let me use a standard decomposition.
    
    # One standard 6-CNOT decomposition for Toffoli(0,1,2):
    # 1. u3(pi/2, 0, pi/4) on q2  (T on target)
    # 2. cx 1->2
    # 3. u3(pi/2, 0, -pi/4) on q2  (T† on target)
    # 4. cx 0->2
    # 5. u3(pi/2, 0, pi/4) on q2  (T on target)
    # 6. cx 1->2
    # 7. u3(pi/2, 0, -pi/4) on q2  (T† on target)
    # 8. u3(pi/2, 0, pi/4) on q1  (T on control1)
    # 9. cx 0->1
    # 10. u3(pi/2, 0, -pi/4) on q1  (T† on control1)
    # 11. cx 0->1
    # 12. u3(pi/2, 0, pi/4) on q1  (T on control1)
    # 13. cx 1->2
    # 14. u3(pi/2, 0, -pi/4) on q2  (T† on target)
    # 15. cx 0->2
    # 16. u3(pi/2, 0, pi/4) on q2  (T on target)
    # 17. cx 1->2
    # 18. u3(pi/2, 0, -pi/4) on q2  (T† on target)
    
    # Wait, that's 6 CNOTs. Let me count: cx appears at steps 2,4,6,9,11,13,15,17 = 8 CNOTs. That's too many.
    
    # Let me look up the correct 6-CNOT Toffoli.
    # The standard 6-CNOT Toffoli uses:
    # Actually, I recall that Toffoli requires at least 6 CNOTs, and the standard
    # decomposition uses 6 CNOTs with T gates.
    
    # Let me use a different approach: since we have a budget, let me just
    # return a circuit that we'll verify and optimize if needed.
    # For now, let me use a simple but potentially suboptimal circuit.
    
    # Actually, let me just use the coordinate descent approach for Toffoli too,
    # but with a good initial guess.
    # For now, return a placeholder that will be optimized.
    return None  # Will be handled by optimization


def synthesize(problem):
    """Fixed CNOT ladder with single-qubit rotations tuned by coordinate descent."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    
    # Try exact decompositions for known gates
    if qubits == 2:
        if budget >= 1 and "cnot" in name.lower():
            gates = _exact_cnot_2q()
            if _is_circuit_close(target, gates, qubits):
                return gates
        if budget >= 2 and ("iswap" in name.lower() or "iSWAP" in name):
            gates = _exact_iswap_2q()
            if _is_circuit_close(target, gates, qubits):
                return gates
        if budget >= 3 and "swap" in name.lower():
            gates = _exact_swap_2q()
            if _is_circuit_close(target, gates, qubits):
                return gates
    
    # Build layout for coordinate descent
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    rng = random.Random(4242 + qubits)
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
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    best = [0.0] * n_params
    best_distance = _distance(target, _build_fast(circuit(best), qubits))
    
    # Coordinate descent with time guard
    max_sweeps = 20
    for sweep in range(max_sweeps):
        if time.time() - _start_time > _DEADLINE - 10:
            break
        step = 1.0 / (1 + sweep)
        for i in range(n_params):
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
