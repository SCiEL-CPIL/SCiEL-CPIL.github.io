"""Rewritten program: exact quantum circuit synthesis within CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 255.0  # Safety margin before 270s wall-clock kill
_START_TIME = time.time()


def _check_deadline():
    return (time.time() - _START_TIME) > _DEADLINE


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    """Build unitary from gate list."""
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


def _apply_u3(matrix, q, theta, phi, lam, qubits):
    """Apply u3 gate to matrix (left multiply)."""
    eye = np.eye(2, dtype=complex)
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    single = np.array([[c, -np.exp(1j * lam) * s],
                       [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
    operator = np.array([[1]], dtype=complex)
    for k in range(qubits):
        operator = np.kron(operator, single if k == q else eye)
    return operator @ matrix


def _apply_cx(matrix, control, target, qubits):
    """Apply CNOT gate to matrix (left multiply via row permutation)."""
    index = np.arange(2 ** qubits)
    bit = (index >> (qubits - 1 - control)) & 1
    return matrix[index ^ (bit << (qubits - 1 - target)), :]


def _decompose_2qubit(U):
    """
    Decompose a 2-qubit unitary into CNOTs and single-qubit rotations.
    Returns (U1, U2, U3, U4, U5) where the circuit is:
    U5 on q0, U4 on q1, CNOT(0,1), U3 on q1, CNOT(0,1), U2 on q1, CNOT(0,1), U1 on q0, U0 on q1
    Actually let's use the standard KAK decomposition approach.
    For 2-qubit gates, we can use the known decomposition with 3 CNOTs.
    """
    # For now, use a simpler approach: we'll use coordinate descent on the full circuit
    # This function is a placeholder for future use
    pass


def _synthesize_2qubit(target, budget, qubits=2):
    """Synthesize a 2-qubit gate within CNOT budget."""
    if budget >= 3:
        # Standard 2-qubit decomposition uses 3 CNOTs
        # Layout: U0_q0, U0_q1, CNOT(0,1), U1_q0, U1_q1, CNOT(0,1), U2_q0, U2_q1, CNOT(0,1), U3_q0, U3_q1
        # That's 10 u3 gates and 3 CNOTs
        # But we need to handle the budget properly
        pass
    
    # Use coordinate descent approach
    # Build a layout with the given budget
    pairs = [(0, 1), (1, 0)]
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    def objective(values):
        if _check_deadline():
            return 1.0
        gates = circuit(values)
        matrix = _build(gates, qubits)
        return _distance(target, matrix)
    
    # Start from zero
    best = [0.0] * n_params
    best_dist = objective(best)
    
    # Coordinate descent with random restarts
    import random
    rng = random.Random(4242 + qubits)
    
    max_sweeps = 20
    for sweep in range(max_sweeps):
        if _check_deadline():
            break
        step = math.pi / (2 * (sweep + 1))
        improved = False
        for i in range(n_params):
            if _check_deadline():
                break
            current_val = best[i]
            for trial in (current_val + step, current_val - step, rng.uniform(-math.pi, math.pi)):
                values = best[:]
                values[i] = trial
                value = objective(values)
                if value < best_dist:
                    best, best_dist = values, value
                    improved = True
        if not improved and sweep > 5:
            break
    
    return circuit(best)


def _synthesize_3qubit(target, budget, qubits=3):
    """Synthesize a 3-qubit gate within CNOT budget."""
    # For 3-qubit gates, use a more structured approach
    # Common decompositions use 6-14 CNOTs depending on the gate
    
    pairs = [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)]
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    def objective(values):
        if _check_deadline():
            return 1.0
        gates = circuit(values)
        matrix = _build(gates, qubits)
        return _distance(target, matrix)
    
    # Start from zero
    best = [0.0] * n_params
    best_dist = objective(best)
    
    # Coordinate descent with random restarts
    import random
    rng = random.Random(4242 + qubits)
    
    max_sweeps = 15
    for sweep in range(max_sweeps):
        if _check_deadline():
            break
        step = math.pi / (2 * (sweep + 1))
        improved = False
        for i in range(n_params):
            if _check_deadline():
                break
            current_val = best[i]
            for trial in (current_val + step, current_val - step, rng.uniform(-math.pi, math.pi)):
                values = best[:]
                values[i] = trial
                value = objective(values)
                if value < best_dist:
                    best, best_dist = values, value
                    improved = True
        if not improved and sweep > 5:
            break
    
    return circuit(best)


def _exact_circuit_2qubit(target, qubits=2):
    """
    Try to find an exact or near-exact circuit for 2-qubit gates using
    known decompositions.
    """
    # For specific known gates, we can use known decompositions
    # CNOT: 1 CNOT
    # iSWAP: 2 CNOTs
    # SWAP: 3 CNOTs
    # Generic: 3 CNOTs
    
    # Use the standard 2-qubit decomposition with 3 CNOTs
    # The circuit is:
    # q0: U0_a --x-- U1_a --x-- U2_a --x-- U3_a
    # q1: U0_b ---o--- U1_b ---o--- U2_b ---o--- U3_b
    # Wait, let me think about this more carefully.
    # Standard 2-qubit decomposition:
    # U = (A ⊗ B) CNOT (C ⊗ D) CNOT (E ⊗ F) CNOT (G ⊗ H)
    # where A,B,C,D,E,F,G,H are single-qubit unitaries
    # This uses 3 CNOTs and 16 single-qubit parameters (8 gates * 2 params each, but u3 has 3)
    
    # Let's just use coordinate descent with a good initial guess
    return None


def synthesize(problem):
    """
    Synthesize a quantum circuit for the given target unitary within the CNOT budget.
    """
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    
    # Check deadline
    if _check_deadline():
        # Return identity circuit as fallback
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    # For 2-qubit gates
    if qubits == 2:
        # Try known exact decompositions first
        # CNOT: just return CNOT
        # Check if target is CNOT (up to phase)
        cnot_ref = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        dist_cnot = _distance(target, cnot_ref)
        if dist_cnot < 1e-10:
            return [["cx", 0, 1]]
        
        # Check if target is iSWAP
        iswap_ref = np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex)
        dist_iswap = _distance(target, iswap_ref)
        if dist_iswap < 1e-10:
            # iSWAP decomposition: 2 CNOTs
            # iSWAP = (H ⊗ H) CNOT (S† ⊗ S) CNOT (H ⊗ H)
            # H = u3(pi/2, 0, pi)
            # S = u3(0, 0, pi/2)
            # S† = u3(0, 0, -pi/2)
            h = [math.pi/2, 0.0, math.pi]
            s = [0.0, 0.0, math.pi/2]
            sdag = [0.0, 0.0, -math.pi/2]
            gates = [
                ["u3", 0, h[0], h[1], h[2]],
                ["u3", 1, h[0], h[1], h[2]],
                ["cx", 0, 1],
                ["u3", 0, sdag[0], sdag[1], sdag[2]],
                ["u3", 1, s[0], s[1], s[2]],
                ["cx", 0, 1],
                ["u3", 0, h[0], h[1], h[2]],
                ["u3", 1, h[0], h[1], h[2]],
            ]
            return gates
        
        # Check if target is SWAP
        swap_ref = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        dist_swap = _distance(target, swap_ref)
        if dist_swap < 1e-10:
            # SWAP = 3 CNOTs
            gates = [
                ["cx", 0, 1],
                ["cx", 1, 0],
                ["cx", 0, 1],
            ]
            return gates
        
        # Use coordinate descent for generic 2-qubit
        return _synthesize_2qubit(target, budget, qubits)
    
    # For 3-qubit gates
    elif qubits == 3:
        # Check if target is Toffoli
        # Toffoli: |00>->|00>, |01>->|01>, |10>->|10>, |11>->|00>, |00>... wait
        # Toffoli flips target when both controls are 1
        # Toffoli(0,1,2): control q0, q1, target q2
        toffoli_ref = np.zeros((8, 8), dtype=complex)
        for i in range(8):
            c0 = (i >> 2) & 1
            c1 = (i >> 1) & 1
            t = i & 1
            j = i
            if c0 == 1 and c1 == 1:
                j = i ^ 1  # flip target
            toffoli_ref[j, i] = 1.0
        dist_toffoli = _distance(target, toffoli_ref)
        if dist_toffoli < 1e-10:
            # Toffoli decomposition: 6 CNOTs
            # Standard Toffoli decomposition
            gates = [
                ["u3", 2, 0.0, 0.0, -math.pi/4],
                ["cx", 1, 2],
                ["u3", 2, 0.0, 0.0, math.pi/4],
                ["cx", 0, 2],
                ["u3", 2, 0.0, 0.0, -math.pi/4],
                ["cx", 1, 2],
                ["u3", 2, 0.0, 0.0, math.pi/4],
                ["u3", 1, math.pi/2, 0.0, math.pi/2],
                ["cx", 0, 1],
                ["u3", 1, 0.0, 0.0, -math.pi/4],
                ["cx", 2, 1],
                ["u3", 1, 0.0, 0.0, math.pi/4],
                ["cx", 0, 1],
                ["u3", 1, math.pi/2, 0.0, -math.pi/2],
            ]
            return gates
        
        # Use coordinate descent for generic 3-qubit
        return _synthesize_3qubit(target, budget, qubits)
    
    # Fallback
    return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]


# EVOLVE-BLOCK-END
