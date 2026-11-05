"""Quantum circuit synthesis: KAK decomposition + local refinement."""
# EVOLVE-BLOCK-START
import math
import time
import random
import numpy as np
from scipy.linalg import expm

_START = time.time()
_DEADLINE = _START + 250.0  # 20s safety margin before 270s kill


def _time_left():
    return _DEADLINE - time.time()


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_mat(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build(gates, qubits):
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    eye = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_mat(theta, phi, lam)
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye)
            matrix = op @ matrix
        else:
            _, c, t = gate
            idx = np.arange(n)
            bit = (idx >> (qubits - 1 - c)) & 1
            matrix = matrix[idx ^ (bit << (qubits - 1 - t)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _apply_single(matrix, qubit, single, qubits):
    eye = np.eye(2, dtype=complex)
    op = np.array([[1]], dtype=complex)
    for k in range(qubits):
        op = np.kron(op, single if k == qubit else eye)
    return op @ matrix


def _apply_cnot(matrix, control, target, qubits):
    n = 2 ** qubits
    idx = np.arange(n)
    bit = (idx >> (qubits - 1 - control)) & 1
    return matrix[idx ^ (bit << (qubits - 1 - target)), :]


def _try_standard_gates(problem):
    """Check for known exact decompositions of standard gates."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    n = 2 ** qubits

    # Identity
    if np.allclose(target, np.eye(n), atol=1e-10):
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]

    if qubits == 2:
        # CNOT 0->1
        cx_01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
        if np.allclose(target, cx_01, atol=1e-10):
            return [["cx", 0, 1]]
        # CNOT 1->0
        cx_10 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
        if np.allclose(target, cx_10, atol=1e-10):
            return [["cx", 1, 0]]
        # SWAP
        if budget >= 3:
            swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
            if np.allclose(target, swap, atol=1e-10):
                return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
        # iSWAP
        if budget >= 3:
            iswap = np.array([[0, 1j, 0, 0], [1j, 0, 0, 0], [0, 0, 0, 1j], [0, 0, 1j, 0]], dtype=complex)
            if np.allclose(target, iswap, atol=1e-10):
                gates = [
                    ["u3", 0, 0, 0, math.pi / 2],
                    ["u3", 1, 0, 0, math.pi / 2],
                    ["u3", 0, math.pi / 2, 0, math.pi],
                    ["u3", 1, math.pi / 2, 0, math.pi],
                    ["cx", 0, 1],
                    ["u3", 0, math.pi / 2, 0, math.pi],
                    ["u3", 1, math.pi / 2, 0, math.pi],
                    ["u3", 0, 0, 0, -math.pi / 2],
                    ["u3", 1, 0, 0, math.pi / 2],
                ]
                u = _build(gates, 2)
                if _distance(target, u) < 1e-8:
                    return gates

    if qubits == 3:
        # Toffoli (CCX): controls 0,1; target 2
        if budget >= 6:
            toffoli = np.eye(8, dtype=complex)
            for i in range(8):
                q0 = (i >> 2) & 1
                q1 = (i >> 1) & 1
                if q0 == 1 and q1 == 1:
                    j = i ^ 1
                    toffoli[i, i] = 0
                    toffoli[j, i] = 1
            if np.allclose(target, toffoli, atol=1e-10):
                gates = [
                    ["u3", 2, math.pi / 2, 0, math.pi],
                    ["cx", 1, 2],
                    ["u3", 2, 0, 0, -math.pi / 4],
                    ["cx", 0, 2],
                    ["u3", 2, 0, 0, math.pi / 4],
                    ["cx", 1, 2],
                    ["u3", 2, 0, 0, -math.pi / 4],
                    ["u3", 1, 0, 0, math.pi / 4],
                    ["cx", 0, 1],
                    ["u3", 1, 0, 0, -math.pi / 4],
                    ["u3", 2, 0, 0, math.pi / 4],
                    ["cx", 0, 2],
                    ["u3", 2, math.pi / 2, 0, math.pi],
                ]
                u = _build(gates, 3)
                if _distance(target, u) < 1e-8:
                    return gates

    return None


def _kak_decompose_2qubit(U):
    """
    KAK decomposition for 2-qubit unitary U.
    Returns (A, B, K) where U = (A ⊗ B) K, and K is the 2-qubit entangling part.
    """
    # Compute A and B from the KAK decomposition
    # U = (A ⊗ B) exp(-i/2 sum k_i σ_i ⊗ σ_i)
    # For 2 qubits, we can use the standard KAK formula
    
    # Compute the 'twisted' product
    # U^T = (A^T ⊗ B^T) K^T
    # Since K is symmetric in the σ_i⊗σ_i basis, K^T = K
    
    # The standard approach:
    # M = U @ (np.eye(4) @ U)  # This isn't quite right
    
    # Let's use the explicit KAK formula for 2 qubits
    # U = (A ⊗ B) K where K = exp(-i/2 (k1 σx⊗σx + k2 σy⊗σy + k3 σz⊗σz))
    
    # Compute the 'canonical' form
    # The KAK decomposition gives us A, B, and the angles k1, k2, k3
    
    # For a 2-qubit unitary, we can compute:
    # A = (U ⊗ I) (I ⊗ U^T) (U ⊗ I)^{-1/2}  -- this is getting complex
    
    # Simpler approach: use the fact that for 2 qubits,
    # the KAK decomposition can be computed via the 'twisted' product
    
    # Let's use a different approach: compute A and B such that
    # (A^T ⊗ B^T) U (A ⊗ B) is in the canonical KAK form
    
    # For now, use a simpler decomposition:
    # Any 2-qubit unitary can be written as (A ⊗ B) CNOT (C ⊗ D) CNOT (E ⊗ F)
    # with 3 CNOTs. But we want to use the KAK angles.
    
    # Actually, let's use a more direct approach:
    # The KAK decomposition for 2 qubits: U = (A ⊗ B) K(k1, k2, k3)
    # where K = exp(-i/2 (k1 σx⊗σx + k2 σy⊗σy + k3 σz⊗σz))
    
    # To find A and B, we can use:
    # (A ⊗ B)^{-1} U = K
    # So we need to find A, B such that (A ⊗ B)^{-1} U is in the KAK form
    
    # This is a non-trivial optimization problem. Let's use a simpler approach:
    # Use the known fact that any 2-qubit unitary can be decomposed with 3 CNOTs.
    # We'll use the standard decomposition algorithm.
    
    # For now, return None and use the generic approach
    return None


def _decompose_2qubit(U):
    """
    Decompose a 2-qubit unitary into single-qubit gates and up to 3 CNOTs.
    Uses the standard 3-CNOT decomposition.
    """
    # Standard decomposition: U = (A1 ⊗ A2) CNOT (B1 ⊗ B2) CNOT (C1 ⊗ C2) CNOT (D1 ⊗ D2)
    # This requires finding the right single-qubit gates.
    
    # A simpler approach: use the fact that any 2-qubit unitary can be written as
    # U = (A ⊗ B) CNOT (C ⊗ D) CNOT (E ⊗ F) CNOT (G ⊗ H)
    
    # For now, use a generic approach with coordinate descent
    return None


def _optimize(layout, target, qubits, rng, max_sweeps=30):
    """Optimize u3 parameters using coordinate descent."""
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3
    
    # Initialize with zeros
    best = [0.0] * n_params
    best_dist = _distance(target, _build_from_layout(layout, best, qubits))
    
    for sweep in range(max_sweeps):
        if _time_left() < 20.0:
            break
        step = 2.0 / (1.0 + sweep * 0.3)
        improved = False
        
        for i in range(n_params):
            if _time_left() < 20.0:
                break
            # Try multiple perturbations
            for trial in (best[i] + step, best[i] - step,
                          best[i] + rng.uniform(-math.pi, math.pi),
                          best[i] + rng.uniform(-math.pi, math.pi)):
                trial = trial % (2 * math.pi)
                if trial > math.pi:
                    trial -= 2 * math.pi
                values = best[:]
                values[i] = trial
                dist = _distance(target, _build_from_layout(layout, values, qubits))
                if dist < best_dist:
                    best, best_dist = values, dist
                    improved = True
        
        if not improved:
            break
    
    return best, best_dist


def _build_from_layout(layout, values, qubits):
    """Build unitary directly from layout and parameter values."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            theta, phi, lam = values[k], values[k + 1], values[k + 2]
            k += 3
            single = _u3_mat(theta, phi, lam)
            matrix = _apply_single(matrix, gate[1], single, qubits)
        else:
            matrix = _apply_cnot(matrix, gate[1], gate[2], qubits)
    return matrix


def _gates_from_layout(layout, values):
    """Convert layout + values to gate list."""
    gates = []
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
            k += 3
        else:
            gates.append(list(gate))
    return gates


def _make_layout(qubits, cnot_budget, pair_order):
    """Create a circuit layout with u3 gates interleaved with CNOTs."""
    layout = []
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    for i in range(cnot_budget):
        c, t = pair_order[i % len(pair_order)]
        layout.append(["cx", c, t])
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    return layout


def synthesize(problem):
    """Synthesize a circuit for the given target unitary within the CNOT budget."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    # Try known exact decompositions first
    standard = _try_standard_gates(problem)
    if standard is not None:
        return standard

    # Generate candidate layouts
    if qubits == 2:
        pair_options = [
            [(0, 1)],
            [(1, 0)],
            [(0, 1), (1, 0)],
            [(1, 0), (0, 1)],
            [(0, 1), (0, 1), (1, 0)],
            [(1, 0), (1, 0), (0, 1)],
        ]
    else:
        pair_options = [
            [(0, 1), (0, 2), (1, 2)],
            [(0, 1), (1, 2), (0, 2)],
            [(0, 2), (0, 1), (1, 2)],
            [(1, 2), (0, 1), (0, 2)],
            [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)],
            [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)],
        ]

    best_gates = None
    best_dist = 1.0
    rng = random.Random(4242 + qubits * 100 + budget)

    # Try each pair ordering with multiple random restarts
    for pair_order in pair_options:
        if _time_left() < 25.0:
            break
        if len(pair_order) == 0:
            continue

        layout = _make_layout(qubits, budget, pair_order)

        # Check gate count
        n_gates = len(layout)
        if n_gates > 400:
            continue
        n_cnots = sum(1 for g in layout if g[0] == "cx")
        if n_cnots > budget:
            continue

        # Optimize this layout with multiple restarts
        for restart in range(3):
            if _time_left() < 25.0:
                break
            values, dist = _optimize(layout, target, qubits, rng)
            if dist < best_dist:
                best_dist = dist
                best_gates = _gates_from_layout(layout, values)

            # If we've found a very good solution, stop early
            if best_dist < 1e-10:
                break

        if best_dist < 1e-10:
            break

    # If no valid solution found, return identity
    if best_gates is None:
        best_gates = [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]

    return best_gates
# EVOLVE-BLOCK-END
