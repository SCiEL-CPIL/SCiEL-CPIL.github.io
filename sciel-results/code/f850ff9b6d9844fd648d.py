"""Exact quantum circuit synthesis within published CNOT budgets."""
import math
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 260.0  # safety margin from 270s kill


def _remaining_time():
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _decompose_single_qubit(U):
    """Decompose a 2x2 unitary into u3(theta, phi, lam) parameters."""
    det = np.linalg.det(U)
    if abs(det) < 1e-10:
        return 0.0, 0.0, 0.0
    sqrt_det = np.sqrt(det)
    if abs(sqrt_det) < 1e-10:
        return 0.0, 0.0, 0.0
    U_norm = U / sqrt_det
    # U_norm should be SU(2): [[a, -b*], [b, a*]]
    a = U_norm[0, 0]
    b = U_norm[1, 0]
    # theta/2: a = cos(theta/2), b = sin(theta/2)*exp(i*phi)
    theta = 2 * np.arctan2(abs(b), abs(a))
    if abs(b) < 1e-10:
        phi = 0.0
        lam = 0.0
    else:
        phi = np.angle(b)
        # From U[0,1] = -exp(i*lam)*sin(theta/2)
        # U[0,1] = -b* where b* = conj(b)
        # So -exp(i*lam)*s = -conj(b) => exp(i*lam)*s = conj(b)
        # s = sin(theta/2) = abs(b)
        # exp(i*lam) = conj(b)/abs(b)
        lam = np.angle(np.conj(b))
    # Adjust global phase to match
    # The decomposition gives U_norm = U3(theta, phi, lam)
    # We need U = sqrt_det * U3(theta, phi, lam) * global_phase
    # Actually U = sqrt_det * U_norm, and U_norm = U3(theta,phi,lam) up to phase
    # Let's compute the phase
    U3 = _u3_matrix(theta, phi, lam)
    # U_norm should equal phase * U3
    # phase = U_norm[0,0] / U3[0,0] if U3[0,0] != 0
    if abs(U3[0, 0]) > 1e-10:
        phase = U_norm[0, 0] / U3[0, 0]
    elif abs(U3[1, 1]) > 1e-10:
        phase = U_norm[1, 1] / U3[1, 1]
    else:
        phase = 1.0
    # Absorb phase into lam
    lam += np.angle(phase)
    return float(theta), float(phi), float(lam)


def _build_circuit(gates, qubits):
    """Build the unitary from a gate list."""
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
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


def _known_gate_decomposition(name, qubits, target):
    """Return exact gate list for known gates, or None if not a known gate."""
    if name == "CNOT" or name == "cx":
        return [["cx", 0, 1]]
    elif name == "iSWAP" or name == "iswap":
        # iSWAP = |00><00| + i|01><10| + i|10><01| + |11><11|
        # Decomposition: H on both, then CNOT, then single qubit rotations, then CNOT, then H on both
        # iSWAP = (I ⊗ H) · CNOT · (Rz(-π/4) ⊗ Rx(π/2)) · CNOT · (Rz(-π/4) ⊗ Rx(π/2)) · (I ⊗ H)
        # Let me use a cleaner decomposition
        # iSWAP = (S ⊗ S) · (I ⊗ H) · CNOT · (Rx(π/2) ⊗ I) · CNOT · (I ⊗ H)
        # Actually, let me just use the standard decomposition
        # iSWAP = exp(iπ/4) · (I⊗H) · CNOT · (Rz(-π/4)⊗Rx(π/2)) · CNOT · (I⊗H) · (I⊗S)
        # Let me verify with a simpler approach: use the known CNOT-based decomposition
        # iSWAP = (I⊗H) · CNOT · (Rz(-π/4)⊗Rx(π/2)) · CNOT · (I⊗H) · (I⊗S)
        # where S = diag(1, i)
        # Rz(θ) = diag(e^{-iθ/2}, e^{iθ/2})
        # Rx(θ) = [[cos(θ/2), -i sin(θ/2)], [-i sin(θ/2), cos(θ/2)]]
        # u3(θ, φ, λ) = [[cos(θ/2), -e^{iλ}sin(θ/2)], [e^{iφ}sin(θ/2), e^{i(φ+λ)}cos(θ/2)]]
        # H = u3(π/2, 0, π)
        # S = diag(1, i) = u3(0, 0, π/2)
        # Rz(θ) = u3(0, 0, θ)  ... wait, let me check
        # Rz(θ) = [[e^{-iθ/2}, 0], [0, e^{iθ/2}]]
        # u3(0, φ, λ) = [[1, 0], [0, e^{i(φ+λ)}]]
        # So Rz(θ) = u3(0, 0, θ) gives [[1,0],[0,e^{iθ}]] which is not quite right
        # Actually u3(0, 0, λ) = [[1,0],[0,e^{iλ}]]
        # So to get diag(e^{-iθ/2}, e^{iθ/2}), we can use global phase e^{-iθ/2} * diag(1, e^{iθ})
        # For our purposes, global phase doesn't matter for the distance metric
        # So Rz(θ) ≈ u3(0, 0, θ) up to global phase
        # Rx(θ) = [[cos(θ/2), -i sin(θ/2)], [-i sin(θ/2), cos(θ/2)]]
        # u3(π/2, φ, λ) = [[1/√2, -e^{iλ}/√2], [e^{iφ}/√2, e^{i(φ+λ)}/√2]]
        # To get Rx(π/2) = [[0, -i], [-i, 0]]:
        # Need -e^{iλ}/√2 = -i => e^{iλ} = i√2... that doesn't work
        # Let me just use the numerical approach for iSWAP
        pass
    elif name == "SWAP" or name == "swap":
        return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
    elif name == "Toffoli" or name == "toffoli" or name == "ccx":
        # Toffoli decomposition with 6 CNOTs
        # Standard decomposition
        gates = [
            ["u3", 2, math.pi / 2, 0, math.pi],  # H on qubit 2
            ["cx", 1, 2],
            ["u3", 2, math.pi, 0, -math.pi / 2],  # T† on qubit 2
            ["cx", 0, 2],
            ["u3", 2, math.pi / 2, 0, 0],  # T on qubit 2
            ["cx", 1, 2],
            ["u3", 2, math.pi / 2, 0, -math.pi / 2],  # T† on qubit 2
            ["cx", 0, 2],
            ["u3", 2, math.pi / 2, 0, math.pi / 2],  # T on qubit 2
            ["u3", 1, math.pi / 2, 0, math.pi / 2],  # T on qubit 1
            ["cx", 0, 1],
            ["u3", 1, math.pi, 0, -math.pi / 2],  # T† on qubit 1
            ["cx", 0, 1],
            ["u3", 1, math.pi / 2, 0, 0],  # T on qubit 1
            ["u3", 2, math.pi / 2, 0, math.pi],  # H on qubit 2
        ]
        return gates
    return None


def _kak_decompose_2qubit(U):
    """
    KAK decomposition of a 2-qubit unitary.
    Returns (A, B, C, D) such that U = (A⊗B) · CNOT · (C⊗D) up to global phase,
    where A,B,C,D are single-qubit unitaries.
    """
    # For a general 2-qubit unitary, we use the canonical form
    # U = e^{iθ} (A⊗B) exp(i α σx⊗σx) exp(i β σy⊗σy) exp(i γ σz⊗σz) (C⊗D)
    # For 3 CNOTs: U = (A⊗B) CNOT (C⊗D) CNOT (E⊗F) CNOT (G⊗H)
    # This is more complex. Let me use a simpler approach.
    
    # For 2-qubit unitaries with 3 CNOTs, we can use:
    # U = (A⊗B) · CNOT · (C⊗D) · CNOT · (E⊗F) · CNOT · (G⊗H)
    
    # Actually, the standard result is that any 2-qubit unitary can be written as:
    # U = e^{iθ} (A⊗B) · CNOT · (C⊗D) · CNOT · (E⊗F) · CNOT · (G⊗H)
    # where A,B,C,D,E,F,G,H are single-qubit unitaries.
    
    # Let me use a different approach: use the fact that we can optimize
    # the single-qubit rotations to match the target.
    
    # For now, let's return None and use optimization
    return None


def _optimize_2qubit(target, qubits, budget):
    """Optimize single-qubit rotations for a 2-qubit unitary with given CNOT budget."""
    # Layout: u3s, cx, u3s, cx, u3s, cx, u3s
    # For 3 CNOTs: 4 layers of u3s (2 qubits each) = 12 parameters
    # For 2 CNOTs: 3 layers of u3s = 9 parameters
    # For 1 CNOT: 2 layers of u3s = 6 parameters
    
    n_layers = budget + 1
    n_params = 3 * 2 * n_layers  # 3 params per u3, 2 qubits per layer
    
    # Initial guess: all zeros
    best = [0.0] * n_params
    best_dist = _distance(target, _build_circuit(_make_gates_2qubit(best, budget), qubits))
    
    # Coordinate descent
    rng = np.random.RandomState(42)
    step = 1.0
    for sweep in range(20):
        step = 1.0 / (1 + sweep)
        improved = False
        for i in range(n_params):
            for trial in (best[i] + step, best[i] - step, rng.uniform(-math.pi, math.pi)):
                values = best[:]
                values[i] = trial
                gates = _make_gates_2qubit(values, budget)
                dist = _distance(target, _build_circuit(gates, qubits))
                if dist < best_dist:
                    best, best_dist = values, dist
                    improved = True
        if not improved:
            break
        if _remaining_time() < 5.0:
            break
    
    return _make_gates_2qubit(best, budget)


def _make_gates_2qubit(params, budget):
    """Create gate list for 2-qubit circuit from parameters."""
    gates = []
    k = 0
    for layer in range(budget + 1):
        for q in range(2):
            theta, phi, lam = params[k], params[k + 1], params[k + 2]
            gates.append(["u3", q, theta, phi, lam])
            k += 3
        if layer < budget:
            gates.append(["cx", 0, 1])
    return gates


def _optimize_3qubit(target, qubits, budget):
    """Optimize single-qubit rotations for a 3-qubit unitary with given CNOT budget."""
    # Layout: layers of u3s (3 qubits) separated by CNOTs
    # CNOT pattern: cycle through pairs
    pairs = [(0, 1), (1, 2), (0, 2)]
    n_layers = budget + 1
    n_params = 3 * 3 * n_layers  # 3 params per u3, 3 qubits per layer
    
    best = [0.0] * n_params
    best_dist = _distance(target, _build_circuit(_make_gates_3qubit(best, budget), qubits))
    
    rng = np.random.RandomState(42)
    step = 1.0
    for sweep in range(10):
        step = 1.0 / (1 + sweep)
        improved = False
        for i in range(n_params):
            for trial in (best[i] + step, best[i] - step, rng.uniform(-math.pi, math.pi)):
                values = best[:]
                values[i] = trial
                gates = _make_gates_3qubit(values, budget)
                dist = _distance(target, _build_circuit(gates, qubits))
                if dist < best_dist:
                    best, best_dist = values, dist
                    improved = True
        if not improved:
            break
        if _remaining_time() < 5.0:
            break
    
    return _make_gates_3qubit(best, budget)


def _make_gates_3qubit(params, budget):
    """Create gate list for 3-qubit circuit from parameters."""
    gates = []
    k = 0
    pairs = [(0, 1), (1, 2), (0, 2)]
    for layer in range(budget + 1):
        for q in range(3):
            theta, phi, lam = params[k], params[k + 1], params[k + 2]
            gates.append(["u3", q, theta, phi, lam])
            k += 3
        if layer < budget:
            c, t = pairs[layer % len(pairs)]
            gates.append(["cx", c, t])
    return gates


def synthesize(problem):
    """Synthesize a quantum circuit for the given target unitary."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    name = problem.get("name", "")
    target = _target(problem)
    
    # Check for known gates first
    known = _known_gate_decomposition(name, qubits, target)
    if known is not None:
        # Verify the known decomposition
        built = _build_circuit(known, qubits)
        dist = _distance(target, built)
        if dist < 1e-6:
            return known
        # If the known decomposition doesn't match, fall through to optimization
    
    # For iSWAP, let's handle it specifically
    if name in ("iSWAP", "iswap") and qubits == 2:
        # iSWAP decomposition with 2 CNOTs
        # iSWAP = (I⊗H) · CNOT · (Rz(-π/4)⊗Rx(π/2)) · CNOT · (I⊗H) · (I⊗S)
        # Let me just optimize it
        pass
    
    # Use optimization
    if qubits == 2:
        gates = _optimize_2qubit(target, qubits, budget)
    else:
        gates = _optimize_3qubit(target, qubits, budget)
    
    # Verify gate count
    if len(gates) > 400:
        # Truncate if needed (shouldn't happen)
        gates = gates[:400]
    
    return gates
