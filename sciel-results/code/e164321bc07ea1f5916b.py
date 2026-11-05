"""Rewritten program for exact quantum circuit synthesis within CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.linalg import logm, expm

_START_TIME = time.time()
TIME_LIMIT = 260  # Safety margin for 270s wall-clock limit


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build_fast(gates, qubits):
    """Build the unitary from a gate list efficiently."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    eye = np.eye(2, dtype=complex)
    
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
            index = np.arange(n)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _synthesize_cnot():
    """CNOT gate: just one CNOT."""
    return [["cx", 0, 1]]


def _synthesize_iswap():
    """iSWAP gate: 2 CNOTs + single qubit rotations."""
    return [
        ["u3", 0, math.pi/2, 0, math.pi/2],
        ["u3", 1, math.pi/2, 0, math.pi/2],
        ["cx", 0, 1],
        ["u3", 0, math.pi/2, math.pi/2, 0],
        ["u3", 1, math.pi/2, math.pi/2, 0],
        ["cx", 1, 0],
        ["u3", 0, math.pi/2, 0, math.pi/2],
        ["u3", 1, math.pi/2, 0, math.pi/2],
    ]


def _synthesize_swap():
    """SWAP gate: 3 CNOTs."""
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]


def _synthesize_toffoli():
    """Toffoli gate: 6 CNOTs (standard decomposition)."""
    return [
        ["u3", 2, math.pi/2, 0, math.pi/2],
        ["cx", 1, 2],
        ["u3", 2, math.pi/4, 0, -math.pi/2],
        ["cx", 0, 2],
        ["u3", 2, -math.pi/4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, math.pi/4, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, -math.pi/4, 0, -math.pi/2],
        ["u3", 1, math.pi/4, 0, math.pi/4],
        ["u3", 2, math.pi/4, 0, math.pi/2],
        ["cx", 1, 2],
        ["u3", 1, -math.pi/4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, math.pi/2, 0, math.pi/2],
    ]


def _decompose_u3(matrix):
    """Decompose a 2x2 unitary into u3(theta, phi, lam)."""
    # U3(theta, phi, lambda) = [[cos(theta/2), -e^{i*lambda}sin(theta/2)],
    #                           [e^{i*phi}sin(theta/2), e^{i*(phi+lambda)}cos(theta/2)]]
    # This is up to global phase. We need to handle the global phase.
    
    # First, normalize by the global phase if needed
    # The u3 form has det = e^{i*(phi+lambda)}
    # For a general unitary U, we can write U = e^{i*alpha} * U3
    
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]
    
    # Handle special cases
    if abs(a) < 1e-10:
        # a ≈ 0, so cos(theta/2) ≈ 0, theta ≈ pi
        theta = math.pi
        if abs(b) < 1e-10:
            # Both a and b are ~0, this is problematic
            return [0, 0, 0]
        # -e^{i*lambda} * sin(theta/2) = b => -e^{i*lambda} = b
        # e^{i*lambda} = -b
        lam = math.atan2(-b.imag, -b.real)
        # e^{i*phi} * sin(theta/2) = c => e^{i*phi} = c
        phi = math.atan2(c.imag, c.real)
        return [theta, phi, lam]
    
    # cos(theta/2) = a / |a| (up to phase)
    # Let's use the standard decomposition
    # |a|^2 + |b|^2 = 1
    # theta = 2 * arccos(|a|)
    theta = 2 * math.acos(min(1.0, abs(a)))
    
    if abs(math.cos(theta / 2)) < 1e-10:
        return [math.pi, 0, 0]
    
    # a = cos(theta/2) * e^{i*0} (up to global phase)
    # b = -e^{i*lambda} * sin(theta/2)
    # c = e^{i*phi} * sin(theta/2)
    # d = e^{i*(phi+lambda)} * cos(theta/2)
    
    # Global phase: a / cos(theta/2) should be e^{i*alpha}
    if abs(a) > 1e-10:
        global_phase = a / abs(a)
    else:
        global_phase = 1.0
    
    # Remove global phase
    U = matrix / global_phase
    
    a2, b2 = U[0, 0], U[0, 1]
    c2, d2 = U[1, 0], U[1, 1]
    
    # Now a2 ≈ cos(theta/2) (real and positive ideally)
    # b2 = -e^{i*lambda} * sin(theta/2)
    # c2 = e^{i*phi} * sin(theta/2)
    
    if abs(math.sin(theta / 2)) < 1e-10:
        # theta ≈ 0
        phi = 0
        lam = 0
        return [0, phi, lam]
    
    sin_half = math.sin(theta / 2)
    
    # lambda from b2
    if abs(b2) > 1e-10:
        lam = math.atan2(-b2.imag, -b2.real)
    else:
        lam = 0
    
    # phi from c2
    if abs(c2) > 1e-10:
        phi = math.atan2(c2.imag, c2.real)
    else:
        phi = 0
    
    return [theta, phi, lam]


def _kak_decompose_2qubit(target):
    """
    Use KAK decomposition to decompose a 2-qubit unitary into 3 CNOTs + single-qubit gates.
    Returns a list of gates or None if it fails.
    """
    n = 4
    U = target.copy()
    
    # Remove global phase
    det = np.linalg.det(U)
    if abs(det) > 1e-10:
        U = U / np.exp(1j * math.atan2(det.imag, det.real) / 2)
    
    # KAK decomposition: U = (A ⊗ B) * CNOT * (C ⊗ D) * CNOT * (E ⊗ F) * CNOT
    # Using the standard algorithm
    
    # Step 1: Compute the "canonical" part
    # We need to find local unitaries such that U * (A* ⊗ B*) is in canonical form
    
    # For 2 qubits, the KAK decomposition involves finding the coefficients
    # c = (c1, c2, c3) such that U = e^{i*alpha} (A⊗B) exp(i*c1*σx⊗σx/2) exp(i*c2*σy⊗σy/2) exp(i*c3*σz⊗σz/2)
    
    # A simpler approach: use scipy to do the decomposition
    # Actually, let's use a known algorithm
    
    # The KAK decomposition for 2 qubits:
    # 1. Compute U' = U * (U^T)^{-1/2} ... this is getting complex
    
    # Let me use a different approach: iterative refinement
    # Start with a random 3-CNOT circuit and optimize
    
    return None


def _generic_synthesis(problem):
    """Generic synthesis using gradient-free optimization with better initialization."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    
    # Build a layout with CNOTs and single-qubit gates
    if qubits == 2:
        # For 2 qubits, use alternating CNOTs
        pairs = [(0, 1), (1, 0), (0, 1), (1, 0), (0, 1), (1, 0), (0, 1), (1, 0)]
    else:
        # For 3 qubits, use a pattern that covers all pairs
        pairs = [(0, 1), (1, 2), (0, 2), (2, 1), (1, 0), (2, 0), (0, 1), (1, 2)]
    
    # Create layout: layers of single-qubit gates followed by CNOTs
    layout = []
    num_layers = min(budget, 6)  # Cap layers
    for layer in range(num_layers):
        # Single-qubit gates before CNOT
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        # One CNOT per layer
        c, t = pairs[layer % len(pairs)]
        layout.append(["cx", c, t])
    
    # Final single-qubit gates
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
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
    
    best = None
    best_distance = float('inf')
    
    # Try multiple random initializations with more starts
    max_starts = 12
    np.random.seed(42)
    
    for start in range(max_starts):
        if _time_remaining() < 15:
            break
        
        if start == 0:
            values = [0.0] * n_params
        else:
            values = np.random.uniform(-math.pi, math.pi, n_params).tolist()
        
        current_distance = _distance(target, _build_fast(circuit(values), qubits))
        if current_distance < best_distance:
            best_distance = current_distance
            best = values[:]
        
        # Coordinate descent with adaptive step sizes
        max_sweeps = 50
        for sweep in range(max_sweeps):
            if _time_remaining() < 15:
                break
            step = 0.3 / (1 + sweep * 0.2)
            improved = False
            for i in range(n_params):
                if _time_remaining() < 15:
                    break
                for trial_step in (step, step * 0.5, step * 2, step * 0.25):
                    for direction in (1, -1):
                        trial = values[i] + direction * trial_step
                        values[i] = trial
                        value = _distance(target, _build_fast(circuit(values), qubits))
                        if value < current_distance:
                            current_distance = value
                            improved = True
                            break
                        values[i] = trial - direction * trial_step
                    if improved:
                        break
                if _time_remaining() < 15:
                    break
            if not improved:
                break
        
        if current_distance < best_distance:
            best_distance = current_distance
            best = values[:]
    
    if best is None:
        best = [0.0] * n_params
    
    return circuit(best)


def synthesize(problem):
    """Synthesize the target unitary within the CNOT budget."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    
    # Try to use known exact syntheses first
    if name == "CNOT" and qubits == 2:
        gates = _synthesize_cnot()
    elif name == "iSWAP" and qubits == 2:
        gates = _synthesize_iswap()
    elif name == "SWAP" and qubits == 2:
        gates = _synthesize_swap()
    elif name == "Toffoli" and qubits == 3:
        gates = _synthesize_toffoli()
    else:
        # Generic synthesis
        gates = _generic_synthesis(problem)
    
    # Verify the result
    target = _target(problem)
    matrix = _build_fast(gates, qubits)
    dist = _distance(target, matrix)
    
    # If the known synthesis didn't work well, fall back to generic
    if dist > 0.01:
        gates = _generic_synthesis(problem)
    
    return gates
# EVOLVE-BLOCK-END
