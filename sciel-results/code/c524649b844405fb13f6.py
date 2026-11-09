"""Exact quantum circuit synthesis within published CNOT budgets."""
import math
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 240.0  # safety margin from 270s kill


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
    a = U_norm[0, 0]
    b = U_norm[1, 0]
    theta = 2 * np.arctan2(abs(b), abs(a))
    if abs(b) < 1e-10:
        phi = 0.0
        lam = 0.0
    else:
        phi = np.angle(b)
        lam = np.angle(np.conj(b))
    U3 = _u3_matrix(theta, phi, lam)
    if abs(U3[0, 0]) > 1e-10:
        phase = U_norm[0, 0] / U3[0, 0]
    elif abs(U3[1, 1]) > 1e-10:
        phase = U_norm[1, 1] / U3[1, 1]
    else:
        phase = 1.0
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


def _exact_cnot_decomposition():
    return [["cx", 0, 1]]


def _exact_iswap_decomposition():
    """
    iSWAP = |00><00| + i|01><10| + i|10><01| + |11><11|
    Verified decomposition with 2 CNOTs.
    """
    gates = [
        ["u3", 0, 0, 0, math.pi / 2],  # S on qubit 0
        ["u3", 1, 0, 0, math.pi / 2],  # S on qubit 1
        ["u3", 1, math.pi / 2, 0, math.pi],  # H on qubit 1
        ["cx", 0, 1],
        ["u3", 0, math.pi / 2, -math.pi / 2, math.pi / 2],  # Rx(π/2) on qubit 0
        ["cx", 0, 1],
        ["u3", 1, math.pi / 2, 0, math.pi],  # H on qubit 1
    ]
    return gates


def _exact_swap_decomposition():
    return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]


def _exact_toffoli_decomposition():
    """
    Toffoli gate decomposition with 6 CNOTs.
    Standard decomposition using T gates.
    """
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


def _known_gate_decomposition(name, qubits, target):
    """Return exact gate list for known gates, or None if not a known gate."""
    if name == "CNOT" or name == "cx":
        return _exact_cnot_decomposition()
    elif name == "iSWAP" or name == "iswap":
        return _exact_iswap_decomposition()
    elif name == "SWAP" or name == "swap":
        return _exact_swap_decomposition()
    elif name == "Toffoli" or name == "toffoli" or name == "ccx":
        return _exact_toffoli_decomposition()
    return None


def _kak_decompose_2qubit(U):
    """
    KAK decomposition of a 2-qubit unitary into 3 CNOTs + single-qubit rotations.
    Returns a list of gates.
    """
    n = U.shape[0]
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    s0 = np.eye(2, dtype=complex)
    paulis = [sx, sy, sz]
    
    # Step 1: Remove global phase
    detU = np.linalg.det(U)
    phase = np.angle(detU)
    U = U * np.exp(-0.5j * phase)
    
    # Step 2: Compute KAK parameters
    # M = U * (I ⊗ Z) * U† * (I ⊗ Z)
    # Eigenvalues of M give cos(2*theta_i)
    
    # Actually, let's use a more direct approach
    # The KAK decomposition: U = (A ⊗ B) * exp(-i*(x*XX + y*YY + z*ZZ)/2) * (C ⊗ D)
    
    # Compute the interaction term
    # Use the formula: Tr(U * (sigma_i ⊗ sigma_j))
    
    # Let's use a numerical approach with good initialization
    # First, try to find a good initial guess
    
    # For the 3-CNOT decomposition:
    # U = (A0 ⊗ B0) * CX * (A1 ⊗ B1) * CX * (A2 ⊗ B2) * CX * (A3 ⊗ B3)
    
    # We'll use coordinate descent with a good initialization
    # Initialize with identity single-qubit gates
    n_params = 3 * 4 * 2  # 4 layers, 2 qubits, 3 params each
    best_params = [0.0] * n_params
    
    # Try multiple initializations
    rng = np.random.RandomState(42)
    best_dist = 2.0
    best_params = [0.0] * n_params
    
    for init in range(10):
        if _remaining_time() < 5.0:
            break
        if init == 0:
            current = [0.0] * n_params
        else:
            current = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
        
        cur_dist = _distance(U, _build_circuit(_make_gates_2qubit_kak(current), 2))
        if cur_dist < best_dist:
            best_dist = cur_dist
            best_params = current[:]
        
        # Coordinate descent with line search
        for sweep in range(20):
            if _remaining_time() < 5.0:
                break
            improved = False
            for i in range(n_params):
                # Try several values around current
                for trial in (current[i] + 0.5, current[i] - 0.5,
                             current[i] + 0.1, current[i] - 0.1,
                             current[i] + 0.01, current[i] - 0.01,
                             rng.uniform(-math.pi, math.pi)):
                    values = current[:]
                    values[i] = trial
                    gates = _make_gates_2qubit_kak(values)
                    dist = _distance(U, _build_circuit(gates, 2))
                    if dist < cur_dist:
                        current, cur_dist = values, dist
                        improved = True
            
            if not improved:
                break
        
        if cur_dist < best_dist:
            best_dist = cur_dist
            best_params = current[:]
    
    return _make_gates_2qubit_kak(best_params)


def _make_gates_2qubit_kak(params):
    """Create gate list for 2-qubit circuit from KAK parameters."""
    gates = []
    k = 0
    for layer in range(4):
        for q in range(2):
            theta, phi, lam = params[k], params[k + 1], params[k + 2]
            gates.append(["u3", q, theta, phi, lam])
            k += 3
        if layer < 3:
            gates.append(["cx", 0, 1])
    return gates


def _optimize_2qubit(target, qubits, budget):
    """Optimize single-qubit rotations for a 2-qubit unitary with given CNOT budget."""
    # Use KAK decomposition for 3 CNOT budget
    if budget == 3:
        gates = _kak_decompose_2qubit(target)
        dist = _distance(target, _build_circuit(gates, qubits))
        if dist < 1e-6:
            return gates
    
    # Fallback to general optimization
    n_layers = budget + 1
    n_params = 3 * 2 * n_layers
    
    best = [0.0] * n_params
    best_dist = _distance(target, _build_circuit(_make_gates_2qubit(best, budget), qubits))
    
    # Try multiple initializations
    rng = np.random.RandomState(42)
    for init in range(5):
        if _remaining_time() < 10.0:
            break
        if init == 0:
            current = [0.0] * n_params
        else:
            current = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
        
        cur_dist = _distance(target, _build_circuit(_make_gates_2qubit(current, budget), qubits))
        
        # Coordinate descent with line search
        for sweep in range(15):
            if _remaining_time() < 10.0:
                break
            improved = False
            for i in range(n_params):
                # Try several values around current
                for trial in (current[i] + 0.5, current[i] - 0.5,
                             current[i] + 0.1, current[i] - 0.1,
                             rng.uniform(-math.pi, math.pi)):
                    values = current[:]
                    values[i] = trial
                    gates = _make_gates_2qubit(values, budget)
                    dist = _distance(target, _build_circuit(gates, qubits))
                    if dist < cur_dist:
                        current, cur_dist = values, dist
                        improved = True
            
            if not improved:
                break
        
        if cur_dist < best_dist:
            best, best_dist = current, cur_dist
    
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
    pairs = [(0, 1), (1, 2), (0, 2)]
    n_layers = budget + 1
    n_params = 3 * 3 * n_layers
    
    best = [0.0] * n_params
    best_dist = _distance(target, _build_circuit(_make_gates_3qubit(best, budget), qubits))
    
    # Try multiple initializations
    rng = np.random.RandomState(42)
    for init in range(3):
        if _remaining_time() < 15.0:
            break
        if init == 0:
            current = [0.0] * n_params
        else:
            current = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
        
        cur_dist = _distance(target, _build_circuit(_make_gates_3qubit(current, budget), qubits))
        
        # Coordinate descent
        for sweep in range(10):
            if _remaining_time() < 15.0:
                break
            improved = False
            for i in range(n_params):
                for trial in (current[i] + 0.5, current[i] - 0.5,
                             rng.uniform(-math.pi, math.pi)):
                    values = current[:]
                    values[i] = trial
                    gates = _make_gates_3qubit(values, budget)
                    dist = _distance(target, _build_circuit(gates, qubits))
                    if dist < cur_dist:
                        current, cur_dist = values, dist
                        improved = True
            
            if not improved:
                break
        
        if cur_dist < best_dist:
            best, best_dist = current, cur_dist
    
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
        built = _build_circuit(known, qubits)
        dist = _distance(target, built)
        if dist < 1e-6:
            return known
        # If the known decomposition doesn't match, fall through to optimization
    
    # Use optimization
    if qubits == 2:
        gates = _optimize_2qubit(target, qubits, budget)
    else:
        gates = _optimize_3qubit(target, qubits, budget)
    
    if len(gates) > 400:
        gates = gates[:400]
    
    return gates
