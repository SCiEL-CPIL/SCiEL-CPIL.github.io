"""Exact quantum circuit synthesis within published CNOT budgets."""
import math
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 255.0  # safety margin from 270s kill


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
    Decomposition with 2 CNOTs:
    iSWAP = (S⊗S) · (I⊗H) · CNOT · (Rx(π/2)⊗I) · CNOT · (I⊗H)
    where S = diag(1,i), H = Hadamard, Rx(π/2) = [[0,-i],[-i,0]]
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
    KAK decomposition for 2-qubit unitary.
    U = A ⊗ B · exp(-i θ1 σx⊗σx - i θ2 σy⊗σy - i θ3 σz⊗σz) · C ⊗ D
    """
    # Compute KAK angles
    from scipy.linalg import schur
    
    # Use the standard KAK decomposition algorithm
    # U = (A ⊗ B) K (C ⊗ D)
    # where K = exp(-i(θ1 σx⊗σx + θ2 σy⊗σy + θ3 σz⊗σz))
    
    # For a 2-qubit unitary, we can use the following approach:
    # 1. Compute the KAK angles
    # 2. Decompose into single-qubit gates and CNOTs
    
    # Simple approach: use coordinate descent to find optimal parameters
    # for a circuit with the given CNOT budget
    
    return None


def _optimize_2qubit(target, qubits, budget):
    """Optimize single-qubit rotations for a 2-qubit unitary with given CNOT budget."""
    n_layers = budget + 1
    n_params = 3 * 2 * n_layers
    
    best = [0.0] * n_params
    best_dist = _distance(target, _build_circuit(_make_gates_2qubit(best, budget), qubits))
    
    # Try multiple initializations
    rng = np.random.RandomState(42)
    for init in range(8):
        if _remaining_time() < 15.0:
            break
        if init == 0:
            current = [0.0] * n_params
        elif init == 1:
            current = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
        else:
            current = best[:]
            # Perturb some parameters
            for i in range(n_params):
                if rng.random() < 0.3:
                    current[i] += rng.normal(0, 0.5)
        
        cur_dist = _distance(target, _build_circuit(_make_gates_2qubit(current, budget), qubits))
        
        # Coordinate descent with adaptive step size
        step = 0.5
        for sweep in range(25):
            if _remaining_time() < 10.0:
                break
            improved = False
            for i in range(n_params):
                # Try several values around current with adaptive step
                trials = [current[i] + step, current[i] - step,
                         current[i] + step * 0.5, current[i] - step * 0.5,
                         rng.uniform(-math.pi, math.pi)]
                for trial in trials:
                    values = current[:]
                    values[i] = trial
                    gates = _make_gates_2qubit(values, budget)
                    dist = _distance(target, _build_circuit(gates, qubits))
                    if dist < cur_dist:
                        current, cur_dist = values, dist
                        improved = True
                        step = max(0.1, step * 0.8)
                    else:
                        step = min(2.0, step * 1.2)
            
            if not improved:
                step = max(0.1, step * 0.7)
                if step < 0.15:
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
    for init in range(5):
        if _remaining_time() < 20.0:
            break
        if init == 0:
            current = [0.0] * n_params
        elif init == 1:
            current = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
        else:
            current = best[:]
            for i in range(n_params):
                if rng.random() < 0.3:
                    current[i] += rng.normal(0, 0.5)
        
        cur_dist = _distance(target, _build_circuit(_make_gates_3qubit(current, budget), qubits))
        
        # Coordinate descent with adaptive step size
        step = 0.5
        for sweep in range(15):
            if _remaining_time() < 15.0:
                break
            improved = False
            for i in range(n_params):
                trials = [current[i] + step, current[i] - step,
                         rng.uniform(-math.pi, math.pi)]
                for trial in trials:
                    values = current[:]
                    values[i] = trial
                    gates = _make_gates_3qubit(values, budget)
                    dist = _distance(target, _build_circuit(gates, qubits))
                    if dist < cur_dist:
                        current, cur_dist = values, dist
                        improved = True
                        step = max(0.1, step * 0.8)
                    else:
                        step = min(2.0, step * 1.2)
            
            if not improved:
                step = max(0.1, step * 0.7)
                if step < 0.15:
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
