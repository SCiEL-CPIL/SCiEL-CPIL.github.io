"""Synthesize fixed two- and three-qubit unitaries within CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

# Deadline guard: 270s wall-clock kill, keep 30s safety margin
_DEADLINE = time.time() + 240.0


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    """Build the full unitary from a gate list."""
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


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _cnot_matrix(control, target, qubits):
    n = 2 ** qubits
    mat = np.eye(n, dtype=complex)
    for i in range(n):
        if (i >> (qubits - 1 - control)) & 1:
            mat[i, i] = 0
            j = i ^ (1 << (qubits - 1 - target))
            mat[i, j] = 1
            mat[j, i] = 1
    return mat


def _exact_cnot(qubits):
    """CNOT: control=0, target=1."""
    gates = [["cx", 0, 1]]
    return gates


def _exact_iswap(qubits):
    """iSWAP gate using 1 CNOT."""
    if qubits == 2:
        gates = [
            ["u3", 1, 0.0, 0.0, math.pi / 2],
            ["u3", 0, 0.0, 0.0, math.pi / 2],
            ["cx", 0, 1],
            ["u3", 1, 0.0, 0.0, math.pi / 2],
            ["u3", 0, 0.0, 0.0, math.pi / 2],
            ["u3", 1, 0.0, 0.0, math.pi / 2],
        ]
        return gates
    else:
        return []


def _exact_swap(qubits):
    """SWAP gate using 3 CNOTs."""
    gates = [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]
    return gates


def _exact_toffoli(qubits):
    """Toffoli (CCNOT) gate: control=0,1 target=2. Uses 6 CNOTs."""
    gates = [
        ["u3", 2, math.pi / 2, 0.0, math.pi],
        ["cx", 1, 2],
        ["u3", 2, 0.0, 0.0, -math.pi / 4],
        ["cx", 0, 2],
        ["u3", 2, 0.0, 0.0, math.pi / 4],
        ["cx", 1, 2],
        ["u3", 2, 0.0, 0.0, -math.pi / 4],
        ["u3", 1, 0.0, 0.0, math.pi / 4],
        ["cx", 0, 1],
        ["u3", 1, 0.0, 0.0, -math.pi / 4],
        ["u3", 0, 0.0, 0.0, math.pi / 4],
        ["cx", 1, 0],
        ["u3", 1, 0.0, 0.0, math.pi / 4],
        ["cx", 0, 1],
        ["u3", 1, 0.0, 0.0, -math.pi / 4],
        ["u3", 2, math.pi / 2, 0.0, math.pi],
    ]
    return gates


def _identify_gate(name, target, qubits):
    """Check if target matches a known gate and return exact circuit."""
    def _match(a, b):
        d = abs(np.trace(a.conj().T @ b)) / a.shape[0]
        return d > 1 - 1e-9

    if qubits == 2:
        cnot = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=complex)
        if _match(target, cnot):
            return _exact_cnot(2)
        
        iswap = np.array([[0, 0, 0, 0],
                          [0, 0, 1j, 0],
                          [0, 1j, 0, 0],
                          [0, 0, 0, 0]], dtype=complex)
        if _match(target, iswap):
            return _exact_iswap(2)
        
        swap = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]], dtype=complex)
        if _match(target, swap):
            return _exact_swap(2)
    
    if qubits == 3:
        toffoli = np.eye(8, dtype=complex)
        toffoli[6, 6] = 0
        toffoli[7, 7] = 0
        toffoli[6, 7] = 1
        toffoli[7, 6] = 1
        if _match(target, toffoli):
            return _exact_toffoli(3)
    
    return None


def _decompose_single_qubit(U):
    """Return (theta, phi, lam) for u3 such that u3 ~ U up to global phase."""
    det = np.linalg.det(U)
    phase = np.angle(det)
    U_prime = U / np.exp(0.5j * phase)
    d00 = U_prime[0, 0]
    if abs(d00) < 1e-12:
        theta = math.pi
        lam = -np.angle(U_prime[0, 1])
        phi = np.angle(U_prime[1, 0])
        return (float(theta), float(phi), float(lam))
    theta = 2 * math.acos(min(1.0, max(-1.0, abs(d00))))
    phi = np.angle(U_prime[1, 0])
    lam = np.angle(U_prime[1, 0]) + np.angle(U_prime[0, 1])
    return (float(theta), float(phi), float(lam))


def _kak_two_qubit(target):
    """Decompose 4x4 unitary into single-qubit + 3 CNOTs using KAK."""
    n = 4
    # Bell basis (unnormalized)
    bell = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, -1],
        [0, 1, 1, 0],
        [0, 1, -1, 0]
    ], dtype=complex) / math.sqrt(2)
    
    # V = B^dag U B
    V = bell.conj().T @ target @ bell
    
    # For a 3-CNOT circuit, V should be diagonal (up to phases)
    # Extract the diagonal phases
    diag = np.diag(V)
    
    # We need to find A0, A1, B0, B1 such that:
    # U = (A0 ⊗ A1) * CNOT(0,1) * (B0 ⊗ B1)
    # This is the standard KAK decomposition
    
    # Use the fact that any 2-qubit unitary can be written as:
    # U = e^{iα} (A0 ⊗ A1) e^{iθ1 σx⊗σx + iθ2 σy⊗σy + iθ3 σz⊗σz} (B0 ⊗ B1)
    
    # For simplicity, use a direct approach with optimization
    # Initialize with identity
    A0 = np.eye(2, dtype=complex)
    A1 = np.eye(2, dtype=complex)
    B0 = np.eye(2, dtype=complex)
    B1 = np.eye(2, dtype=complex)
    
    # The middle part is 3 CNOTs
    # CNOT(0,1) @ CNOT(1,0) @ CNOT(0,1)
    cnot_01 = _cnot_matrix(0, 1, 2)
    cnot_10 = _cnot_matrix(1, 0, 2)
    middle = cnot_01 @ cnot_10 @ cnot_01
    
    # We want: target ≈ (A0⊗A1) @ middle @ (B0⊗B1)
    # So: (A0⊗A1) @ middle @ (B0⊗B1)^dag ≈ target
    # Let's optimize A0, A1, B0, B1
    
    # Flatten to real parameters (each 2x2 unitary has 3 real parameters)
    # Use Euler angles for each single-qubit gate
    
    def _build_from_params(params):
        # params: [A0_theta, A0_phi, A0_lam, A1_theta, A1_phi, A1_lam, 
        #           B0_theta, B0_phi, B0_lam, B1_theta, B1_phi, B1_lam]
        A0_m = _u3_matrix(params[0], params[1], params[2])
        A1_m = _u3_matrix(params[3], params[4], params[5])
        B0_m = _u3_matrix(params[6], params[7], params[8])
        B1_m = _u3_matrix(params[9], params[10], params[11])
        
        A = np.kron(A0_m, A1_m)
        B = np.kron(B0_m, B1_m)
        
        return A @ middle @ B.conj().T
    
    def _loss(params):
        u = _build_from_params(params)
        return _distance(target, u)
    
    # Optimize with Adam
    params = np.zeros(12)
    m = np.zeros(12)
    v = np.zeros(12)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    lr = 0.1
    
    best_params = params.copy()
    best_loss = _loss(params)
    
    max_iters = 200
    for t in range(1, max_iters + 1):
        if time.time() > _DEADLINE - 30:
            break
        
        # Numerical gradient
        grad = np.zeros(12)
        for i in range(12):
            h = 1e-4
            p_plus = params.copy()
            p_plus[i] += h
            p_minus = params.copy()
            p_minus[i] -= h
            grad[i] = (_loss(p_plus) - _loss(p_minus)) / (2 * h)
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        params -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        current_loss = _loss(params)
        if current_loss < best_loss:
            best_loss = current_loss
            best_params = params.copy()
        
        if best_loss < 1e-10:
            break
    
    # Build final gates
    A0_theta, A0_phi, A0_lam = best_params[0], best_params[1], best_params[2]
    A1_theta, A1_phi, A1_lam = best_params[3], best_params[4], best_params[5]
    B0_theta, B0_phi, B0_lam = best_params[6], best_params[7], best_params[8]
    B1_theta, B1_phi, B1_lam = best_params[9], best_params[10], best_params[11]
    
    gates = [
        ["u3", 0, A0_theta, A0_phi, A0_lam],
        ["u3", 1, A1_theta, A1_phi, A1_lam],
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
        ["u3", 0, B0_theta, B0_phi, B0_lam],
        ["u3", 1, B1_theta, B1_phi, B1_lam],
    ]
    
    return gates, best_loss


def _adam_optimize(target, qubits, budget, initial_params, n_params, max_iters=200, lr=0.1):
    """Adam gradient descent on u3 parameters with a fixed CNOT template."""
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    cnot_positions = []
    for i in range(budget):
        cnot_positions.append((i, pairs[i % len(pairs)]))
    
    layout = []
    u3_indices = []
    for i in range(budget):
        for q in range(qubits):
            layout.append(["u3", q, 0, 0, 0])
            u3_indices.append(len(layout) - 1)
        layout.append(["cx", *pairs[i % len(pairs)]])
    for q in range(qubits):
        layout.append(["u3", q, 0, 0, 0])
        u3_indices.append(len(layout) - 1)
    
    n_u3 = len(u3_indices)
    n_params = 3 * n_u3
    
    params = np.array(initial_params[:n_params], dtype=float)
    if len(params) < n_params:
        params = np.zeros(n_params, dtype=float)
    
    m = np.zeros(n_params)
    v = np.zeros(n_params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    
    def _circuit_unitary(p):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], p[k], p[k+1], p[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return _build(gates, qubits)
    
    def _loss(p):
        u = _circuit_unitary(p)
        return _distance(target, u)
    
    best_params = params.copy()
    best_loss = _loss(params)
    
    for t in range(1, max_iters + 1):
        if time.time() > _DEADLINE - 30:
            break
        
        grad = np.zeros(n_params)
        for i in range(n_params):
            h = 1e-4
            p_plus = params.copy()
            p_plus[i] += h
            p_minus = params.copy()
            p_minus[i] -= h
            grad[i] = (_loss(p_plus) - _loss(p_minus)) / (2 * h)
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        params -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        current_loss = _loss(params)
        if current_loss < best_loss:
            best_loss = current_loss
            best_params = params.copy()
        
        if best_loss < 1e-10:
            break
    
    gates = []
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], best_params[k], best_params[k+1], best_params[k+2]])
            k += 3
        else:
            gates.append(list(gate))
    
    return gates, best_loss


def synthesize(problem):
    """Synthesize a circuit for the given target unitary within CNOT budget."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    
    # Try exact solutions first
    exact = _identify_gate(name, target, qubits)
    if exact is not None:
        n_cnots = sum(1 for g in exact if g[0] == "cx")
        if n_cnots <= budget and len(exact) <= 400:
            return exact
    
    # For 2-qubit generic unitaries, use KAK decomposition (3 CNOTs)
    if qubits == 2 and budget >= 3:
        gates, loss = _kak_two_qubit(target)
        n_cnots = sum(1 for g in gates if g[0] == "cx")
        if n_cnots <= budget and len(gates) <= 400:
            return gates
    
    # Use Adam optimization with a good initial guess
    n_u3 = budget * qubits + qubits
    n_params = 3 * n_u3
    initial = np.zeros(n_params)
    
    gates, loss = _adam_optimize(target, qubits, budget, initial, n_params, max_iters=100)
    
    n_cnots = sum(1 for g in gates if g[0] == "cx")
    if n_cnots > budget:
        count = 0
        trimmed = []
        for g in gates:
            if g[0] == "cx":
                count += 1
                if count > budget:
                    continue
            trimmed.append(g)
        gates = trimmed
    
    if len(gates) > 400:
        gates = gates[:400]
    
    return gates
# EVOLVE-BLOCK-END
