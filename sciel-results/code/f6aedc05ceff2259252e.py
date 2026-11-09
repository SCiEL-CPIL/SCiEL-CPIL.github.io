import math
import time
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 240.0

def _remaining_time():
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _build(gates, qubits):
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
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
            idx = np.arange(n)
            bit = (idx >> (qubits - 1 - control)) & 1
            matrix = matrix[idx ^ (bit << (qubits - 1 - target)), :]
    return matrix

def _distance(target, matrix):
    n = target.shape[0]
    val = abs(np.trace(target.conj().T @ matrix)) / n
    return 1.0 - min(1.0, val)

def _count_cnots(gates):
    return sum(1 for g in gates if g[0] == "cx")

def _u3_matrix(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)

def _u3_to_su2(u):
    """Convert a 2x2 unitary to SU(2) by removing global phase."""
    det = np.linalg.det(u)
    phase = np.sqrt(det)
    if abs(phase) < 1e-15:
        phase = 1.0
    return u / phase

def _su2_to_u3(v):
    """Convert an SU(2) matrix to u3(theta, phi, lam) parameters."""
    a, b = v[0, 0], v[0, 1]
    c, d = v[1, 0], v[1, 1]
    
    theta = 2.0 * math.acos(min(1.0, max(-1.0, abs(a))))
    if theta < 1e-10:
        # a = 1, b=0, c=0, d=1 (or close)
        # u3(0, phi, lam) = [[1, 0], [0, e^{i(phi+lam)}]]
        # We have [[1, 0], [0, d]]. So e^{i(phi+lam)} = d.
        # Let phi=0, lam=arg(d).
        return 0.0, 0.0, math.atan2(d.imag, d.real)
    else:
        # a = cos(t/2)
        # b = -e^{i lam} sin(t/2)
        # c = e^{i phi} sin(t/2)
        # d = e^{i(phi+lam)} cos(t/2)
        
        # From b: e^{i lam} = -b / sin(t/2)
        # From c: e^{i phi} = c / sin(t/2)
        
        s = math.sin(theta / 2)
        if s < 1e-10:
            return 0.0, 0.0, 0.0
            
        lam = math.atan2((-b / s).imag, (-b / s).real)
        phi = math.atan2((c / s).imag, (c / s).real)
        
        return theta, phi, lam

def _decompose_2qubit_kak(U):
    """
    Decompose 2-qubit unitary U into single-qubit gates and CNOTs.
    Returns a list of gates.
    Uses KAK decomposition: U = L1 (CNOTs) L2 (CNOTs) L3 (CNOTs) L4
    Actually, the standard form is U = A (CNOT^k) B where A, B are local.
    For 2 qubits, max 3 CNOTs.
    
    We will use the algorithm from "Synthesis of Quantum Logic Gates" (Barenco et al.) or similar.
    Here we use a simpler approach:
    1. Compute the "interaction" part.
    2. Extract local unitaries.
    
    Alternatively, we can use the fact that any 2-qubit gate can be written as:
    U = (A ⊗ B) CNOT (C ⊗ D) CNOT (E ⊗ F) CNOT (G ⊗ H)
    for some single-qubit gates.
    
    A robust method:
    1. Compute M = U ⊗ U^T (sigma_y ⊗ sigma_y)
    2. Diagonalize M to get canonical angles.
    3. This is complex.
    
    Let's use a different strategy:
    We know the budget is likely 3 for generic 2-qubit.
    We can use a fixed 3-CNOT layout and optimize the 24 parameters (8 u3 gates * 3 params).
    But KAK is exact and fast.
    
    Let's try to implement the KAK extraction properly.
    Reference: "Exact synthesis of two-qubit quantum gates" (e.g. by V. V. Shende et al.)
    
    Steps:
    1. Compute the local invariants.
    2. Determine the canonical angles (a, b, c).
    3. If a=b=c=0, it's local (0 CNOTs).
    4. If one angle is 0, it's 1 CNOT.
    5. Otherwise, 3 CNOTs.
    
    Implementation:
    M = (U ⊗ U^T) (sigma_y ⊗ sigma_y)
    Eigenvalues of M are e^{i(a+b+c)}, e^{i(a+b-c)}, e^{i(a-b+c)}, e^{i(-a+b+c)}
    
    This requires careful sorting of eigenvalues.
    
    Given time constraints and complexity, let's use the optimization approach for generic 2-qubit
    but with a very good initial guess or a structured search.
    Actually, the seed program's optimization was slow.
    
    Let's try a hybrid:
    1. Try to match known patterns (CNOT, SWAP, iSWAP, etc.)
    2. For generic, use a 3-CNOT layout and Nelder-Mead.
    3. To speed up Nelder-Mead, we can use a better initial guess.
    4. Or, we can use the fact that we can decompose any 2-qubit gate into 3 CNOTs.
       There are known formulas.
    
    Let's implement the exact KAK decomposition.
    """
    n = U.shape[0]
    if n != 4:
        return []
    
    sy = np.array([[0, -1j], [1j, 0]])
    M = np.kron(U, U.T) @ np.kron(sy, sy)
    
    eigvals, eigvecs = np.linalg.eig(M)
    
    # The eigenvalues should be on the unit circle.
    # Let's extract the phases.
    phases = np.angle(eigvals)
    
    # The canonical angles a, b, c are related to these phases.
    # The sum of all 4 phases is 0 (mod 2pi) because det(M) = 1.
    # Actually, det(M) = det(U)^2 det(U^T)^2 det(sy)^2 det(sy)^2 = 1.
    # So sum(phases) = 0 mod 2pi.
    
    # We need to assign the 4 phases to the 4 combinations.
    # This is a combinatorial problem.
    # There are 4! = 24 permutations.
    # We can try all permutations and see which one gives a consistent (a, b, c).
    
    best_angles = None
    best_error = float('inf')
    
    perms = [(0,1,2,3), (0,1,3,2), (0,2,1,3), (0,2,3,1), (0,3,1,2), (0,3,2,1),
             (1,0,2,3), (1,0,3,2), (1,2,0,3), (1,2,3,0), (1,3,0,2), (1,3,2,0),
             (2,0,1,3), (2,0,3,1), (2,1,0,3), (2,1,3,0), (2,3,0,1), (2,3,1,0),
             (3,0,1,2), (3,0,2,1), (3,1,0,2), (3,1,2,0), (3,2,0,1), (3,2,1,0)]
    
    for p in perms:
        p0, p1, p2, p3 = p
        # phases[p0] = a+b+c
        # phases[p1] = a+b-c
        # phases[p2] = a-b+c
        # phases[p3] = -a+b+c
        
        # a = (p0 + p1 + p2 - p3) / 4
        # b = (p0 + p1 - p2 + p3) / 4
        # c = (p0 - p1 + p2 + p3) / 4
        
        a = (phases[p0] + phases[p1] + phases[p2] - phases[p3]) / 4.0
        b = (phases[p0] + phases[p1] - phases[p2] + phases[p3]) / 4.0
        c = (phases[p0] - phases[p1] + phases[p2] + phases[p3]) / 4.0
        
        # Check consistency:
        # The reconstructed phases should match the original (mod 2pi).
        r0 = a + b + c
        r1 = a + b - c
        r2 = a - b + c
        r3 = -a + b + c
        
        # Check error
        err = 0
        err += abs(math.sin((phases[p0] - r0) / 2))
        err += abs(math.sin((phases[p1] - r1) / 2))
        err += abs(math.sin((phases[p2] - r2) / 2))
        err += abs(math.sin((phases[p3] - r3) / 2))
        
        if err < best_error:
            best_error = err
            best_angles = (a, b, c)
    
    if best_angles is None:
        return []
        
    a, b, c = best_angles
    
    # Now we have the canonical angles.
    # We need to find the local unitaries.
    # U = e^{i alpha} (A ⊗ B) e^{i (a XX + b YY + c ZZ)} (C ⊗ D)
    
    # This part is tricky. We need to solve for A, B, C, D.
    # We can use the fact that (A ⊗ B) = e^{i alpha} U e^{-i (a XX + b YY + c ZZ)} (C ⊗ D)^†
    # This is underdetermined.
    
    # A simpler approach:
    # If the gate is "close" to a known gate, use that.
    # Otherwise, use the 3-CNOT optimization.
    
    # For now, let's return a placeholder that uses the optimization.
    # This function will be called by synthesize, which will fall back to optimization if needed.
    return None

def _detect_known_gate(target, qubits):
    n = target.shape[0]
    if qubits == 2:
        cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
        if np.allclose(target, cnot, atol=1e-10):
            return "cnot"
        iswap = 0.5 * np.array([[0, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 0]], dtype=complex)
        if np.allclose(target, iswap, atol=1e-10):
            return "iswap"
        swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
        if np.allclose(target, swap, atol=1e-10):
            return "swap"
    if qubits == 3:
        toffoli = np.zeros((8, 8), dtype=complex)
        for i in range(8):
            b2 = (i >> 2) & 1
            b1 = (i >> 1) & 1
            if b2 == 1 and b1 == 1:
                toffoli[i, i ^ 1] = 1
            else:
                toffoli[i, i] = 1
        if np.allclose(target, toffoli, atol=1e-10):
            return "toffoli"
    return "generic"

def _exact_decomposition(gate_name, qubits, budget):
    if gate_name == "cnot" and qubits == 2:
        return [["cx", 0, 1]]
    elif gate_name == "swap" and qubits == 2:
        return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
    elif gate_name == "iswap" and qubits == 2:
        # 2-CNOT iSWAP
        # CNOT(0,1), H(1), CNOT(0,1), H(1), S(0), S(1)
        # H = u3(pi/2, 0, pi)
        # S = diag(1, i).
        # S = e^{i pi/4} u3(pi/2, 0, -pi/2) ? No.
        # Let's use the 3-CNOT version if budget allows, else 2-CNOT.
        # For 2-CNOT, we need S.
        # S = u3(pi/2, 0, 0) is H.
        # S = u3(0, 0, pi/2) ? No, u3(0, phi, lam) = [[1, 0], [0, e^{i(phi+lam)}]]
        # So u3(0, 0, pi/2) = diag(1, i) = S.
        # So S = u3(0, 0, pi/2).
        
        H = math.pi / 2
        phi_h = 0
        lam_h = math.pi
        
        S = 0.0
        phi_s = 0
        lam_s = math.pi / 2
        
        return [
            ["cx", 0, 1],
            ["u3", 1, H, phi_h, lam_h],
            ["cx", 0, 1],
            ["u3", 1, H, phi_h, lam_h],
            ["u3", 0, S, phi_s, lam_s],
            ["u3", 1, S, phi_s, lam_s]
        ]
    elif gate_name == "toffoli" and qubits == 3:
        # 7-CNOT Toffoli
        gates = []
        cnots = [(2, 1), (1, 0), (2, 1), (1, 0), (2, 1), (0, 1), (2, 1)]
        for c, t in cnots:
            gates.append(["cx", c, t])
        return gates
    return None

def _nelder_mead(objective, x0, bounds=None, max_iter=500, tol=1e-6):
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    simplex = [x0.copy()]
    for i in range(n):
        point = x0.copy()
        if point[i] != 0:
            point[i] += 0.25 * point[i]
        else:
            point[i] += 0.25
        simplex.append(point)
        
    fvals = [objective(p) for p in simplex]
    
    for it in range(max_iter):
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[-1] - fvals[0]) < tol:
            break
            
        centroid = np.mean(simplex[:-1], axis=0)
        
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        
        if fvals[0] <= fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        elif fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        else:
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
                    
    return simplex[0], fvals[0]

def _optimize_generic(target, qubits, budget, time_budget=30.0):
    start = time.time()
    best_gates = None
    best_dist = 1.0
    
    if qubits == 2:
        # 3 CNOT layout
        layout = []
        pairs = [(0, 1), (1, 0), (0, 1)]
        for i in range(4):
            layout += [["u3", 0, 0.0, 0.0, 0.0], ["u3", 1, 0.0, 0.0, 0.0]]
            if i < 3:
                layout.append(["cx", pairs[i][0], pairs[i][1]])
        num_restarts = 10
    else:
        # 14 CNOT layout
        layout = []
        pairs = [(0, 1), (1, 2)]
        for i in range(15):
            layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(3)]
            if i < 14:
                layout.append(["cx", pairs[i % 2][0], pairs[i % 2][1]])
        num_restarts = 5
        
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    def objective(params):
        values = params.tolist()
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        U = _build(gates, qubits)
        return _distance(target, U)
        
    for restart in range(num_restarts):
        if time.time() - start > time_budget:
            break
        
        if restart == 0:
            x0 = np.zeros(n_params)
        else:
            rng = random.Random(42 + restart * 100)
            x0 = np.array([rng.uniform(-math.pi, math.pi) for _ in range(n_params)])
            
        try:
            max_iter = 200 if qubits == 2 else 100
            res_x, res_f = _nelder_mead(objective, x0, max_iter=max_iter)
            if res_f < best_dist:
                best_dist = res_f
                values = res_x.tolist()
                gates = []
                k = 0
                for gate in layout:
                    if gate[0] == "u3":
                        gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                        k += 3
                    else:
                        gates.append(list(gate))
                best_gates = gates
        except Exception:
            continue
            
    if best_gates is None:
        best_gates = [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        
    return best_gates, best_dist

def synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    name = problem.get("name", "")
    target = _target(problem)
    
    gate_type = _detect_known_gate(target, qubits)
    
    if gate_type != "generic":
        exact = _exact_decomposition(gate_type, qubits, budget)
        if exact is not None and _count_cnots(exact) <= budget and len(exact) <= 400:
            dist = _distance(target, _build(exact, qubits))
            if dist < 1e-6:
                return exact
    
    time_budget = min(35.0, _remaining_time() - 10.0)
    if time_budget < 2.0:
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    gates, dist = _optimize_generic(target, qubits, budget, time_budget)
    
    while _count_cnots(gates) > budget and len(gates) > 0:
        for i in range(len(gates) - 1, -1, -1):
            if gates[i][0] == "cx":
                gates.pop(i)
                break
        else:
            break
            
    return gates
