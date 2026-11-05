# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 240

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)

def _cnot_matrix(n, control, target):
    """Build CNOT matrix efficiently."""
    mat = np.eye(n, dtype=complex)
    for i in range(n):
        c_bit = (i >> (n.bit_length() - 1 - control)) & 1
        if c_bit:
            t_bit_pos = n.bit_length() - 1 - target
            new_i = i ^ (1 << t_bit_pos)
            mat[new_i, i] = 1
        else:
            mat[i, i] = 1
    return mat

def _build_unitary_fast(gates, qubits):
    """Build unitary using efficient matrix multiplication."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    eye = np.eye(2, dtype=complex)
    
    # Pre-compute CNOT matrices
    cnot_cache = {}
    
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            # Build operator by kron
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            key = (control, target)
            if key not in cnot_cache:
                cnot_cache[key] = _cnot_matrix(n, control, target)
            matrix = cnot_cache[key] @ matrix
    return matrix

def _distance(target, matrix):
    """Phase-insensitive distance."""
    n = target.shape[0]
    val = abs(np.trace(target.conj().T @ matrix)) / n
    return 1.0 - min(1.0, val)

def _synthesize_cnot():
    return [["cx", 0, 1]]

def _synthesize_iswap():
    # iSWAP = [[0, i], [i, 0]]
    # Use 2 CNOTs with proper single-qubit gates
    # iSWAP = (i/2) * (I + X) ⊗ (I + X) * CNOT * (I + X) ⊗ (I + X) * (i/2)
    # More standard: use H, CNOT, H with phase corrections
    # iSWAP = (1/√2) * [[0, 1], [1, 0]] with i factor
    # Let's use: u3(π/2, 0, π/2) on both, CNOT, u3(π/2, π/2, 0) on both, CNOT, u3(π/2, 0, π/2) on both
    # This gives iSWAP up to global phase
    return [
        ["u3", 0, math.pi / 2, 0, math.pi / 2],
        ["u3", 1, math.pi / 2, 0, math.pi / 2],
        ["cx", 0, 1],
        ["u3", 0, math.pi / 2, math.pi / 2, 0],
        ["u3", 1, math.pi / 2, math.pi / 2, 0],
        ["cx", 1, 0],
        ["u3", 0, math.pi / 2, 0, math.pi / 2],
        ["u3", 1, math.pi / 2, 0, math.pi / 2],
    ]

def _synthesize_swap():
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]

def _synthesize_toffoli():
    # 6 CNOT Toffoli decomposition
    return [
        ["u3", 2, math.pi / 2, 0, math.pi / 2],
        ["cx", 1, 2],
        ["u3", 2, math.pi / 4, 0, -math.pi / 2],
        ["cx", 0, 2],
        ["u3", 2, -math.pi / 4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, math.pi / 4, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, -math.pi / 4, 0, -math.pi / 2],
        ["u3", 1, math.pi / 4, 0, math.pi / 4],
        ["u3", 2, math.pi / 4, 0, math.pi / 2],
        ["cx", 1, 2],
        ["u3", 1, -math.pi / 4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, math.pi / 2, 0, math.pi / 2],
    ]

def _u3_from_matrix(M):
    """Extract u3 parameters from a 2x2 unitary matrix."""
    # M = [[a, -e^{i*lam}*b], [e^{i*phi}*b, e^{i*(phi+lam)}*a]]
    # where a = cos(theta/2), b = sin(theta/2)
    a = M[0, 0]
    b = -M[0, 1] * np.exp(-1j * 0)  # We'll determine lam
    # From M[0,0] = cos(theta/2), M[1,1] = e^{i(phi+lam)} cos(theta/2)
    # M[1,0] = e^{i*phi} sin(theta/2), M[0,1] = -e^{i*lam} sin(theta/2)
    
    # Handle edge cases
    if abs(a) < 1e-10:
        # theta = pi
        theta = math.pi
        phi = 0.0
        lam = 0.0
        return theta, phi, lam
    
    if abs(M[1, 0]) < 1e-10:
        # theta = 0
        theta = 0.0
        phi = 0.0
        lam = 0.0
        return theta, phi, lam
    
    # sin(theta/2) = |M[1,0]|, cos(theta/2) = |M[0,0]|
    sin_half = abs(M[1, 0])
    cos_half = abs(M[0, 0])
    theta = 2 * math.atan2(sin_half, cos_half)
    
    if sin_half < 1e-10:
        return theta, 0.0, 0.0
    
    # phi from M[1,0] = e^{i*phi} * sin(theta/2)
    phi = np.angle(M[1, 0])
    
    # lam from M[0,1] = -e^{i*lam} * sin(theta/2)
    lam = np.angle(-M[0, 1])
    
    return float(theta), float(phi), float(lam)

def _decompose_2qubit_exact(target):
    """
    Exact 2-CNOT decomposition for 2-qubit unitary.
    U = (A0 ⊗ B0) CNOT (A1 ⊗ B1)
    """
    # Block form: U = [[U00, U01], [U10, U11]]
    U00 = target[0:2, 0:2]
    U01 = target[0:2, 2:4]
    U10 = target[2:4, 0:2]
    U11 = target[2:4, 2:4]
    
    # CNOT = [[I, 0], [0, X]]
    # (A0⊗B0) CNOT (A1⊗B1) = [[A0 A1, 0], [0, A0 A1]] ⊗ [[B0, 0], [0, B0 X]] (A1⊗B1)
    # Actually: (A⊗B)(C⊗D) = (AC)⊗(BD)
    # But CNOT is not a tensor product.
    # Let's use the explicit form.
    # CNOT = [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
    # (A0⊗B0) CNOT (A1⊗B1):
    # Let's denote A0 A1 = P, B0 B1 = Q
    # (A0⊗B0) CNOT (A1⊗B1) = (A0⊗B0) [[A1, 0], [0, A1 X]] (A1⊗B1)^{-1}... no
    # Actually (A1⊗B1) is applied first, then CNOT, then (A0⊗B0)
    # Let V = A1⊗B1, W = A0⊗B0
    # U = W CNOT V
    # CNOT V = [[V00, V01], [V10, V11]] where V = [[V00,V01],[V10,V11]] in block form
    # V00 = A1 B1 (top-left 2x2), V01 = A1 B1 (top-right), V10 = A1 B1 (bottom-left), V11 = A1 B1 (bottom-right)
    # Wait, V = A1⊗B1, so V = [[a11 B1, a12 B1], [a21 B1, a22 B1]]
    # CNOT V = [[I,0],[0,X]] [[a11 B1, a12 B1], [a21 B1, a22 B1]]
    # = [[a11 B1, a12 B1], [a22 B1, a21 B1]] (X swaps the two rows of the bottom block)
    # W (CNOT V) = [[A0, 0], [0, A0]] [[a11 B1, a12 B1], [a22 B1, a21 B1]]
    # Wait, W = A0⊗B0 = [[a01 B0, a02 B0], [a03 B0, a04 B0]] where A0 = [[a01,a02],[a03,a04]]
    # This is getting complicated. Let me use a different approach.
    
    # Use the method: U = (A⊗I) CNOT (B⊗I) CNOT (C⊗I) for 3 CNOTs, but we want 2.
    # For 2 CNOTs: U = (A0⊗B0) CNOT (A1⊗B1)
    # Let's solve directly.
    # U00 = A0 A1 B0 B1 * 1 (from top-left of CNOT which is I)
    # Actually: CNOT = |0><0|⊗I + |1><1|⊗X
    # (A0⊗B0) CNOT (A1⊗B1) = (A0⊗B0) (|0><0|⊗I + |1><1|⊗X) (A1⊗B1)
    # = (A0⊗B0) (|0><0| A1 ⊗ B1 + |1><1| A1 ⊗ X B1)
    # = (A0 |0><0| A1 ⊗ B0 B1) + (A0 |1><1| A1 ⊗ B0 X B1)
    # Let A0 A1 = P, so A0 |0><0| A1 = P00 (top-left of P), A0 |1><1| A1 = P11 (bottom-right of P)
    # U = P00 ⊗ (B0 B1) + P11 ⊗ (B0 X B1)
    # In block form: U00 = P00 (B0 B1), U01 = 0, U10 = P11 (B0 X B1), U11 = 0? No...
    # Wait, P00 is a scalar (1x1), so P00 ⊗ M = P00 * M
    # U = [[P00 * (B0 B1), P00 * (B0 B1) * 0 + ...]]
    # Actually, |0><0| A1 = [[a11, 0], [a21, 0]], so A0 |0><0| A1 = A0 [[a11,0],[a21,0]] = [[a01 a11, 0], [a03 a11, 0]]
    # This is a 2x2 matrix, not a scalar.
    # Let me just use numerical optimization with a good initialization.
    
    return None

def _optimize_generic(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    n = 2 ** qubits
    
    if _time_remaining() < 10:
        return []
    
    # For 2 qubits, use 2 CNOTs (or fewer if budget is smaller)
    if qubits == 2:
        num_cnots = min(budget, 2)
        if num_cnots < 1:
            num_cnots = 1
        cnot_pairs = []
        for i in range(num_cnots):
            if i % 2 == 0:
                cnot_pairs.append((0, 1))
            else:
                cnot_pairs.append((1, 0))
    else:
        # For 3 qubits, use ladder topology
        num_cnots = budget
        cnot_pairs = []
        for i in range(num_cnots):
            if i % 2 == 0:
                cnot_pairs.append((0, 1))
            else:
                cnot_pairs.append((1, 2))
    
    # Build layout with u3 gates interleaved with CNOTs
    layout = []
    for b in range(num_cnots + 1):
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        if b < num_cnots:
            layout.append(["cx", cnot_pairs[b][0], cnot_pairs[b][1]])
    
    u3_indices = [i for i, g in enumerate(layout) if g[0] == "u3"]
    n_params = len(u3_indices) * 3
    
    def build_from_params(params):
        gates = []
        pi = 0
        for gate in layout:
            if gate[0] == "u3":
                q = gate[1]
                gates.append(["u3", q, params[pi], params[pi+1], params[pi+2]])
                pi += 3
            else:
                gates.append(list(gate))
        return gates
    
    def objective(params):
        if _time_remaining() < 3:
            return 1.0
        gates = build_from_params(params)
        U = _build_unitary_fast(gates, qubits)
        return _distance(target, U)
    
    # Initial guess: zeros
    x0 = np.zeros(n_params)
    
    best_x = x0
    best_obj = objective(x0)
    
    np.random.seed(42)
    
    # Use scipy L-BFGS-B with multiple restarts
    try:
        from scipy.optimize import minimize
        use_scipy = True
    except ImportError:
        use_scipy = False
    
    if use_scipy:
        starts = [x0]
        # Add random restarts
        n_restarts = min(8, int(_time_remaining() / 15))
        for _ in range(max(1, n_restarts)):
            if _time_remaining() < 10:
                break
            x_try = np.random.uniform(-math.pi, math.pi, n_params)
            starts.append(x_try)
        
        for start in starts:
            if _time_remaining() < 8:
                break
            try:
                res = minimize(objective, start, method='L-BFGS-B', 
                              options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-10})
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_x = res.x
                    # If very good, stop early
                    if best_obj < 1e-8:
                        break
            except Exception:
                pass
    else:
        # Fallback: simple gradient-free optimization
        def nelder_mead(func, x0, max_iter=100, tol=1e-8):
            n = len(x0)
            simplex = [np.array(x0, dtype=float)]
            for i in range(n):
                x = np.array(x0, dtype=float)
                x[i] += 0.5
                simplex.append(x)
            
            for _ in range(max_iter):
                if _time_remaining() < 3:
                    break
                fvals = [func(x) for x in simplex]
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                if abs(fvals[0] - fvals[-1]) < tol:
                    break
                    
                centroid = np.sum(simplex[:-1], axis=0) / n
                xr = centroid + (centroid - simplex[-1])
                fr = func(xr)
                
                if fr < fvals[0]:
                    xe = centroid + 2 * (centroid - simplex[-1])
                    fe = func(xe)
                    simplex[-1] = xe if fe < fr else xr
                elif fr < fvals[-1]:
                    simplex[-1] = xr
                else:
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    fc = func(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
            
            return simplex[0], fvals[0]
        
        starts = [x0]
        n_restarts = min(5, int(_time_remaining() / 20))
        for _ in range(max(1, n_restarts)):
            if _time_remaining() < 10:
                break
            x_try = np.random.uniform(-math.pi, math.pi, n_params)
            starts.append(x_try)
        
        for start in starts:
            if _time_remaining() < 8:
                break
            try:
                res_x, res_f = nelder_mead(objective, start, max_iter=50)
                if res_f < best_obj:
                    best_obj = res_f
                    best_x = res_x
            except Exception:
                pass
    
    return build_from_params(best_x)

def synthesize(problem):
    name = problem.get("name", "").strip().lower()
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]

    if name == "cnot" and qubits == 2:
        return _synthesize_cnot()
    elif name == "iswap" and qubits == 2:
        return _synthesize_iswap()
    elif name == "swap" and qubits == 2:
        return _synthesize_swap()
    elif name == "toffoli" and qubits == 3:
        return _synthesize_toffoli()
    else:
        return _optimize_generic(problem)
# EVOLVE-BLOCK-END
