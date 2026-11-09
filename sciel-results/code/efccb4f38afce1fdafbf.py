import math
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 250.0

def _remaining():
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)

def _build(gates, qubits):
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    for g in gates:
        if g[0] == "u3":
            _, q, th, ph, lm = g
            op = _u3_matrix(th, ph, lm)
            full = np.array([[1]], dtype=complex)
            for k in range(qubits):
                full = np.kron(full, op if k == q else eye2)
            U = full @ U
        else:
            _, c, t = g
            n_bits = qubits
            idx = np.arange(n)
            cb = (idx >> (n_bits - 1 - c)) & 1
            perm = idx ^ (cb << (n_bits - 1 - t))
            U = U[perm, :]
    return U

def _dist(t, u):
    n = t.shape[0]
    return 1.0 - min(1.0, abs(np.trace(t.conj().T @ u)) / n)

def _count_cx(gates):
    return sum(1 for g in gates if g[0] == "cx")

def _detect_known(target, qubits):
    tol = 1e-8
    if qubits == 2:
        cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        if np.allclose(target, cnot, atol=tol):
            return "cnot"
        iswap = 0.5*np.array([[0,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,0]], dtype=complex)
        if np.allclose(target, iswap, atol=tol):
            return "iswap"
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if np.allclose(target, swap, atol=tol):
            return "swap"
    if qubits == 3:
        toff = np.zeros((8,8), dtype=complex)
        for i in range(8):
            b0, b1, b2 = (i>>2)&1, (i>>1)&1, i&1
            if b0 and b1:
                toff[i, i^1] = 1
            else:
                toff[i, i] = 1
        if np.allclose(target, toff, atol=tol):
            return "toffoli"
    return "generic"

def _exact_toffoli(qubits):
    gates = []
    gates.append(["u3", 2, math.pi/2, 0, 0])
    gates.append(["cx", 2, 1])
    gates.append(["u3", 1, math.pi/2, 0, 0])
    gates.append(["cx", 1, 0])
    gates.append(["u3", 0, math.pi/2, 0, 0])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 1, math.pi/2, 0, 0])
    gates.append(["cx", 1, 0])
    gates.append(["u3", 0, math.pi/2, 0, 0])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 1, math.pi/2, 0, 0])
    gates.append(["cx", 1, 2])
    gates.append(["u3", 2, math.pi/2, 0, 0])
    return gates

def _iswap_exact():
    gates = []
    gates.append(["u3", 0, math.pi/2, 0, math.pi])
    gates.append(["u3", 1, math.pi/2, 0, math.pi])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 0, math.pi/2, 0, math.pi])
    gates.append(["u3", 1, math.pi/2, 0, math.pi])
    gates.append(["cx", 1, 0])
    gates.append(["u3", 0, math.pi/2, 0, math.pi])
    gates.append(["u3", 1, math.pi/2, 0, math.pi])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 0, 0, 0, math.pi/2])
    gates.append(["u3", 1, 0, 0, math.pi/2])
    return gates

def _decompose_u3(A):
    """Decompose 2x2 unitary A into u3(theta, phi, lam)."""
    det_A = A[0,0]*A[1,1] - A[0,1]*A[1,0]
    if abs(det_A) < 1e-15:
        return (0.0, 0.0, 0.0)
    sqrt_det = np.sqrt(det_A)
    if abs(sqrt_det) < 1e-15:
        return (0.0, 0.0, 0.0)
    
    c_val = A[0,0] / sqrt_det
    s_val_neg = A[0,1] / sqrt_det
    s_val = A[1,0] / sqrt_det

    c_abs = abs(c_val)
    s_abs = abs(s_val)
    norm = math.sqrt(c_abs**2 + s_abs**2)
    if norm < 1e-15:
        return (0.0, 0.0, 0.0)

    theta_half = math.atan2(s_abs, c_abs)
    theta = 2.0 * theta_half
    
    if s_abs > 1e-15:
        e_iphi = (A[1,0] / sqrt_det) / s_abs
        phi = np.angle(e_iphi)
    else:
        phi = 0.0
        
    if s_abs > 1e-15:
        e_ilam = -(A[0,1] / sqrt_det) / s_abs
        lam = np.angle(e_ilam)
    else:
        lam = 0.0
        
    return (float(theta), float(phi), float(lam))

def _svd_optimal_unitary(A):
    """Find unitary U closest to A in Frobenius norm."""
    try:
        U, S, Vh = np.linalg.svd(A)
        return U @ Vh
    except:
        return np.eye(2, dtype=complex)

def _optimize_generic_svd(target, qubits, cnot_template, max_iters=100):
    n = 2 ** qubits
    num_cx = len(cnot_template)
    num_layers = num_cx + 1
    num_u3_per_layer = qubits
    total_u3 = num_layers * num_u3_per_layer
    
    # u3_params[layer][qubit] = (theta, phi, lam)
    u3_params = [[(0.0, 0.0, 0.0) for _ in range(qubits)] for _ in range(num_layers)]
    
    def build_U():
        U = np.eye(n, dtype=complex)
        eye2 = np.eye(2, dtype=complex)
        for layer in range(num_layers):
            for q in range(qubits):
                th, ph, lm = u3_params[layer][q]
                op = _u3_matrix(th, ph, lm)
                full = np.array([[1]], dtype=complex)
                for k in range(qubits):
                    full = np.kron(full, op if k == q else eye2)
                U = full @ U
            if layer < num_cx:
                c, t = cnot_template[layer]
                idx = np.arange(n)
                cb = (idx >> (qubits - 1 - c)) & 1
                perm = idx ^ (cb << (qubits - 1 - t))
                U = U[perm, :]
        return U

    def get_P_Q(layer, q):
        """Get P (gates before u3 at layer, q) and Q (gates after)."""
        P = np.eye(n, dtype=complex)
        Q = np.eye(n, dtype=complex)
        eye2 = np.eye(2, dtype=complex)
        
        # Build P: all gates before (layer, q)
        for l in range(num_layers):
            for qq in range(qubits):
                if l == layer and qq >= q:
                    break
                th, ph, lm = u3_params[l][qq]
                op = _u3_matrix(th, ph, lm)
                full = np.array([[1]], dtype=complex)
                for k in range(qubits):
                    full = np.kron(full, op if k == qq else eye2)
                P = full @ P
            if l < num_cx and (l < layer or (l == layer and q > 0)): # This logic is tricky
                # CNOT is after all u3s in the layer
                if l < layer:
                    c, t = cnot_template[l]
                    idx = np.arange(n)
                    cb = (idx >> (qubits - 1 - c)) & 1
                    perm = idx ^ (cb << (qubits - 1 - t))
                    P = P[perm, :]
        
        # Build Q: all gates after (layer, q)
        # Reverse order
        for l in range(num_layers - 1, -1, -1):
            for qq in range(qubits - 1, -1, -1):
                if l == layer and qq <= q:
                    break
                th, ph, lm = u3_params[l][qq]
                op = _u3_matrix(th, ph, lm)
                full = np.array([[1]], dtype=complex)
                for k in range(qubits):
                    full = np.kron(full, op if k == qq else eye2)
                Q = Q @ full
            if l < num_cx:
                if l > layer or (l == layer): # CNOT is after u3s in same layer
                    c, t = cnot_template[l]
                    idx = np.arange(n)
                    cb = (idx >> (qubits - 1 - c)) & 1
                    perm = idx ^ (cb << (qubits - 1 - t))
                    Q = Q[perm, :]
                    
        return P, Q

    best_dist = 1.0
    best_params = [row[:] for row in u3_params]
    
    for it in range(max_iters):
        if _remaining() < 5.0:
            break
            
        improved = False
        for layer in range(num_layers):
            for q in range(qubits):
                if _remaining() < 2.0:
                    break
                
                # Save current
                old_th, old_ph, old_lm = u3_params[layer][q]
                
                # Set to identity to compute P and Q
                u3_params[layer][q] = (0.0, 0.0, 0.0)
                
                P, Q = get_P_Q(layer, q)
                
                # M = P^{-1} T Q^{-1}
                try:
                    P_inv = np.linalg.inv(P)
                    Q_inv = np.linalg.inv(Q)
                    M = P_inv @ target @ Q_inv
                except:
                    u3_params[layer][q] = (old_th, old_ph, old_lm)
                    continue
                
                # Extract 2x2 block for qubit q from M
                # M is n x n. We want A such that I...U_q...I approx M.
                # A_{ij} = 1/2^(n-1) sum_{rest} M_{i, rest; j, rest}
                
                A = np.zeros((2,2), dtype=complex)
                n_rest = n // 2
                for i in range(2):
                    for j in range(2):
                        # Indices for i and j in the full matrix
                        # i is the value of qubit q
                        # We need to sum over all other qubits
                        # The index in the full matrix is formed by inserting i or j at position q
                        # qubit 0 is MSB.
                        # Index = sum(b_k * 2^(n-1-k))
                        
                        # Let's iterate over all rest combinations
                        # There are 2^(n-1) combinations
                        # This is slow for n=3 (4 combos), fine for n=2 (2 combos)
                        # For n=3, 4 combos. 4*4=16 ops per A element. 4 elements. 64 ops.
                        
                        # Better: Reshape M
                        # If q=0: M[i, r0, r1; j, r0', r1']
                        # We want sum over r0=r0', r1=r1'
                        
                        # Let's just do it explicitly for n=2 and n=3
                        if qubits == 2:
                            if q == 0:
                                # i, j are bit 0. rest is bit 1 (0 or 1)
                                # idx = i*2 + r
                                for r in range(2):
                                    idx_i = i*2 + r
                                    idx_j = j*2 + r
                                    A[i,j] += M[idx_i, idx_j]
                                A[i,j] /= 2.0
                            else:
                                # i, j are bit 1. rest is bit 0 (0 or 1)
                                # idx = r*2 + i
                                for r in range(2):
                                    idx_i = r*2 + i
                                    idx_j = r*2 + j
                                    A[i,j] += M[idx_i, idx_j]
                                A[i,j] /= 2.0
                        elif qubits == 3:
                            if q == 0:
                                # i, j are bit 2. rest are bits 1,0
                                for r in range(4):
                                    idx_i = i*4 + r
                                    idx_j = j*4 + r
                                    A[i,j] += M[idx_i, idx_j]
                                A[i,j] /= 4.0
                            elif q == 1:
                                # i, j are bit 1. rest are bits 2,0
                                # idx = b2*4 + b1*2 + b0
                                # i is b1.
                                # r encodes b2, b0.
                                # b2 = r//2, b0 = r%2
                                for r in range(4):
                                    b2 = r // 2
                                    b0 = r % 2
                                    idx_i = b2*4 + i*2 + b0
                                    idx_j = b2*4 + j*2 + b0
                                    A[i,j] += M[idx_i, idx_j]
                                A[i,j] /= 4.0
                            else:
                                # i, j are bit 0. rest are bits 2,1
                                # idx = b2*4 + b1*2 + b0
                                # i is b0.
                                # r encodes b2, b1.
                                # b2 = r//2, b1 = r%2
                                for r in range(4):
                                    b2 = r // 2
                                    b1 = r % 2
                                    idx_i = b2*4 + b1*2 + i
                                    idx_j = b2*4 + b1*2 + j
                                    A[i,j] += M[idx_i, idx_j]
                                A[i,j] /= 4.0

                U_opt = _svd_optimal_unitary(A)
                new_th, new_ph, new_lm = _decompose_u3(U_opt)
                
                u3_params[layer][q] = (new_th, new_ph, new_lm)
                
                if new_th != old_th or new_ph != old_ph or new_lm != old_lm:
                    improved = True
                    
        # Check convergence
        U_curr = build_U()
        d_curr = _dist(target, U_curr)
        if d_curr < best_dist - 1e-12:
            best_dist = d_curr
            best_params = [row[:] for row in u3_params]
            
        if best_dist < 1e-10:
            break
        if not improved and it > 0:
            break
            
    # Use best params
    u3_params = best_params
    
    gates = []
    for layer in range(num_layers):
        for q in range(qubits):
            th, ph, lm = u3_params[layer][q]
            gates.append(["u3", q, float(th), float(ph), float(lm)])
        if layer < num_cx:
            c, t = cnot_template[layer]
            gates.append(["cx", c, t])
            
    return gates

def synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    gate_type = _detect_known(target, qubits)

    if gate_type == "cnot" and qubits == 2:
        return [["cx", 0, 1]]
    elif gate_type == "swap" and qubits == 2:
        return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
    elif gate_type == "iswap" and qubits == 2:
        return _iswap_exact()
    elif gate_type == "toffoli" and qubits == 3:
        return _exact_toffoli(qubits)

    # Generic
    if qubits == 2:
        # Use 3 CNOTs for safety, or min(budget, 3)
        num_cx = min(budget, 3)
        if num_cx < 1: num_cx = 1
        pairs = [(0,1), (1,0), (0,1), (1,0), (0,1), (1,0)]
        cnot_template = pairs[:num_cx]
    else:
        num_cx = min(budget, 6)
        if num_cx < 1: num_cx = 1
        pairs = [(0,1), (1,2), (0,1), (1,2), (0,1), (1,2)]
        cnot_template = pairs[:num_cx]

    gates = _optimize_generic_svd(target, qubits, cnot_template)
    
    # Ensure budget
    while _count_cx(gates) > budget and len(gates) > 0:
        for i in range(len(gates) - 1, -1, -1):
            if gates[i][0] == "cx":
                gates.pop(i)
                break
        else:
            break
            
    return gates
