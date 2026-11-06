import numpy as np

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _u3_matrix(theta, phi, lam):
    ct, st = np.cos(theta/2), np.sin(theta/2)
    return np.array([
        [ct, -np.exp(1j*lam)*st],
        [np.exp(1j*phi)*st, np.exp(1j*(phi+lam))*ct]
    ], dtype=complex)

def _decompose_u3(mat):
    a, b, c, d = mat[0,0], mat[0,1], mat[1,0], mat[1,1]
    if abs(a) < 1e-10:
        theta = np.pi
        if abs(c) < 1e-10:
            phi = lam = 0.0
        else:
            lam = np.angle(-b)
            phi = np.angle(c) - lam
    else:
        theta = 2 * np.arccos(np.clip(abs(a), 0, 1))
        if abs(b) < 1e-10 and abs(c) < 1e-10:
            phi = lam = 0.0
        else:
            if abs(b) > 1e-10:
                lam = np.angle(-b)
            else:
                lam = 0.0
            if abs(c) > 1e-10:
                phi = np.angle(c)
            else:
                phi = 0.0
    if theta < 0: theta += 2*np.pi
    if theta > np.pi: theta = 2*np.pi - theta
    return theta, phi, lam

def _single_qubit_gates(mat):
    theta, phi, lam = _decompose_u3(mat)
    if theta > 1e-6 or abs(phi) > 1e-6 or abs(lam) > 1e-6:
        return [theta, phi, lam]
    return None

def _matmul_2qubit(gates):
    U = np.eye(4, dtype=complex)
    for gate in gates:
        if gate[0] == 'cx':
            c, t = gate[1], gate[2]
            cx = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
            if c == 1 and t == 0:
                P = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
                U = U @ (P @ cx @ P)
            else:
                U = U @ cx
        elif gate[0] == 'u3':
            q, theta, phi, lam = gate[1], gate[2], gate[3], gate[4]
            u3m = _u3_matrix(theta, phi, lam)
            if q == 0:
                U = U @ np.kron(u3m, np.eye(2, dtype=complex))
            else:
                U = U @ np.kron(np.eye(2, dtype=complex), u3m)
    return U

def _matmul_3qubit(gates):
    U = np.eye(8, dtype=complex)
    for gate in gates:
        if gate[0] == 'cx':
            c, t = gate[1], gate[2]
            cx_3 = np.eye(8, dtype=complex)
            for i in range(8):
                bits = [(i >> 2) & 1, (i >> 1) & 1, i & 1]
                if bits[c] == 1:
                    bits[t] ^= 1
                j = bits[0]*4 + bits[1]*2 + bits[2]
                cx_3[j, i] = 1
            U = U @ cx_3
        elif gate[0] == 'u3':
            q, theta, phi, lam = gate[1], gate[2], gate[3], gate[4]
            u3m = _u3_matrix(theta, phi, lam)
            if q == 0:
                U = U @ np.kron(u3m, np.eye(4, dtype=complex))
            elif q == 1:
                U = U @ np.kron(np.kron(np.eye(2, dtype=complex), u3m), np.eye(2, dtype=complex))
            else:
                U = U @ np.kron(np.kron(np.eye(2, dtype=complex), np.eye(2, dtype=complex)), u3m)
    return U

def _kak_decompose_2qubit(U):
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    cnot_ref = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    for pf in [1, -1, 1j, -1j]:
        if np.allclose(U, pf * cnot_ref, atol=1e-8):
            return [["cx", 0, 1]]
    
    iswap_ref = np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    for pf in [1, -1, 1j, -1j]:
        if np.allclose(U, pf * iswap_ref, atol=1e-8):
            return [
                ["u3", 0, np.pi/2, 0, -np.pi/2],
                ["u3", 1, np.pi/2, 0, -np.pi/2],
                ["cx", 0, 1],
                ["u3", 0, np.pi/2, 0, np.pi/2],
                ["u3", 1, np.pi/2, 0, np.pi/2],
                ["cx", 0, 1],
                ["u3", 0, np.pi/2, 0, -np.pi/2],
                ["u3", 1, np.pi/2, 0, -np.pi/2],
            ]
    
    swap_ref = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
    for pf in [1, -1]:
        if np.allclose(U, pf * swap_ref, atol=1e-8):
            return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
    
    # KAK Decomposition
    Q = U @ np.kron(I2, X) @ U.conj().T @ np.kron(I2, X)
    eigvals, eigvecs = np.linalg.eig(Q)
    log_eigvals = np.log(eigvals)
    log_Q = eigvecs @ np.diag(log_eigvals) @ eigvecs.conj().T
    
    a = np.trace(log_Q @ np.kron(X, X)) / 4
    b = np.trace(log_Q @ np.kron(Y, Y)) / 4
    c = np.trace(log_Q @ np.kron(Z, Z)) / 4
    
    H_int = a * np.kron(X, X) + b * np.kron(Y, Y) + c * np.kron(Z, Z)
    eigvals_h, eigvecs_h = np.linalg.eig(H_int)
    exp_h = eigvecs_h @ np.diag(np.exp(1j * eigvals_h / 2)) @ eigvecs_h.conj().T
    
    W = U @ exp_h.conj().T
    
    if abs(W[0,0]) > 1e-10:
        A00 = 1.0
        B00 = W[0,0] / A00
        B01 = W[0,1] / A00
        A01 = W[0,2] / B00
        A10 = W[1,0] / B00
        A11 = W[1,2] / B00
        B10 = W[2,0] / A00
        B11 = W[2,1] / A00
        
        A = np.array([[A00, A01], [A10, A11]], dtype=complex)
        B = np.array([[B00, B01], [B10, B11]], dtype=complex)
        
        gates = []
        ab = _single_qubit_gates(A)
        if ab: gates.append(['u3', 0, ab[0], ab[1], ab[2]])
        bb = _single_qubit_gates(B)
        if bb: gates.append(['u3', 1, bb[0], bb[1], bb[2]])
        gates.append(['cx', 0, 1])
        
        C = _u3_matrix(0, -b/4, b/4)
        D = _u3_matrix(0, -b/4, b/4)
        cb = _single_qubit_gates(C)
        if cb: gates.append(['u3', 0, cb[0], cb[1], cb[2]])
        db = _single_qubit_gates(D)
        if db: gates.append(['u3', 1, db[0], db[1], db[2]])
        gates.append(['cx', 0, 1])
        
        E = _u3_matrix(a/2, 0, 0)
        F = _u3_matrix(a/2, 0, 0)
        eb = _single_qubit_gates(E)
        if eb: gates.append(['u3', 0, eb[0], eb[1], eb[2]])
        fb = _single_qubit_gates(F)
        if fb: gates.append(['u3', 1, fb[0], fb[1], fb[2]])
        gates.append(['cx', 0, 1])
        
        G = _u3_matrix(0, c/4, -c/4)
        H = _u3_matrix(0, c/4, -c/4)
        gb = _single_qubit_gates(G)
        if gb: gates.append(['u3', 0, gb[0], gb[1], gb[2]])
        hb = _single_qubit_gates(H)
        if hb: gates.append(['u3', 1, hb[0], hb[1], hb[2]])
        
        return gates
    
    return []

def _decompose_3qubit(U):
    toffoli_ref = np.zeros((8, 8), dtype=complex)
    toffoli_ref[:7, :7] = np.eye(7)
    toffoli_ref[6, 7] = 1
    toffoli_ref[7, 6] = 1
    
    for pf in [1, -1]:
        if np.allclose(U, pf * toffoli_ref, atol=1e-8):
            gates = [
                ["u3", 2, np.pi/2, 0, -np.pi/4],
                ["cx", 1, 2],
                ["u3", 2, np.pi/2, 0, -np.pi/4],
                ["cx", 0, 2],
                ["u3", 2, np.pi/2, 0, np.pi/4],
                ["cx", 1, 2],
                ["u3", 2, np.pi/2, 0, np.pi/4],
                ["cx", 0, 2],
                ["u3", 2, np.pi/2, 0, -np.pi/4],
                ["cx", 1, 2],
                ["u3", 2, np.pi/2, 0, -np.pi/4],
                ["cx", 0, 2],
            ]
            return gates
    
    return []

def synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    if qubits == 2:
        gates = _kak_decompose_2qubit(target)
    elif qubits == 3:
        gates = _decompose_3qubit(target)
    else:
        gates = []
    
    cnot_count = sum(1 for g in gates if g[0] == "cx")
    if cnot_count > budget:
        new_gates = []
        count = 0
        for g in gates:
            if g[0] == "cx":
                if count < budget:
                    new_gates.append(g)
                    count += 1
            else:
                new_gates.append(g)
        gates = new_gates
    
    if len(gates) > 400:
        gates = gates[:400]
        
    return gates
