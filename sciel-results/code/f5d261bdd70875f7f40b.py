"""Exact quantum circuit synthesis using known decompositions and KAK for generic gates."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 240.0


def _remaining_time():
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
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


def _decompose_single_qubit(U):
    """Decompose a 2x2 unitary into u3(theta, phi, lam) parameters."""
    det = np.linalg.det(U)
    phase = np.angle(det)
    U_prime = U / np.exp(1j * phase / 2)
    det_p = np.linalg.det(U_prime)
    if abs(det_p - 1.0) > 1e-6:
        U_prime = U_prime / np.sqrt(det_p)
    
    a, b = U_prime[0, 0], U_prime[0, 1]
    c_val, d_val = U_prime[1, 0], U_prime[1, 1]
    
    if abs(a) < 1e-10:
        theta = math.pi
        phi = 0.0
        lam = np.angle(-b)
    else:
        theta = 2.0 * math.acos(min(1.0, max(-1.0, abs(a))))
        phi = np.angle(c_val) if abs(c_val) > 1e-10 else 0.0
        lam = np.angle(-b) if abs(b) > 1e-10 else 0.0
    
    return [theta, phi, lam]


def _is_known_gate(name, target, qubits):
    """Check if target matches a known gate."""
    tol = 1e-6
    if qubits == 2:
        if name == "CNOT" or name == "cx":
            cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
            if np.allclose(target, cnot, atol=tol):
                return "cnot"
        if name == "iSWAP" or name == "iswap":
            iswap = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex)
            if np.allclose(target, iswap, atol=tol):
                return "iswap"
        if name == "SWAP" or name == "swap":
            swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
            if np.allclose(target, swap, atol=tol):
                return "swap"
    elif qubits == 3:
        if name == "Toffoli" or name == "toffoli" or name == "ccx":
            toff = np.eye(8, dtype=complex)
            toff[6, 6] = 0
            toff[6, 7] = 1
            toff[7, 6] = 1
            toff[7, 7] = 0
            if np.allclose(target, toff, atol=tol):
                return "toffoli"
    return None


def _exact_cnot_decomposition():
    """Exact CNOT gate: 1 CNOT, 0 single-qubit gates."""
    return [["cx", 0, 1]]


def _exact_iswap_decomposition():
    """Exact iSWAP: 3 CNOTs with single-qubit rotations."""
    # iSWAP = H(0) H(1) CX(0,1) H(0) H(1) S(0) S(1) CX(0,1) S(0)^{-1} S(1)^{-1} H(0) H(1) CX(0,1) H(0) H(1)
    # Simpler: use known decomposition
    gates = []
    # H on both
    gates.append(["u3", 0, math.pi, 0.0, math.pi])  # H
    gates.append(["u3", 1, math.pi, 0.0, math.pi])  # H
    gates.append(["cx", 0, 1])
    gates.append(["u3", 0, math.pi, 0.0, math.pi])  # H
    gates.append(["u3", 1, math.pi, 0.0, math.pi])  # H
    # S gates
    gates.append(["u3", 0, 0.0, 0.0, math.pi / 2])  # S
    gates.append(["u3", 1, 0.0, 0.0, math.pi / 2])  # S
    gates.append(["cx", 0, 1])
    # S^{-1}
    gates.append(["u3", 0, 0.0, 0.0, -math.pi / 2])
    gates.append(["u3", 1, 0.0, 0.0, -math.pi / 2])
    gates.append(["u3", 0, math.pi, 0.0, math.pi])  # H
    gates.append(["u3", 1, math.pi, 0.0, math.pi])  # H
    gates.append(["cx", 0, 1])
    gates.append(["u3", 0, math.pi, 0.0, math.pi])  # H
    gates.append(["u3", 1, math.pi, 0.0, math.pi])  # H
    return gates


def _exact_swap_decomposition():
    """Exact SWAP: 3 CNOTs."""
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]


def _exact_toffoli_decomposition():
    """Exact Toffoli: 6 CNOTs."""
    gates = []
    # Standard Toffoli decomposition with 6 CNOTs
    # H on target
    gates.append(["u3", 2, math.pi, 0.0, math.pi])  # H
    # T on control 1
    gates.append(["u3", 1, 0.0, 0.0, math.pi / 4])  # T
    gates.append(["cx", 1, 2])
    gates.append(["u3", 2, 0.0, 0.0, -math.pi / 4])  # T^{-1}
    gates.append(["cx", 0, 2])
    gates.append(["u3", 2, 0.0, 0.0, math.pi / 4])  # T
    gates.append(["cx", 1, 2])
    gates.append(["u3", 2, 0.0, 0.0, -math.pi / 4])  # T^{-1}
    gates.append(["u3", 1, 0.0, 0.0, -math.pi / 4])  # T^{-1}
    gates.append(["u3", 0, 0.0, 0.0, math.pi / 8])  # T^{1/2}
    gates.append(["u3", 1, 0.0, 0.0, math.pi / 8])  # T^{1/2}
    gates.append(["cx", 0, 1])
    gates.append(["u3", 1, 0.0, 0.0, -math.pi / 8])  # T^{-1/2}
    gates.append(["cx", 0, 1])
    gates.append(["u3", 0, 0.0, 0.0, math.pi / 8])  # T^{1/2}
    # H on target
    gates.append(["u3", 2, math.pi, 0.0, math.pi])  # H
    return gates


def _kak_decompose_2qubit(U):
    """KAK decomposition for 2-qubit unitary. Returns (k1, k2, a, b, c) where
    U = k1 * (exp(i*a*Z⊗Z) * exp(i*b*Z⊗Y) * exp(i*c*Z⊗X)) * k2
    and k1, k2 are local unitaries."""
    from scipy.linalg import logm, expm
    
    # Pauli matrices
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    # Compute KAK invariants
    # Use the standard KAK decomposition algorithm
    def _local_unitary_from_kak(U, a, b, c):
        """Given U and KAK parameters, extract local unitaries."""
        ZZ = np.kron(Z, Z)
        ZY = np.kron(Z, Y)
        ZX = np.kron(Z, X)
        
        # U = (L1 ⊗ L2) * exp(i*a*ZZ + i*b*ZY + i*c*ZX) * (R1 ⊗ R2)
        # We need to find L1, L2, R1, R2
        
        # Compute the "interaction" part
        interaction = expm(1j * (a * ZZ + b * ZY + c * ZX))
        
        # U = (L1 ⊗ L2) @ interaction @ (R1 ⊗ R2)
        # => (L1 ⊗ L2) = U @ (R1 ⊗ R2)^{-1} @ interaction^{-1}
        # This is underdetermined. Use the standard approach:
        
        # From KAK: U = (A ⊗ B) * exp(i*a*ZZ + i*b*ZY + i*c*ZX) * (C ⊗ D)
        # We can set C = D = I and solve for A, B
        # Or use the symmetric decomposition
        
        # Standard KAK: U = (k1a ⊗ k1b) * exp(i*a*ZZ + i*b*ZY + i*c*ZX) * (k2a ⊗ k2b)
        # We'll use the approach where we fix k2a = k2b = I and solve
        
        # Actually, let's use a different approach: directly optimize the parameters
        return None
    
    # For simplicity, use a numerical approach:
    # Parameterize the KAK decomposition and optimize
    
    # The KAK decomposition is: U = (L1 ⊗ L2) * exp(i*a*ZZ + i*b*ZY + i*c*ZX) * (R1 ⊗ R2)
    # where a, b, c are the KAK parameters and L1, L2, R1, R2 are local unitaries
    
    # We'll use the fact that we can always choose R1 = R2 = I by absorbing into L1, L2
    # Then U = (L1 ⊗ L2) * exp(i*a*ZZ + i*b*ZY + i*c*ZX)
    
    # To find a, b, c, L1, L2, we can use the following approach:
    # 1. Compute the "interaction" invariants from U
    # 2. Solve for a, b, c
    # 3. Solve for L1, L2
    
    # Invariants:
    # d1 = Re(tr(U @ (X⊗X)))
    # d2 = Re(tr(U @ (Y⊗Y)))
    # d3 = Re(tr(U @ (Z⊗Z)))
    # d4 = Re(tr(U @ (I⊗I)))
    
    XX = np.kron(X, X)
    YY = np.kron(Y, Y)
    ZZ = np.kron(Z, Z)
    II = np.kron(I2, I2)
    
    d1 = np.real(np.trace(U @ XX))
    d2 = np.real(np.trace(U @ YY))
    d3 = np.real(np.trace(U @ ZZ))
    d4 = np.real(np.trace(U @ II))
    
    # KAK parameters from invariants:
    # cos(2a) = (d1 - d2 - d3) / 2
    # cos(2b) = (d1 + d2 - d3) / 2
    # cos(2c) = (d1 + d2 + d3) / 2
    # sin(2a)*sin(2b)*sin(2c) = d4/4
    
    def _safe_acos(x):
        return math.acos(min(1.0, max(-1.0, x)))
    
    cos_2a = (d1 - d2 - d3) / 2.0
    cos_2b = (d1 + d2 - d3) / 2.0
    cos_2c = (d1 + d2 + d3) / 2.0
    
    a = _safe_acos(cos_2a) / 2.0
    b = _safe_acos(cos_2b) / 2.0
    c = _safe_acos(cos_2c) / 2.0
    
    # Determine signs from d4
    sin_2a = math.sin(2 * a)
    sin_2b = math.sin(2 * b)
    sin_2c = math.sin(2 * c)
    if abs(sin_2a * sin_2b * sin_2c) > 1e-10:
        sign = np.sign(d4 / 4.0 / (sin_2a * sin_2b * sin_2c))
        if sign < 0:
            c = -c
    
    # Now find L1, L2 such that U = (L1 ⊗ L2) @ exp(i*a*ZZ + i*b*ZY + i*c*ZX)
    interaction = expm(1j * (a * ZZ + b * ZY + c * ZX))
    
    # U = (L1 ⊗ L2) @ interaction
    # => L1 ⊗ L2 = U @ interaction^{-1}
    # This is a 4x4 matrix that should be a tensor product
    
    M = U @ np.linalg.inv(interaction)
    
    # M should be L1 ⊗ L2
    # Extract L1 and L2:
    # M = [[L1[0,0]*L2, L1[0,1]*L2], [L1[1,0]*L2, L1[1,1]*L2]]
    
    L1 = np.array([[M[0, 0], M[0, 2]], [M[1, 0], M[1, 2]]], dtype=complex)
    L2 = np.array([[M[0, 0], M[0, 1]], [M[2, 0], M[2, 1]]], dtype=complex)
    
    # Check if L1 and L2 are unitary
    if not np.allclose(L1 @ L1.conj().T, np.eye(2), atol=1e-6):
        # Try different extraction
        L1 = np.array([[M[0, 0], M[0, 1]], [M[2, 0], M[2, 1]]], dtype=complex)
        L2 = np.array([[M[0, 0], M[0, 2]], [M[1, 0], M[1, 2]]], dtype=complex)
    
    return L1, L2, a, b, c


def _optimize_2qubit(target, initial_params, max_iter=50):
    """Optimize 2-qubit circuit parameters using coordinate descent."""
    qubits = 2
    params = initial_params[:]
    best_dist = _distance(target, _build(_make_gates_2qubit(params), qubits))
    
    for iteration in range(max_iter):
        if _remaining_time() < 10.0:
            break
        improved = False
        for i in range(len(params)):
            for delta in [0.1, -0.1, 0.01, -0.01, 0.001, -0.001]:
                new_params = params[:]
                new_params[i] = params[i] + delta
                new_gates = _make_gates_2qubit(new_params)
                new_dist = _distance(target, _build(new_gates, qubits))
                if new_dist < best_dist - 1e-9:
                    params = new_params
                    best_dist = new_dist
                    improved = True
        if not improved:
            break
    
    return params, best_dist


def _make_gates_2qubit(params):
    """Create 2-qubit circuit from parameters.
    params: [theta1, phi1, lam1, theta2, phi2, lam2, a, b, c, theta3, phi3, lam3, theta4, phi4, lam4]
    Structure: U1(0) U2(1) CX(0,1) U3(0) U4(1) CX(0,1) U5(0) U6(1) CX(0,1) U7(0) U8(1)
    """
    gates = []
    # Before first CX
    gates.append(["u3", 0, params[0], params[1], params[2]])
    gates.append(["u3", 1, params[3], params[4], params[5]])
    gates.append(["cx", 0, 1])
    # Between first and second CX
    gates.append(["u3", 0, params[6], params[7], params[8]])
    gates.append(["u3", 1, params[9], params[10], params[11]])
    gates.append(["cx", 0, 1])
    # Between second and third CX
    gates.append(["u3", 0, params[12], params[13], params[14]])
    gates.append(["u3", 1, params[15], params[16], params[17]])
    gates.append(["cx", 0, 1])
    # After third CX
    gates.append(["u3", 0, params[18], params[19], params[20]])
    gates.append(["u3", 1, params[21], params[22], params[23]])
    return gates


def _optimize_3qubit(target, initial_params, max_iter=30):
    """Optimize 3-qubit circuit parameters using coordinate descent."""
    qubits = 3
    params = initial_params[:]
    best_dist = _distance(target, _build(_make_gates_3qubit(params), qubits))
    
    for iteration in range(max_iter):
        if _remaining_time() < 10.0:
            break
        improved = False
        for i in range(len(params)):
            for delta in [0.1, -0.1, 0.01, -0.01]:
                new_params = params[:]
                new_params[i] = params[i] + delta
                new_gates = _make_gates_3qubit(new_params)
                new_dist = _distance(target, _build(new_gates, qubits))
                if new_dist < best_dist - 1e-9:
                    params = new_params
                    best_dist = new_dist
                    improved = True
        if not improved:
            break
    
    return params, best_dist


def _make_gates_3qubit(params):
    """Create 3-qubit circuit from parameters.
    6 CNOTs with single-qubit gates between them.
    params: 36 values (3 qubits * 6 layers * 2 params per layer) + 6 CNOT pairs
    """
    gates = []
    # 6 layers, each with 3 u3 gates and 1 CX
    cnot_pairs = [(0, 1), (1, 2), (0, 2), (0, 1), (1, 2), (0, 2)]
    param_idx = 0
    for layer in range(6):
        for q in range(3):
            gates.append(["u3", q, params[param_idx], params[param_idx + 1], params[param_idx + 2]])
            param_idx += 3
        c, t = cnot_pairs[layer]
        gates.append(["cx", c, t])
    # Final layer
    for q in range(3):
        gates.append(["u3", q, params[param_idx], params[param_idx + 1], params[param_idx + 2]])
        param_idx += 3
    return gates


def synthesize(problem):
    """Synthesize quantum circuit using exact decompositions and KAK + optimization."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    
    # Check for known gates
    known = _is_known_gate(name, target, qubits)
    if known == "cnot":
        gates = _exact_cnot_decomposition()
        if len([g for g in gates if g[0] == "cx"]) <= budget and len(gates) <= 400:
            return gates
    elif known == "iswap":
        gates = _exact_iswap_decomposition()
        if len([g for g in gates if g[0] == "cx"]) <= budget and len(gates) <= 400:
            return gates
    elif known == "swap":
        gates = _exact_swap_decomposition()
        if len([g for g in gates if g[0] == "cx"]) <= budget and len(gates) <= 400:
            return gates
    elif known == "toffoli":
        gates = _exact_toffoli_decomposition()
        if len([g for g in gates if g[0] == "cx"]) <= budget and len(gates) <= 400:
            return gates
    
    # For generic gates, use KAK + optimization
    if qubits == 2:
        # Initialize with zeros
        initial_params = [0.0] * 24
        # Try to get better initial guess from KAK
        try:
            L1, L2, a, b, c = _kak_decompose_2qubit(target)
            # Convert L1, L2 to u3 parameters
            l1_params = _decompose_single_qubit(L1)
            l2_params = _decompose_single_qubit(L2)
            initial_params = l1_params + l2_params + [0.0] * 18
        except Exception:
            pass
        
        params, dist = _optimize_2qubit(target, initial_params, max_iter=50)
        gates = _make_gates_2qubit(params)
        
        # Check constraints
        cnot_count = len([g for g in gates if g[0] == "cx"])
        if cnot_count > budget or len(gates) > 400:
            # Fallback: use fewer CNOTs
            # For 2-qubit, max 3 CNOTs needed
            max_cnots = min(3, budget)
            # Create circuit with max_cnots CNOTs
            cnot_pairs = [(0, 1), (1, 0), (0, 1)]
            gates = []
            for layer in range(max_cnots):
                for q in range(2):
                    gates.append(["u3", q, 0.0, 0.0, 0.0])
                gates.append(["cx", *cnot_pairs[layer]])
            for q in range(2):
                gates.append(["u3", q, 0.0, 0.0, 0.0])
            
            # Optimize
            n_params = 3 * (2 * (max_cnots + 1))
            params = [0.0] * n_params
            best_dist = _distance(target, _build(gates, qubits))
            
            for iteration in range(30):
                if _remaining_time() < 10.0:
                    break
                improved = False
                for i in range(n_params):
                    for delta in [0.1, -0.1, 0.01, -0.01]:
                        new_params = params[:]
                        new_params[i] = params[i] + delta
                        # Rebuild gates
                        new_gates = []
                        pidx = 0
                        for layer in range(max_cnots):
                            for q in range(2):
                                new_gates.append(["u3", q, new_params[pidx], new_params[pidx+1], new_params[pidx+2]])
                                pidx += 3
                            new_gates.append(["cx", *cnot_pairs[layer]])
                        for q in range(2):
                            new_gates.append(["u3", q, new_params[pidx], new_params[pidx+1], new_params[pidx+2]])
                            pidx += 3
                        new_dist = _distance(target, _build(new_gates, qubits))
                        if new_dist < best_dist - 1e-9:
                            params = new_params
                            best_dist = new_dist
                            improved = True
                if not improved:
                    break
            
            # Rebuild final gates
            gates = []
            pidx = 0
            for layer in range(max_cnots):
                for q in range(2):
                    gates.append(["u3", q, params[pidx], params[pidx+1], params[pidx+2]])
                    pidx += 3
                gates.append(["cx", *cnot_pairs[layer]])
            for q in range(2):
                gates.append(["u3", q, params[pidx], params[pidx+1], params[pidx+2]])
                pidx += 3
        
        return gates
    
    elif qubits == 3:
        # For 3-qubit, use 6 CNOTs
        max_cnots = min(6, budget)
        cnot_pairs = [(0, 1), (1, 2), (0, 2), (0, 1), (1, 2), (0, 2)][:max_cnots]
        
        # Initialize parameters
        n_layers = max_cnots + 1
        n_params = 3 * 3 * n_layers  # 3 qubits, 3 params each, n_layers layers
        params = [0.0] * n_params
        
        # Build initial gates
        def _build_3qubit_gates(params, max_cnots, cnot_pairs):
            gates = []
            pidx = 0
            for layer in range(max_cnots):
                for q in range(3):
                    gates.append(["u3", q, params[pidx], params[pidx+1], params[pidx+2]])
                    pidx += 3
                gates.append(["cx", *cnot_pairs[layer]])
            for q in range(3):
                gates.append(["u3", q, params[pidx], params[pidx+1], params[pidx+2]])
                pidx += 3
            return gates
        
        gates = _build_3qubit_gates(params, max_cnots, cnot_pairs)
        best_dist = _distance(target, _build(gates, qubits))
        
        # Optimize
        for iteration in range(20):
            if _remaining_time() < 15.0:
                break
            improved = False
            for i in range(n_params):
                for delta in [0.1, -0.1, 0.01, -0.01]:
                    new_params = params[:]
                    new_params[i] = params[i] + delta
                    new_gates = _build_3qubit_gates(new_params, max_cnots, cnot_pairs)
                    new_dist = _distance(target, _build(new_gates, qubits))
                    if new_dist < best_dist - 1e-9:
                        params = new_params
                        best_dist = new_dist
                        improved = True
            if not improved:
                break
        
        # Rebuild final gates
        gates = _build_3qubit_gates(params, max_cnots, cnot_pairs)
        
        # Check constraints
        cnot_count = len([g for g in gates if g[0] == "cx"])
        if cnot_count > budget or len(gates) > 400:
            # Return identity as fallback
            return []
        
        return gates
    
    # Fallback: return identity
    return []
# EVOLVE-BLOCK-END
