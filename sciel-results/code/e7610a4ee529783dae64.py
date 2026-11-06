<<<<<<< SEARCH
"""Differentiable tensor-network / state-vector gradient synthesis."""
# EVOLVE-BLOCK-START
import math
import random
import time

import numpy as np


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
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _embed_single(single, q, qubits):
    """Embed a 2x2 single-qubit gate on qubit q (0=MSB) into 2^qubits x 2^qubits."""
    eye = np.eye(2, dtype=complex)
    op = np.array([[1]], dtype=complex)
    for k in range(qubits):
        op = np.kron(op, single if k == q else eye)
    return op


def _cnot_permutation(control, target, qubits):
    """Return the permutation that implements CNOT(control, target) on basis indices."""
    n = 2 ** qubits
    indices = np.arange(n)
    bit = (indices >> (qubits - 1 - control)) & 1
    perm = indices ^ (bit << (qubits - 1 - target))
    return perm


def _apply_gate_to_state(state, gate, qubits):
    """Apply a gate to a state vector (or matrix) from the left."""
    if gate[0] == "u3":
        _, q, theta, phi, lam = gate
        single = _u3_matrix(theta, phi, lam)
        op = _embed_single(single, q, qubits)
        return op @ state
    else:
        _, control, target = gate
        perm = _cnot_permutation(control, target, qubits)
        return state[perm, :]


def _apply_gate_to_operator(mat, gate, qubits):
    """Apply a gate to an operator (unitary) from the left: mat -> G @ mat."""
    if gate[0] == "u3":
        _, q, theta, phi, lam = gate
        single = _u3_matrix(theta, phi, lam)
        op = _embed_single(single, q, qubits)
        return op @ mat
    else:
        _, control, target = gate
        perm = _cnot_permutation(control, target, qubits)
        return mat[perm, :]


def _circuit_unitary(params, layout, qubits):
    """Build the full unitary from parameters and layout.
    params: list of (theta, phi, lam) for each u3 gate in order.
    layout: list of ('u3', q) or ('cx', c, t)
    """
    n = 2 ** qubits
    mat = np.eye(n, dtype=complex)
    pi_idx = 0
    for gate in layout:
        if gate[0] == 'u3':
            theta, phi, lam = params[pi_idx]
            pi_idx += 1
            single = _u3_matrix(theta, phi, lam)
            op = _embed_single(single, gate[1], qubits)
            mat = op @ mat
        else:
            perm = _cnot_permutation(gate[1], gate[2], qubits)
            mat = mat[perm, :]
    return mat


def _fidelity_and_grad(params, layout, target, qubits):
    """Compute fidelity and gradient w.r.t. params using state-vector method.
    
    We compute F = |tr(T^† U)|^2 / N^2 where N = 2^qubits.
    Actually the distance is 1 - |tr(T^† U)| / N, so we want to maximize |tr(T^† U)|.
    
    For gradient-based optimization, we maximize f = |tr(T^† U)|.
    
    Using the state-vector approach:
    Let |psi> = U |0...0>. Then tr(T^† U) = sum_i <i|T^†|i> * <i|U|i> ... this is complex.
    
    Better approach: tr(T^† U) = <0...0| U^† T |0...0> ... no.
    
    Actually: tr(T^† U) = sum_i (T^† U)_{ii} = sum_i <i|T^†|i> <i|U|i>? No.
    
    Let's use: tr(T^† U) = <0| U^† T |0> where |0> is the |0...0> state? No.
    
    tr(A B) = sum_i <i| A B |i> = sum_i <i|A|j><j|B|i>... 
    
    Standard identity: tr(T^† U) = <T_flat| U_flat> where T_flat is the row-major flattening of T^†.
    
    For gradient computation, let's use the fact that:
    f = |tr(T^† U)|
    df/dtheta_k = Re[conj(tr(T^† U)) * d(tr(T^† U))/dtheta_k]
    
    d(tr(T^† U))/dtheta_k = tr(T^† dU/dtheta_k)
    
    To compute this efficiently, we can use the "sandwich" trick:
    tr(T^† dU/dtheta_k) = tr((T^† U_{k-1}^† ... U_1^†) dU_k (U_{k-1} ... U_1)) ... this is getting complex.
    
    Let me use a simpler approach: compute U, then for each parameter, compute the derivative using
    the product rule on the matrix product.
    
    Actually, for 2-3 qubits, the matrices are at most 8x8, so we can just compute the full
    derivative numerically or analytically.
    
    Let's use automatic differentiation via finite differences for the gradient, but that's slow.
    
    Better: use the analytical gradient. For a circuit U = G_m ... G_2 G_1 where each G_i is either
    a single-qubit rotation (parameterized) or a CNOT (fixed), we have:
    
    dU/dtheta_k = G_m ... G_{k+1} dG_k/dtheta_k G_{k-1} ... G_1
    
    So tr(T^† dU/dtheta_k) = tr(T^† G_m ... G_{k+1} dG_k/dtheta_k G_{k-1} ... G_1)
    
    Let A_k = T^† G_m ... G_{k+1} and B_k = G_{k-1} ... G_1
    Then tr(T^† dU/dtheta_k) = tr(A_k dG_k/dtheta_k B_k) = tr(B_k A_k dG_k/dtheta_k) (cyclic property)
    
    So we need to compute the forward and backward passes.
    """
    n = 2 ** qubits
    N = n
    # Count u3 gates
    u3_indices = []
    for i, gate in enumerate(layout):
        if gate[0] == 'u3':
            u3_indices.append(i)
    
    n_params = len(u3_indices)
    if n_params == 0:
        U = np.eye(N, dtype=complex)
        fid = abs(np.trace(target.conj().T @ U)) / N
        return fid, []
    
    # Precompute all gate matrices
    gate_mats = []
    gate_d_mats = []  # derivatives w.r.t. each of the 3 params for u3, None for cx
    pi_idx = 0
    for gate in layout:
        if gate[0] == 'u3':
            theta, phi, lam = params[pi_idx]
            pi_idx += 1
            single = _u3_matrix(theta, phi, lam)
            op = _embed_single(single, gate[1], qubits)
            gate_mats.append(op)
            
            # Compute derivatives of single-qubit gate w.r.t. theta, phi, lam
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            # d/dtheta:
            dc_dt = -0.5 * np.sin(theta / 2)
            ds_dt = 0.5 * np.cos(theta / 2)
            d_single_dt = np.array([[dc_dt, -np.exp(1j * lam) * ds_dt],
                                    [np.exp(1j * phi) * ds_dt, np.exp(1j * (phi + lam)) * dc_dt]], dtype=complex)
            d_op_dt = _embed_single(d_single_dt, gate[1], qubits)
            
            # d/dphi:
            d_single_dp = np.array([[0, 0],
                                    [1j * np.exp(1j * phi) * s, 1j * np.exp(1j * (phi + lam)) * c]], dtype=complex)
            d_op_dp = _embed_single(d_single_dp, gate[1], qubits)
            
            # d/dlam:
            d_single_dl = np.array([[0, 1j * np.exp(1j * lam) * s],
                                    [0, 1j * np.exp(1j * (phi + lam)) * c]], dtype=complex)
            d_op_dl = _embed_single(d_single_dl, gate[1], qubits)
            
            gate_d_mats.append([d_op_dt, d_op_dp, d_op_dl])
        else:
            perm = _cnot_permutation(gate[1], gate[2], qubits)
            # CNOT as a permutation matrix
            op = np.zeros((N, N), dtype=complex)
            for i in range(N):
                op[perm[i], i] = 1.0
            gate_mats.append(op)
            gate_d_mats.append(None)
    
    # Forward pass: compute partial products
    # P[i] = G_i G_{i-1} ... G_1 (product of first i+1 gates)
    m = len(layout)
    P = [None] * m
    P[0] = gate_mats[0]
    for i in range(1, m):
        P[i] = gate_mats[i] @ P[i - 1]
    
    # U = P[m-1]
    U = P[m - 1]
    tr_val = np.trace(target.conj().T @ U)
    fid = abs(tr_val) / N
    
    # Backward pass: compute Q[i] = T^† G_m ... G_{i+1}
    # Q[m-1] = T^†
    # Q[i] = Q[i+1] @ G_{i+1}^† ... wait, let me think again.
    # We want A_k = T^† G_m ... G_{k+1}
    # Let Q[k] = T^† G_m ... G_{k+1}
    # Q[m-1] = T^† (empty product after G_m... no)
    # Actually: A_k = T^† * (G_m G_{m-1} ... G_{k+1})
    # Let R[k] = G_m G_{m-1} ... G_{k+1} (product from k+1 to m-1)
    # R[m-1] = G_m = gate_mats[m-1]
    # R[k] = gate_mats[m-1] @ ... @ gate_mats[k+1]
    # R[k] = gate_mats[m-1] @ gate_mats[m-2] @ ... @ gate_mats[k+1] = R[k+1] with gate_mats[k+1] on the right? No.
    # R[k] = gate_mats[m-1] @ gate_mats[m-2] @ ... @ gate_mats[k+1]
    # R[k+1] = gate_mats[m-1] @ ... @ gate_mats[k+2]
    # So R[k] = R[k+1] @ gate_mats[k+1] is incorrect because matrix multiplication order.
    # R[k] = gate_mats[m-1] @ gate_mats[m-2] @ ... @ gate_mats[k+2] @ gate_mats[k+1] = R[k+1] @ gate_mats[k+1]? 
    # No: R[k+1] is the product from m-1 down to k+2.
    # Then R[k] = R[k+1] @ gate_mats[k+1]. Yes.
    
    R = [None] * m
    R[m - 1] = gate_mats[m - 1]
    for k in range(m - 2, -1, -1):
        R[k] = R[k + 1] @ gate_mats[k + 1]
    
    # A_k = T^† @ R[k]
    # B_k = P[k-1] if k > 0 else I
    # tr(T^† dU/dtheta_k) = tr(A_k @ dG_k @ B_k) = tr(B_k @ A_k @ dG_k)
    
    grads = []
    for idx, k in enumerate(u3_indices):
        A_k = target.conj().T @ R[k]
        if k == 0:
            B_k = np.eye(N, dtype=complex)
        else:
            B_k = P[k - 1]
        
        M = B_k @ A_k  # N x N
        
        dG = gate_d_mats[k]
        g_theta = np.trace(M @ dG[0])
        g_phi = np.trace(M @ dG[1])
        g_lam = np.trace(M @ dG[2])
        
        # Gradient of |tr_val| w.r.t. param = Re[conj(tr_val) * d(tr_val)/dparam]
        # But tr_val = tr(T^† U), and d(tr_val)/dparam = tr(T^† dU/dparam) = g above
        # So grad = Re[conj(tr_val) * g]
        conj_tr = np.conj(tr_val)
        grads.append((np.real(conj_tr * g_theta), np.real(conj_tr * g_phi), np.real(conj_tr * g_lam)))
    
    return fid, grads
=======
"""Differentiable tensor-network / state-vector gradient synthesis."""
# EVOLVE-BLOCK-START
import math
import random
import time

import numpy as np


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _embed_single(single, q, qubits):
    """Embed a 2x2 single-qubit gate on qubit q (0=MSB) into 2^qubits x 2^qubits."""
    eye = np.eye(2, dtype=complex)
    op = np.array([[1]], dtype=complex)
    for k in range(qubits):
        op = np.kron(op, single if k == q else eye)
    return op


def _cnot_permutation(control, target, qubits):
    """Return the permutation indices for CNOT(control, target) on basis indices.
    Qubit 0 is MSB. Basis index bits: qubit 0 is bit (qubits-1), qubit 1 is bit (qubits-2), etc.
    CNOT: if control bit is 1, flip target bit.
    """
    n = 2 ** qubits
    indices = np.arange(n)
    ctrl_bit_pos = qubits - 1 - control
    tgt_bit_pos = qubits - 1 - target
    ctrl_bit = (indices >> ctrl_bit_pos) & 1
    # Flip target bit where control is 1
    perm = indices ^ (ctrl_bit << tgt_bit_pos)
    return perm


def _cnot_permutation_inverse(control, target, qubits):
    """Inverse permutation for CNOT (same as forward since CNOT is self-inverse)."""
    return _cnot_permutation(control, target, qubits)


def _fidelity_and_grad(params, layout, target, qubits):
    """Compute fidelity and gradient w.r.t. params.
    
    We want to maximize |tr(T^† U)| / N where N = 2^qubits.
    U = G_m ... G_2 G_1 where each G_i is either a u3 (parameterized) or CNOT (fixed).
    
    dU/dparam_k = G_m ... G_{k+1} (dG_k/dparam_k) G_{k-1} ... G_1
    
    tr(T^† dU/dparam_k) = tr(T^† G_m ... G_{k+1} (dG_k/dparam_k) G_{k-1} ... G_1)
    
    Using cyclic property: = tr((G_{k-1} ... G_1) (T^† G_m ... G_{k+1}) (dG_k/dparam_k))
    
    Let F[k] = G_k G_{k-1} ... G_1 (forward product up to k)
    Let B[k] = T^† G_m G_{m-1} ... G_{k+1} (backward product from k+1 to m)
    
    Then tr(T^† dU/dparam_k) = tr(F[k-1] @ B[k] @ dG_k)
    
    Gradient of |tr_val| = Re[conj(tr_val) * d(tr_val)/dparam]
    """
    N = 2 ** qubits
    m = len(layout)
    
    # Identify u3 gate positions
    u3_indices = [i for i, g in enumerate(layout) if g[0] == 'u3']
    n_u3 = len(u3_indices)
    
    if n_u3 == 0:
        U = np.eye(N, dtype=complex)
        tr_val = np.trace(target.conj().T @ U)
        fid = abs(tr_val) / N
        return fid, []
    
    # Precompute gate matrices and their derivatives
    gate_mats = [None] * m
    gate_d_mats = [None] * m  # [d/dtheta, d/dphi, d/dlam] for u3, None for cx
    
    pi_idx = 0
    for i, gate in enumerate(layout):
        if gate[0] == 'u3':
            theta, phi, lam = params[pi_idx]
            pi_idx += 1
            single = _u3_matrix(theta, phi, lam)
            gate_mats[i] = _embed_single(single, gate[1], qubits)
            
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            
            # d/dtheta
            dc_dt = -0.5 * np.sin(theta / 2)
            ds_dt = 0.5 * np.cos(theta / 2)
            d_single_dt = np.array([[dc_dt, -np.exp(1j * lam) * ds_dt],
                                    [np.exp(1j * phi) * ds_dt, np.exp(1j * (phi + lam)) * dc_dt]], dtype=complex)
            
            # d/dphi
            d_single_dp = np.array([[0, 0],
                                    [1j * np.exp(1j * phi) * s, 1j * np.exp(1j * (phi + lam)) * c]], dtype=complex)
            
            # d/dlam
            d_single_dl = np.array([[0, 1j * np.exp(1j * lam) * s],
                                    [0, 1j * np.exp(1j * (phi + lam)) * c]], dtype=complex)
            
            gate_d_mats[i] = [_embed_single(d_single_dt, gate[1], qubits),
                             _embed_single(d_single_dp, gate[1], qubits),
                             _embed_single(d_single_dl, gate[1], qubits)]
        else:
            perm = _cnot_permutation(gate[1], gate[2], qubits)
            op = np.zeros((N, N), dtype=complex)
            for i2 in range(N):
                op[perm[i2], i2] = 1.0
            gate_mats[i] = op
            gate_d_mats[i] = None
    
    # Forward pass: F[i] = G_i G_{i-1} ... G_1
    F = [None] * m
    F[0] = gate_mats[0]
    for i in range(1, m):
        F[i] = gate_mats[i] @ F[i - 1]
    
    U = F[m - 1]
    tr_val = np.trace(target.conj().T @ U)
    fid = abs(tr_val) / N
    
    # Backward pass: B[i] = T^† G_m G_{m-1} ... G_{i+1}
    # B[m-1] = T^† (no gates after last)
    # B[i] = T^† G_m ... G_{i+1}
    # B[i] = B[i+1] @ G_{i+1}^† ... no wait.
    # B[i] = T^† @ (G_m @ G_{m-1} @ ... @ G_{i+1})
    # Let R[i] = G_m @ G_{m-1} @ ... @ G_{i+1}
    # R[m-1] = I (empty product)
    # R[i] = G_m @ ... @ G_{i+1} = (G_m @ ... @ G_{i+2}) @ G_{i+1} = R[i+1] @ G_{i+1}? No.
    # R[i+1] = G_m @ G_{m-1} @ ... @ G_{i+2}
    # R[i] = G_m @ G_{m-1} @ ... @ G_{i+2} @ G_{i+1} = R[i+1] @ G_{i+1}
    # Yes!
    
    R = [None] * (m + 1)
    R[m] = np.eye(N, dtype=complex)  # R[m] = empty product = I
    for i in range(m - 1, -1, -1):
        R[i] = R[i + 1] @ gate_mats[i + 1] if i + 1 < m else R[i + 1]
    
    # Actually let me redo this more carefully.
    # B[i] = T^† @ G_m @ G_{m-1} @ ... @ G_{i+1}
    # B[m-1] = T^† (no G_{m}, ..., G_{m} since i+1 = m, empty)
    # Wait, for i = m-1: B[m-1] = T^† @ (G_m @ ... @ G_m)? No.
    # B[i] = T^† @ product from j=i+1 to m-1 of G_j
    # For i = m-1: product is empty, so B[m-1] = T^†
    # For i = m-2: B[m-2] = T^† @ G_{m-1}
    # For i = 0: B[0] = T^† @ G_{m-1} @ ... @ G_1
    
    # Let me define C[i] = product from j=i+1 to m-1 of G_j
    # C[m-1] = I
    # C[i] = C[i+1] @ G_{i+1}
    # B[i] = T^† @ C[i]
    
    C = [None] * m
    C[m - 1] = np.eye(N, dtype=complex)
    for i in range(m - 2, -1, -1):
        C[i] = C[i + 1] @ gate_mats[i + 1]
    
    grads = []
    conj_tr = np.conj(tr_val)
    
    for idx, k in enumerate(u3_indices):
        # tr(T^† dU/dparam_k) = tr(F[k-1] @ B[k] @ dG_k) where B[k] = T^† @ C[k]
        if k == 0:
            F_prev = np.eye(N, dtype=complex)
        else:
            F_prev = F[k - 1]
        
        B_k = target.conj().T @ C[k]
        
        M = F_prev @ B_k
        
        dG = gate_d_mats[k]
        g_theta = np.trace(M @ dG[0])
        g_phi = np.trace(M @ dG[1])
        g_lam = np.trace(M @ dG[2])
        
        grads.append((np.real(conj_tr * g_theta), np.real(conj_tr * g_phi), np.real(conj_tr * g_lam)))
    
    return fid, grads
>>>>>>> REPLACE

