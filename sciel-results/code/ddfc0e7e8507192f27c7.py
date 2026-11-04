"""Matrix multiplication 5x5x5 - Kronecker product of Strassen (2x2) for 4x4 block + schoolbook for 1x1."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np


def _strassen_2x2():
    """
    Rank-7 Strassen decomposition for 2x2 matrix multiplication.
    Returns list of 7 tuples (u_vec, v_vec, w_vec) where each vec has length 4.
    u_vec[i*m+j] gives coefficient for A element (i,j)
    v_vec[j*p+k] gives coefficient for B element (j,k)
    w_vec[k*n+i] gives coefficient for C element (i,k)
    """
    terms = []
    
    # M1 = (a00+a11)(b00+b11) -> C00 += M1, C11 += M1
    terms.append((
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
    ))
    
    # M2 = (a10+a11)b00 -> C10 += M2, C11 -= M2
    terms.append((
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
    ))
    
    # M3 = a00(b01-b11) -> C01 += M3, C11 += M3
    terms.append((
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 1, 0, 1],
    ))
    
    # M4 = a11(b10-b00) -> C00 += M4, C10 += M4
    terms.append((
        [0, 0, 0, 1],
        [0, 0, 1, -1],
        [1, 0, 1, 0],
    ))
    
    # M5 = (a00+a01)b11 -> C00 -= M5, C01 += M5
    terms.append((
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [-1, 1, 0, 0],
    ))
    
    # M6 = (a10-a00)(b00+b01) -> C11 += M6
    terms.append((
        [-1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
    ))
    
    # M7 = (a01-a11)(b10+b11) -> C00 += M7
    terms.append((
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
    ))
    
    return terms


def _verify_decomposition(U, V, W, n=5, m=5, p=5, tol=1e-10):
    """Verify that the tensor decomposition is exact using vectorized computation."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Vectorized verification:
    # For each (i,k): C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r]*A[i,j]) * (sum_j V[j*p+k, r]*B[j,k])
    
    # Compute u_vals: shape (n, m, rank) -> (n, rank) via einsum
    # u_vals[i, r] = sum_j U[i*m+j, r] * A[i, j]
    u_vals = np.einsum('ij,imr->ir', A, U.reshape(n, m, rank))
    # v_vals[j, r] = sum_k V[j*p+k, r] * B[j, k]  -> actually v_vals[k, r] = sum_j V[j*p+k, r] * B[j,k]
    # Let's think: V has shape (m*p, rank). V[j*p+k, r] is coefficient for B[j,k].
    # v_vals[k, r] = sum_j B[j, k] * V[j*p+k, r]
    v_vals = np.einsum('jk,jkr->kr', B, V.reshape(m, p, rank))
    
    # C_decomp[i, k] = sum_r W[k*n+i, r] * u_vals[i, r] * v_vals[k, r]
    C_decomp = np.einsum('kir,ir,kr->ik', W.reshape(p, n, rank), u_vals, v_vals)
    
    err = np.max(np.abs(C_decomp - C_expected))
    return err < tol, err


def _build_decomposition_41():
    """
    Build decomposition using 4+1 block split.
    - 4x4 block: Kronecker product of two Strassen 2x2 decompositions (rank 49)
    - 1x1 corner: schoolbook (rank 1)
    Total rank: 50
    """
    n = m = p = 5
    U_rows = n * m  # 25
    V_rows = m * p  # 25
    W_rows = p * n  # 25
    
    terms = []  # list of (u_dict, v_dict, w_dict)
    
    # Get Strassen terms for 2x2
    strassen = _strassen_2x2()
    
    # Kronecker product of two Strassen decompositions for 4x4
    # The 4x4 matrix can be viewed as a 2x2 block matrix of 2x2 blocks.
    # If A = [[A00, A01], [A10, A11]] where each Ai is 2x2,
    # and B = [[B00, B01], [B10, B11]],
    # then C = A*B has blocks Cij = sum_k Aik * Bkj.
    # 
    # The Kronecker product approach: if we have decompositions for 2x2:
    #   T_2x2 = sum_r u_r ⊗ v_r ⊗ w_r (rank 7)
    # Then for 4x4 (which is 2x2 of 2x2 blocks):
    #   T_4x4 = T_2x2 ⊗ T_2x2 (rank 7*7 = 49)
    #
    # More precisely, for the Kronecker product of tensor decompositions:
    # If T1 = sum_{r1} u1_{r1} ⊗ v1_{r1} ⊗ w1_{r1} for size n1
    # and T2 = sum_{r2} u2_{r2} ⊗ v2_{r2} ⊗ w2_{r2} for size n2
    # Then T1 ⊗ T2 = sum_{r1,r2} (u1_{r1} ⊗ u2_{r2}) ⊗ (v1_{r1} ⊗ v2_{r2}) ⊗ (w1_{r1} ⊗ w2_{r2})
    # This gives a decomposition for size n1*n2 with rank r1*r2.
    
    # For our case, both are 2x2, so we get 4x4 with rank 49.
    # The indexing: for 2x2, U index is i*2+j, V index is j*2+k, W index is k*2+i.
    # For 4x4 via Kronecker: 
    #   A element (i,j) where i = i1*2+i2, j = j1*2+j2
    #   U_4x4[i*4+j, (r1,r2)] = U_2x2[i1*2+j1, r1] * U_2x2[i2*2+j2, r2]
    #   V_4x4[j*4+k, (r1,r2)] = V_2x2[j1*2+k1, r1] * V_2x2[j2*2+k2, r2]
    #   W_4x4[k*4+i, (r1,r2)] = W_2x2[k1*2+i1, r1] * W_2x2[k2*2+i2, r2]
    
    for r1 in range(7):
        u1, v1, w1 = strassen[r1]
        for r2 in range(7):
            u2, v2, w2 = strassen[r2]
            u_dict = {}
            v_dict = {}
            w_dict = {}
            
            # Build U for 4x4: index (i*4+j) where i=i1*2+i2, j=j1*2+j2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            coeff = u1[i1*2+j1] * u2[i2*2+j2]
                            if coeff != 0:
                                global_i = i1*2 + i2
                                global_j = j1*2 + j2
                                global_u = global_i * 4 + global_j
                                u_dict[global_u] = u_dict.get(global_u, 0) + coeff
            
            # Build V for 4x4: index (j*4+k) where j=j1*2+j2, k=k1*2+k2
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            coeff = v1[j1*2+k1] * v2[j2*2+k2]
                            if coeff != 0:
                                global_j = j1*2 + j2
                                global_k = k1*2 + k2
                                global_v = global_j * 4 + global_k
                                v_dict[global_v] = v_dict.get(global_v, 0) + coeff
            
            # Build W for 4x4: index (k*4+i) where k=k1*2+k2, i=i1*2+i2
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            coeff = w1[k1*2+i1] * w2[k2*2+i2]
                            if coeff != 0:
                                global_k = k1*2 + k2
                                global_i = i1*2 + i2
                                global_w = global_k * 4 + global_i
                                w_dict[global_w] = w_dict.get(global_w, 0) + coeff
            
            terms.append((u_dict, v_dict, w_dict))
    
    # Add the 1x1 schoolbook term for position (4,4) -> (4,4)
    # A[4,4] * B[4,4] -> C[4,4]
    terms.append(({4*5+4: 1}, {4*5+4: 1}, {4*5+4: 1}))
    
    rank = len(terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, coeff in u_dict.items():
            if 0 <= idx < U_rows:
                U[idx][c] = coeff
        for idx, coeff in v_dict.items():
            if 0 <= idx < V_rows:
                V[idx][c] = coeff
        for idx, coeff in w_dict.items():
            if 0 <= idx < W_rows:
                W[idx][c] = coeff
    
    return U, V, W


def _build_schoolbook():
    """Build the trivial schoolbook decomposition (rank 125)."""
    n = m = p = 5
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Try the 4+1 block decomposition (Kronecker product of two Strassen + schoolbook)
    U, V, W = _build_decomposition_41()
    
    # Verify
    valid, err = _verify_decomposition(U, V, W)
    
    if valid:
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _build_schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
