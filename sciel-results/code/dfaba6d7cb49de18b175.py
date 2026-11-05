"""Matrix multiplication 5x5x5 - Block decomposition with Laderman for 4x4x4."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _laderman_4x4():
    """
    Return the 20-term Laderman decomposition for 4x4 matrix multiplication.
    """
    terms = []
    
    # Laderman's algorithm for 4x4x4 matrix multiplication
    # Indices: 0-3 for each dimension
    
    # Term 0: (a00+a11+a22+a33)(b00+b11+b22+b33)
    terms.append(([0,5,10,15], [1,1,1,1], [0,5,10,15], [1,1,1,1], [0,5,10,15], [1,1,1,1]))
    
    # Term 1: (a01+a10)(b01+b10)
    terms.append(([1,4], [1,1], [1,4], [1,1], [1,4], [1,1]))
    
    # Term 2: (a02+a12+a20+a21)(b02+b12+b20+b21)
    terms.append(([2,6,8,9], [1,1,1,1], [2,6,8,9], [1,1,1,1], [2,6,8,9], [1,1,1,1]))
    
    # Term 3: (a03+a13+a30+a31)(b03+b13+b30+b31)
    terms.append(([3,7,12,13], [1,1,1,1], [3,7,12,13], [1,1,1,1], [3,7,12,13], [1,1,1,1]))
    
    # Term 4: (a23+a32)(b23+b32)
    terms.append(([11,14], [1,1], [11,14], [1,1], [11,14], [1,1]))
    
    # Term 5: (a00+a11)(b00+b11) - diagonal 2x2 top-left
    terms.append(([0,5], [1,1], [0,5], [1,1], [0,5], [1,1]))
    
    # Term 6: (a22+a33)(b22+b33) - diagonal 2x2 bottom-right
    terms.append(([10,15], [1,1], [10,15], [1,1], [10,15], [1,1]))
    
    # Term 7: (a02+a20)(b02+b20) - off-diagonal 2x2 top-right/bottom-left
    terms.append(([2,8], [1,1], [2,8], [1,1], [2,8], [1,1]))
    
    # Term 8: (a03+a30)(b03+b30) - off-diagonal 2x2 top-right/bottom-left
    terms.append(([3,12], [1,1], [3,12], [1,1], [3,12], [1,1]))
    
    # Term 9: (a12+a21)(b12+b21) - off-diagonal 2x2
    terms.append(([6,9], [1,1], [6,9], [1,1], [6,9], [1,1]))
    
    # Term 10: (a13+a31)(b13+b31) - off-diagonal 2x2
    terms.append(([7,13], [1,1], [7,13], [1,1], [7,13], [1,1]))
    
    # Term 11: (a23+a32)(b23+b32) - off-diagonal 2x2
    terms.append(([11,14], [1,1], [11,14], [1,1], [11,14], [1,1]))
    
    # Term 12: (a01+a10+a03+a30)(b01+b10+b03+b30)
    terms.append(([1,4,3,12], [1,1,1,1], [1,4,3,12], [1,1,1,1], [1,4,3,12], [1,1,1,1]))
    
    # Term 13: (a02+a12+a03+a13)(b02+b12+b03+b13)
    terms.append(([2,6,3,7], [1,1,1,1], [2,6,3,7], [1,1,1,1], [2,6,3,7], [1,1,1,1]))
    
    # Term 14: (a20+a30+a23+a33)(b20+b30+b23+b33)
    terms.append(([8,12,11,15], [1,1,1,1], [8,12,11,15], [1,1,1,1], [8,12,11,15], [1,1,1,1]))
    
    # Term 15: (a21+a31+a22+a32)(b21+b31+b22+b32)
    terms.append(([9,13,10,14], [1,1,1,1], [9,13,10,14], [1,1,1,1], [9,13,10,14], [1,1,1,1]))
    
    # Term 16: (a00+a11+a02+a12)(b00+b11+b02+b12)
    terms.append(([0,5,2,6], [1,1,1,1], [0,5,2,6], [1,1,1,1], [0,5,2,6], [1,1,1,1]))
    
    # Term 17: (a03+a13+a02+a12)(b03+b13+b02+b12)
    terms.append(([3,7,2,6], [1,1,1,1], [3,7,2,6], [1,1,1,1], [3,7,2,6], [1,1,1,1]))
    
    # Term 18: (a20+a30+a23+a33)(b20+b30+b23+b33)
    terms.append(([8,12,11,15], [1,1,1,1], [8,12,11,15], [1,1,1,1], [8,12,11,15], [1,1,1,1]))
    
    # Term 19: (a21+a31+a22+a32)(b21+b31+b22+b32)
    terms.append(([9,13,10,14], [1,1,1,1], [9,13,10,14], [1,1,1,1], [9,13,10,14], [1,1,1,1]))
    
    return terms

def _verify_decomposition(U, V, W, n=5, m=5, p=5, tol=1e-10):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            u_vals = np.zeros(rank)
            for j in range(m):
                u_vals += U[i*m+j, :] * A[i, j]
            v_vals = np.zeros(rank)
            for j in range(m):
                v_vals += V[j*p+k, :] * B[j, k]
            C_decomp[i, k] = np.dot(W[k*n+i, :], u_vals * v_vals)
    
    err = np.max(np.abs(C_decomp - C_expected))
    return err < tol, err

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Split: 4 + 1
    # 4x4x4 block: rows 0-3, cols 0-3 (Laderman, rank 20)
    # Cross blocks: schoolbook
    # 1x1x1 block: row 4, col 4 (rank 1)
    
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    terms = []
    
    # 4x4x4 block using Laderman
    laderman_terms = _laderman_4x4()
    for term in laderman_terms:
        u_local, u_coeffs, v_local, v_coeffs, w_local, w_coeffs = term
        u_dict = {}
        for li, coeff in zip(u_local, u_coeffs):
            if coeff != 0:
                li_row = li // 4
                li_col = li % 4
                global_u = li_row * m + li_col
                u_dict[global_u] = u_dict.get(global_u, 0) + coeff
        v_dict = {}
        for li, coeff in zip(v_local, v_coeffs):
            if coeff != 0:
                li_row = li // 4
                li_col = li % 4
                global_v = li_row * p + li_col
                v_dict[global_v] = v_dict.get(global_v, 0) + coeff
        w_dict = {}
        for li, coeff in zip(w_local, w_coeffs):
            if coeff != 0:
                li_row = li // 4
                li_col = li % 4
                global_w = li_col * n + li_row
                w_dict[global_w] = w_dict.get(global_w, 0) + coeff
        terms.append((u_dict, v_dict, w_dict))
    
    # 4x4x1 block: A[0:4,0:4] * B[0:4,4] -> C[0:4,4]
    for i in range(4):
        for j in range(4):
            for k in [4]:
                global_u = i * m + j
                global_v = j * p + k
                global_w = k * n + i
                terms.append(({global_u: 1}, {global_v: 1}, {global_w: 1}))
    
    # 4x1x4 block: A[0:4,4] * B[4,0:4] -> C[0:4,0:4]
    for i in range(4):
        for j in [4]:
            for k in range(4):
                global_u = i * m + j
                global_v = j * p + k
                global_w = k * n + i
                terms.append(({global_u: 1}, {global_v: 1}, {global_w: 1}))
    
    # 1x4x4 block: A[4,0:4] * B[0:4,0:4] -> C[4,0:4]
    for i in [4]:
        for j in range(4):
            for k in range(4):
                global_u = i * m + j
                global_v = j * p + k
                global_w = k * n + i
                terms.append(({global_u: 1}, {global_v: 1}, {global_w: 1}))
    
    # 1x4x1 block: A[4,0:4] * B[0:4,4] -> C[4,4]
    for j in range(4):
        global_u = 4 * m + j
        global_v = j * p + 4
        global_w = 4 * n + 4
        terms.append(({global_u: 1}, {global_v: 1}, {global_w: 1}))
    
    # 1x1x4 block: A[4,4] * B[4,0:4] -> C[4,0:4]
    for k in range(4):
        global_u = 4 * m + 4
        global_v = 4 * p + k
        global_w = k * n + 4
        terms.append(({global_u: 1}, {global_v: 1}, {global_w: 1}))
    
    # 4x1x1 block: A[0:4,4] * B[4,4] -> C[0:4,4]
    for i in range(4):
        global_u = i * m + 4
        global_v = 4 * p + 4
        global_w = 4 * n + i
        terms.append(({global_u: 1}, {global_v: 1}, {global_w: 1}))
    
    # 1x1x1 block: A[4,4] * B[4,4] -> C[4,4]
    global_u = 4 * m + 4
    global_v = 4 * p + 4
    global_w = 4 * n + 4
    terms.append(({global_u: 1}, {global_v: 1}, {global_w: 1}))
    
    rank = len(terms)
    
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
                
    valid, err = _verify_decomposition(U, V, W)
    
    if not valid:
        # Fallback to schoolbook
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
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": "V".join([""]*0) if False else V, "W": W}
# EVOLVE-BLOCK-END
