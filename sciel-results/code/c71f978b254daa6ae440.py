import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100.0

    n, m, p = 2, 2, 16
    blocks = p // 2  # 8 blocks of 2 columns each
    total_rank = blocks * 7  # 56

    # Strassen decomposition for 2x2 x 2x2 -> 2x2
    # U rows: [A00, A01, A10, A11]
    # V rows: [B00, B01, B10, B11]
    # W rows: [C00, C01, C10, C11]
    # Each entry is a list of 7 coefficients (one per Strassen product)
    
    strassen_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1,  0, 1],   # A01
        [0, 1, 0, 0, 0,  1, 0],   # A10
        [1, 1, 0, 1, 0,  0,-1],   # A11
    ], dtype=np.int8)

    strassen_V = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, 1,  0, 0, 1, 0],   # B01
        [0, 0, 0,  1, 0, 0, 1],   # B10
        [1, 0,-1,  0, 1, 0, 1],   # B11
    ], dtype=np.int8)

    strassen_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0,  1, 0, 0],   # C01
        [0, 1, 0, 1,  0, 0, 0],   # C10
        [1,-1, 1, 0,  0, 1, 0],   # C11
    ], dtype=np.int8)

    U = np.zeros((n * m, total_rank), dtype=np.int8)
    V = np.zeros((m * p, total_rank), dtype=np.int8)
    W = np.zeros((p * n, total_rank), dtype=np.int8)

    for q in range(blocks):
        for s in range(7):
            r = q * 7 + s

            # U: same for all blocks (A is shared)
            U[:, r] = strassen_U[:, s]

            # V: B entries for this block
            # B[j, k] maps to row j*p + k in V
            # Block q uses columns 2q and 2q+1
            # B00 -> B[0, 2q] -> row 0*p + 2q = 2q
            # B01 -> B[0, 2q+1] -> row 0*p + 2q+1 = 2q+1
            # B10 -> B[1, 2q] -> row 1*p + 2q = 16+2q
            # B11 -> B[1, 2q+1] -> row 1*p + 2q+1 = 16+2q+1
            b_rows = [2*q, 2*q+1, p + 2*q, p + 2*q + 1]
            for b_idx in range(4):
                V[b_rows[b_idx], r] = strassen_V[b_idx, s]

            # W: C entries for this block
            # C[i, k] maps to row k*n + i in W (trace convention W[k*n+i, r])
            # Block q uses columns 2q and 2q+1
            # C00 -> C[0, 2q] -> row 2q*n + 0 = 4q
            # C01 -> C[0, 2q+1] -> row (2q+1)*n + 0 = 4q+2
            # C10 -> C[1, 2q] -> row 2q*n + 1 = 4q+1
            # C11 -> C[1, 2q+1] -> row (2q+1)*n + 1 = 4q+3
            c_rows = [4*q, 4*q+2, 4*q+1, 4*q+3]
            for c_idx in range(4):
                W[c_rows[c_idx], r] = strassen_W[c_idx, s]

    # Verification
    def verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        rank = U_mat.shape[1]
        # Check that for all valid (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = 0
                    for r in range(rank):
                        val += U_mat[u_idx, r] * V_mat[v_idx, r] * W_mat[w_idx, r]
                    if val != 1:
                        return False
        # Check no spurious terms: for invalid triples, the sum should be 0
        # An (a,b,c) triple is valid if:
        # a = i*m+j, b = j'*p+k, c = k'*n+i' with j=j' and i=i'
        # i.e., a = i*m+j, b = j*p+k, c = k*n+i
        valid_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid_set.add((i*m+j, j*p+k, k*n+i))
        
        # Only check a subset for speed if needed
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) not in valid_set:
                        val = 0
                        for r in range(rank):
                            val += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                        if val != 0:
                            return False
        return True

    # Quick verification with a few random tests
    if time.time() < deadline - 5:
        np.random.seed(123)
        for trial in range(5):
            A = np.random.randint(-3, 3, size=(n, m)).astype(np.int64)
            B = np.random.randint(-3, 3, size=(m, p)).astype(np.int64)
            C_expected = A @ B
            
            Af = A.flatten().astype(np.int64)
            Bf = B.flatten().astype(np.int64)
            
            u_prod = U.T @ Af  # shape (total_rank,)
            v_prod = V.T @ Bf  # shape (total_rank,)
            c_flat = W @ (u_prod * v_prod)  # shape (p*n,)
            
            c_expected_flat = C_expected.T.flatten().astype(np.int64)
            
            if not np.array_equal(c_flat, c_expected_flat):
                # Fallback to schoolbook
                rank_school = n * m * p
                U_fb = np.zeros((n * m, rank_school), dtype=np.int8)
                V_fb = np.zeros((m * p, rank_school), dtype=np.int8)
                W_fb = np.zeros((p * n, rank_school), dtype=np.int8)
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_fb[i * m + j, col] = 1
                            V_fb[j * p + k, col] = 1
                            W_fb[k * n + i, col] = 1
                            col += 1
                return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
