import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = 2, 5, 5
    start_time = time.time()
    timeout = 130  # safety margin before 150s kill
    
    # Build a rank-46 decomposition using Strassen on 2x2 sub-blocks.
    #
    # The tensor is T[i,j,k] = A[i,j] * B[j,k] for i in [0,2), j in [0,5), k in [0,5).
    # We partition the shared dimension j into blocks: [0,1], [2,3], [4]
    # and the output dimension k into blocks: [0,1], [2,3], [4]
    # and the input dimension i into blocks: [0], [1] (since n=2, each is a single row)
    #
    # For each (i_block, j_block, k_block) triple, we multiply a 1x2 (or 2x2) A sub-block
    # with a 2x2 (or 2x1) B sub-block.
    #
    # Actually, since n=2, m=5, p=5:
    # - A is 2x5, B is 5x5, C is 2x5
    # - We can group j into {0,1}, {2,3}, {4} and k into {0,1}, {2,3}, {4}
    # - For each pair of j-block and k-block, and each i, we have a small multiplication.
    #
    # Let's use Strassen where applicable. For a 2x2x2 block (2 rows of A, 2 cols of A = 2 rows of B, 2 cols of B):
    # We can use 7 terms instead of 8.
    #
    # Strategy:
    # Split j into [0,1] and [2,3] and [4]
    # Split k into [0,1] and [2,3] and [4]
    # 
    # For j-block [0,1] and k-block [0,1]: 2x2x2 block -> Strassen (7 terms)
    # For j-block [0,1] and k-block [2,3]: 2x2x2 block -> Strassen (7 terms)
    # For j-block [0,1] and k-block [4]:    2x1x1 block -> schoolbook (2 terms)
    # For j-block [2,3] and k-block [0,1]: 2x2x2 block -> Strassen (7 terms)
    # For j-block [2,3] and k-block [2,3]: 2x2x2 block -> Strassen (7 terms)
    # For j-block [2,3] and k-block [4]:    2x1x1 block -> schoolbook (2 terms)
    # For j-block [4] and k-block [0,1]:    1x2x2 block -> schoolbook (2 terms)
    # For j-block [4] and k-block [2,3]:    1x2x2 block -> schoolbook (2 terms)
    # For j-block [4] and k-block [4]:      1x1x1 block -> schoolbook (1 term)
    #
    # Total: 7+7+2+7+7+2+2+2+1 = 37? Wait, let me recount.
    #
    # Actually for n=2, each "row" of A is separate. Let me think more carefully.
    # A[i, j] * B[j, k] summed over j.
    # For fixed i, the product is a row of A (length 5) times B (5x5) = row of C (length 5).
    # So it's a (1x5) * (5x5) = (1x5) multiplication, repeated for 2 values of i.
    #
    # Let me use a different approach. Split j into [0,1], [2,3], [4] and k into [0,1], [2,3], [4].
    # For each i in {0,1}, j-block jb, k-block kb:
    #   The contribution is A[i, jb] * B[jb, kb] where A[i,jb] is a row vector and B[jb,kb] is a matrix.
    #   This is a (1 x |jb|) * (|jb| x |kb|) = (1 x |kb|) product.
    #   Number of schoolbook terms = |jb| * |kb|.
    #
    # But we can apply Strassen when |jb|=2 and |kb|=2 (and we have the right structure).
    #
    # Actually, for a 1x2 * 2x2 multiplication (1 row of A, 2x2 block of B), we can't directly use Strassen
    # because Strassen requires 2x2 * 2x2. But we can replicate the row.
    #
    # Let me just use schoolbook for simplicity and see if we can get below 50.
    # Schoolbook: 2*5*5 = 50.
    #
    # With Strassen on 2x2x2 blocks:
    # When we have A[2,2] * B[2,2] = C[2,2], we use 7 terms instead of 8.
    # 
    # Let me partition:
    # j into [0,1], [2,3], [4]
    # k into [0,1], [2,3], [4]
    # 
    # For i in {0,1}:
    #   For each j-block jb and k-block kb:
    #     If |jb|=2 and |kb|=2: we have A[i, jb] (1x2) * B[jb, kb] (2x2) -> 1x2 result
    #       This is not a 2x2x2 problem. We need 2 rows of A to use Strassen.
    #
    # Since n=2, we can group both rows of A together for 2x2x2 blocks.
    # For j-block [0,1] and k-block [0,1]:
    #   A[:, [0,1]] is 2x2, B[[0,1], [0,1]] is 2x2, C[:, [0,1]] is 2x2.
    #   Use Strassen: 7 terms.
    #
    # For j-block [0,1] and k-block [2,3]:
    #   A[:, [0,1]] is 2x2, B[[0,1], [2,3]] is 2x2, C[:, [2,3]] is 2x2.
    #   Use Strassen: 7 terms.
    #
    # For j-block [0,1] and k-block [4]:
    #   A[:, [0,1]] is 2x2, B[[0,1], [4]] is 2x1, C[:, [4]] is 2x1.
    #   Schoolbook: 2*1*1 = 2 terms? No, it's 2x2 * 2x1 = 2x1, which is 2*2*1 = 4 terms.
    #   But we can do it with 2 terms if we're clever? Actually 2x2 * 2x1 requires 4 multiplications in schoolbook.
    #
    # Hmm, let me just count schoolbook terms for each block:
    # (2x2)*(2x2): 8 schoolbook, 7 Strassen
    # (2x2)*(2x1): 4 schoolbook
    # (2x2)*(2x1): 4 schoolbook
    # (2x2)*(2x2): 8 schoolbook, 7 Strassen
    # (2x2)*(2x2): 8 schoolbook, 7 Strassen
    # (2x2)*(2x1): 4 schoolbook
    # (2x1)*(2x2): 4 schoolbook
    # (2x1)*(2x2): 4 schoolbook
    # (2x1)*(2x1): 2 schoolbook
    #
    # Total schoolbook: 8+4+4+8+8+4+4+4+2 = 46
    # With Strassen on the three 2x2x2 blocks: 7+4+4+7+7+4+4+4+2 = 43
    #
    # Wait, that's only 43. Let me verify.
    # j-blocks: [0,1], [2,3], [4]
    # k-blocks: [0,1], [2,3], [4]
    # 
    # (jb=[0,1], kb=[0,1]): A[2x2] * B[2x2] -> C[2x2]: 8 schoolbook, 7 Strassen
    # (jb=[0,1], kb=[2,3]): A[2x2] * B[2x2] -> C[2x2]: 8 schoolbook, 7 Strassen
    # (jb=[0,1], kb=[4]):    A[2x2] * B[2x1] -> C[2x1]: 4 schoolbook
    # (jb=[2,3], kb=[0,1]): A[2x2] * B[2x2] -> C[2x2]: 8 schoolbook, 7 Strassen
    # (jb=[2,3], kb=[2,3]): A[2x2] * B[2x2] -> C[2x2]: 8 schoolbook, 7 Strassen
    # (jb=[2,3], kb=[4]):    A[2x2] * B[2x1] -> C[2x1]: 4 schoolbook
    # (jb=[4],    kb=[0,1]): A[2x1] * B[1x2] -> C[2x2]: 4 schoolbook
    # (jb=[4],    kb=[2,3]): A[2x1] * B[1x2] -> C[2x2]: 4 schoolbook
    # (jb=[4],    kb=[4]):    A[2x1] * B[1x1] -> C[2x1]: 2 schoolbook
    #
    # Schoolbook total: 8+8+4+8+8+4+4+4+2 = 50. Correct.
    # Strassen on three 2x2x2: 7+7+4+7+7+4+4+4+2 = 46.
    #
    # So rank 46 is achievable. Let me construct it.
    
    # Strassen vectors for 2x2*2x2=2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # U vectors (over A's 4 entries: a,b,c,d):
    # M1: [1, 0, 0, 1]  (a+d)
    # M2: [0, 0, 1, 1]  (c+d)
    # M3: [1, 0, 0, 0]  (a)
    # M4: [0, 0, 0, 1]  (d)
    # M5: [1, 1, 0, 0]  (a+b)
    # M6: [1, 0, -1, 0] (c-a)
    # M7: [0, 1, 0, -1] (b-d)
    #
    # V vectors (over B's 4 entries: e,f,g,h):
    # M1: [1, 1, 0, 0]  (e+h)
    # M2: [1, 0, 0, 0]  (e)
    # M3: [0, 1, 0, -1] (f-h)
    # M4: [0, -1, 1, 0] (g-e)
    # M5: [0, 0, 0, 1]  (h)
    # M6: [1, 1, 0, 0]  (e+f)
    # M7: [0, 0, 1, 1]  (g+h)
    #
    # W vectors (over C's 4 entries: C11, C12, C21, C22):
    # M1: [1, 0, 0, 1]   (C11 + C22)
    # M2: [0, 0, 1, -1]  (C21 - C22)
    # M3: [0, 1, 0, 1]   (C12 + C22)
    # M4: [1, 0, 1, 0]   (C11 + C21)
    # M5: [-1, 1, 0, 0]  (-C11 + C12)
    # M6: [0, 0, 0, 1]   (C22)
    # M7: [1, 0, 0, 0]   (C11)
    
    strassen_U = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [1, 0, -1, 0],
        [0, 1, 0, -1],
    ], dtype=float)
    
    strassen_V = np.array([
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, -1, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ], dtype=float)
    
    strassen_W = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ], dtype=float)
    
    # Build the full decomposition
    terms = []  # list of (u_vec, v_vec, w_vec) where each is over the full index set
    
    U_dim = n * m  # 2*5 = 10
    V_dim = m * p  # 5*5 = 25
    W_dim = p * n  # 5*2 = 10
    
    def make_zero_vec(dim):
        return np.zeros(dim, dtype=float)
    
    # j-blocks and k-blocks
    j_blocks = [[0, 1], [2, 3], [4]]
    k_blocks = [[0, 1], [2, 3], [4]]
    
    for jb in j_blocks:
        for kb in k_blocks:
            jb_size = len(jb)
            kb_size = len(kb)
            
            if jb_size == 2 and kb_size == 2:
                # Use Strassen: 7 terms
                # A sub-block: rows 0,1 of A, cols jb[0], jb[1] -> 2x2
                # B sub-block: rows jb[0], jb[1], cols kb[0], kb[1] -> 2x2
                # C sub-block: rows 0,1 of C, cols kb[0], kb[1] -> 2x2
                for t in range(7):
                    u_vec = make_zero_vec(U_dim)
                    v_vec = make_zero_vec(V_dim)
                    w_vec = make_zero_vec(W_dim)
                    
                    # U: over (i, j) where i in {0,1}, j in jb
                    # Local A indices: (0, jb[0]) -> a, (0, jb[1]) -> b, (1, jb[0]) -> c, (1, jb[1]) -> d
                    u_local = strassen_U[t]
                    u_vec[0 * m + jb[0]] = u_local[0]  # a
                    u_vec[0 * m + jb[1]] = u_local[1]  # b
                    u_vec[1 * m + jb[0]] = u_local[2]  # c
                    u_vec[1 * m + jb[1]] = u_local[3]  # d
                    
                    # V: over (j, k) where j in jb, k in kb
                    # Local B indices: (jb[0], kb[0]) -> e, (jb[0], kb[1]) -> f, (jb[1], kb[0]) -> g, (jb[1], kb[1]) -> h
                    v_local = strassen_V[t]
                    v_vec[jb[0] * p + kb[0]] = v_local[0]  # e
                    v_vec[jb[0] * p + kb[1]] = v_local[1]  # f
                    v_vec[jb[1] * p + kb[0]] = v_local[2]  # g
                    v_vec[jb[1] * p + kb[1]] = v_local[3]  # h
                    
                    # W: over (k, i) where k in kb, i in {0,1}
                    # Local C indices: (kb[0], 0) -> C11, (kb[1], 0) -> C12, (kb[0], 1) -> C21, (kb[1], 1) -> C22
                    w_local = strassen_W[t]
                    w_vec[kb[0] * n + 0] = w_local[0]  # C11
                    w_vec[kb[1] * n + 0] = w_local[1]  # C12
                    w_vec[kb[0] * n + 1] = w_local[2]  # C21
                    w_vec[kb[1] * n + 1] = w_local[3]  # C22
                    
                    terms.append((u_vec, v_vec, w_vec))
            else:
                # Schoolbook: jb_size * kb_size terms
                for jj in jb:
                    for kk in kb:
                        for ii in range(n):
                            u_vec = make_zero_vec(U_dim)
                            v_vec = make_zero_vec(V_dim)
                            w_vec = make_zero_vec(W_dim)
                            u_vec[ii * m + jj] = 1
                            v_vec[jj * p + kk] = 1
                            w_vec[kk * n + ii] = 1
                            terms.append((u_vec, v_vec, w_vec))
    
    rank = len(terms)
    
    # Build U, V, W matrices
    U = np.zeros((U_dim, rank), dtype=float)
    V = np.zeros((V_dim, rank), dtype=float)
    W = np.zeros((W_dim, rank), dtype=float)
    
    for t, (u_vec, v_vec, w_vec) in enumerate(terms):
        U[:, t] = u_vec
        V[:, t] = v_vec
        W[:, t] = w_vec
    
    # Verify
    def check_decomposition(U, V, W, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for t in range(U.shape[1]):
                        val += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    if not check_decomposition(U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((U_dim, rank), dtype=float)
        V = np.zeros((V_dim, rank), dtype=float)
        W = np.zeros((W_dim, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    # Convert to lists for the required output format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Snap to half-integers
    def snap_to_half_int(mat):
        return [[round(v * 2) / 2.0 for v in row] for row in mat]
    
    U_list = snap_to_half_int(U_list)
    V_list = snap_to_half_int(V_list)
    W_list = snap_to_half_int(W_list)
    
    return {"U": U_list, "V": V_list, "W": W_list}
