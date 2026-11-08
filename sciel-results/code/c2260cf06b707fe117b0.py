"""Matrix multiplication 2,4,5 tensor decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # A is 2x4, B is 4x5, C is 2x5
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (10, rank)
    
    start_time = time.time()
    timeout = 110.0
    
    # We use a block decomposition:
    # A = [A_L | A_R] where A_L, A_R are 2x2
    # B = [B_1; B_2] where B_1, B_2 are 2x5
    # C = A_L @ B_1 + A_R @ B_2
    #
    # For each 2x2 @ 2x5 block, we decompose into:
    #   - Two 2x2 @ 2x2 (Strassen, rank 7 each)
    #   - One 2x2 @ 2x1 (rank 4)
    # Total per block: 18
    # Two blocks: 36
    #
    # However, we can potentially merge the two 2x2 @ 2x1 parts.
    # For A_L @ B_1[:,4] and A_R @ B_2[:,4]:
    #   C[:,4] = A_L @ B_1[:,4] + A_R @ B_2[:,4]
    #
    # Let's try to find a rank-35 or lower solution.
    # For now, let's construct the rank-36 solution and verify it.
    
    rank = 36
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    def u_idx(i, j): return i*m + j
    def v_idx(j, k): return j*p + k
    def w_idx(k, i): return k*n + i
    
    col = 0
    
    def add_strassen_2x2x2(ai0, ai1, aj0, aj1, bj0, bj1, bk0, bk1):
        nonlocal col
        a00 = u_idx(ai0, aj0)
        a01 = u_idx(ai0, aj1)
        a10 = u_idx(ai1, aj0)
        a11 = u_idx(ai1, aj1)
        b00 = v_idx(bj0, bk0)
        b01 = v_idx(bj0, bk1)
        b10 = v_idx(bj1, bk0)
        b11 = v_idx(bj1, bk1)
        c00 = w_idx(bk0, ai0)
        c01 = w_idx(bk1, ai0)
        c10 = w_idx(bk0, ai1)
        c11 = w_idx(bk1, ai1)
        
        # M1 = (A00+A11)(B00+B11)
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1; W[c11, col] -= 1
        col += 1
        
        # M3 = A00(B01-B11)
        U[a00, col] += 1
        V[b01, col] += 1; V[b11, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[a11, col] += 1
        V[b10, col] += 1; V[b00, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] -= 1; W[c01, col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[a10, col] += 1; U[a00, col] -= 1
        V[b00, col] += 1; V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a01, col] += 1; U[a11, col] -= 1
        V[b10, col] += 1; V[b11, col] += 1
        W[c00, col] += 1
        col += 1
    
    def add_vec_mat_2x2x1(ai0, ai1, aj0, aj1, bj0, bj1, bk):
        nonlocal col
        a00 = u_idx(ai0, aj0)
        a01 = u_idx(ai0, aj1)
        a10 = u_idx(ai1, aj0)
        a11 = u_idx(ai1, aj1)
        b0 = v_idx(bj0, bk)
        b1 = v_idx(bj1, bk)
        c0 = w_idx(bk, ai0)
        c1 = w_idx(bk, ai1)
        
        U[a00, col] += 1; V[b0, col] += 1; W[c0, col] += 1; col += 1
        U[a01, col] += 1; V[b1, col] += 1; W[c0, col] += 1; col += 1
        U[a10, col] += 1; V[b0, col] += 1; W[c1, col] += 1; col += 1
        U[a11, col] += 1; V[b1, col] += 1; W[c1, col] += 1; col += 1
    
    # Block 1: A_L @ B_11
    add_strassen_2x2x2(0, 1, 0, 1, 0, 1, 0, 1)
    # Block 2: A_L @ B_12
    add_strassen_2x2x2(0, 1, 0, 1, 0, 1, 2, 3)
    # Block 3: A_L @ B_13
    add_vec_mat_2x2x1(0, 1, 0, 1, 0, 1, 4)
    # Block 4: A_R @ B_21
    add_strassen_2x2x2(0, 1, 2, 3, 2, 3, 0, 1)
    # Block 5: A_R @ B_22
    add_strassen_2x2x2(0, 1, 2, 3, 2, 3, 2, 3)
    # Block 6: A_R @ B_23
    add_vec_mat_2x2x1(0, 1, 2, 3, 2, 3, 4)
    
    # Verify: C[k,i] = sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
    # Let's do a quick sanity check with random inputs
    if time.time() - start_time < timeout:
        np.random.seed(42)
        A_test = np.random.randint(-2, 3, size=(n, m))
        B_test = np.random.randint(-2, 3, size=(m, p))
        C_direct = A_test @ B_test
        
        # Compute via decomposition
        C_decomp = np.zeros((p, n), dtype=int)
        for l in range(rank):
            u_l = U[:, l]
            v_l = V[:, l]
            w_l = W[:, l]
            # u_l is (8,), v_l is (20,), w_l is (10,)
            # u_l[i*m+j] * v_l[j*p+k] * w_l[k*n+i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        C_decomp[k, i] += u_l[i*m+j] * v_l[j*p+k] * w_l[k*n+i]
        
        # C_decomp[k,i] should equal C_direct[i,k]
        C_direct_transposed = C_direct.T  # shape (p, n)
        if not np.array_equal(C_decomp, C_direct_transposed):
            # Fallback: return direct multiplication rank
            rank_direct = n * m * p  # 40
            U2 = np.zeros((n*m, rank_direct), dtype=int)
            V2 = np.zeros((m*p, rank_direct), dtype=int)
            W2 = np.zeros((p*n, rank_direct), dtype=int)
            col2 = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U2[i*m+j, col2] = 1
                        V2[j*p+k, col2] = 1
                        W2[k*n+i, col2] = 1
                        col2 += 1
            return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
