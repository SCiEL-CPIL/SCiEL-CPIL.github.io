"""Matrix multiplication 4,4,8 - rank-minimal exact tensor decomposition using Strassen."""
import time
DIMENSIONS = [4, 4, 8]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 110  # Safety margin before 120s limit

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    if time.time() > _DEADLINE:
        # Fallback: return schoolbook rank 128
        U = [[0]*128 for _ in range(16)]
        V = [[0]*128 for _ in range(32)]
        W = [[0]*128 for _ in range(32)]
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    r = i*32 + j*8 + k
                    U[i*4+j][r] = 1
                    V[j*8+k][r] = 1
                    W[k*4+i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    # Strategy: Strassen at 4x4 block level (7 products of 2x2x2),
    # each 2x2x2 uses schoolbook (rank 4). Total: 7*4 = 28 for 4x4x4.
    # For 4x4x8: rank 56 (two copies, sharing U structure).
    # 
    # 4x4x4 via Strassen at block level:
    # A = [[A00, A01], [A10, A11]] (2x2 blocks of 2x2)
    # B = [[B00, B01], [B10, B11]]
    # 
    # Strassen products (each is a 2x2 matrix product):
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B10-B11)
    # S4 = A11(B01-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    # 
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    # 
    # Each S_r is a 2x2 matrix product. For each, we need a rank-4 schoolbook
    # decomposition of 2x2x2.
    # 
    # 2x2x2 schoolbook: C[i,k] = sum_j A[i,j]*B[j,k]
    # 4 multiplications: (i,j,k) with the constraint that each term is A[i,j]*B[j,k]
    # But actually there are 8 terms in schoolbook... 
    # 
    # Wait, I need to think about this differently. The 2x2x2 tensor has rank 7.
    # There is no rank-4 exact decomposition. The "schoolbook" uses 4 scalar 
    # multiplications in the sense of the algorithm, but as a tensor decomposition,
    # it requires rank 8 (one for each of the 8 terms A[i,j]*B[j,k] contributing 
    # to C[i,k]).
    #
    # Actually no. In a tensor decomposition, each rank-1 term is:
    # sum over r of U[i,j,r] * V[j,k,r] * W[k,i,r]
    # 
    # For schoolbook: each term is A[i,j] * B[j,k] which contributes to C[i,k].
    # This is rank-1 with U[i,j,r]=1 if (i,j) matches, V[j,k,r]=1 if (j,k) matches,
    # W[k,i,r]=1 if (k,i) matches. But we need ALL (i,j) for a fixed r to have the 
    # same j... 
    #
    # The schoolbook decomposition of 2x2x2 actually has rank 8 (one term per 
    # (i,j,k) triple where the term is A[i,j]*B[j,k] -> C[i,k]). But these 8 terms
    # can be grouped: for a fixed j, the terms A[i,j]*B[j,k] for all i,k form a 
    # rank-4 sub-tensor (outer product of A[:,j] and B[j,:]). So rank is 2*4=8.
    #
    # Hmm, this means using schoolbook for the inner 2x2x2 gives rank 8, not 4.
    # So total would be 7*8 = 56 for 4x4x4, and 112 for 4x4x8.
    #
    # That's worse than the current 98! Let me reconsider.
    #
    # The current approach (2-level Strassen) gives 7*7=49 for 4x4x4 and 98 for 4x4x8.
    # 
    # Can we do better? The known upper bound for 4x4x4 is 28.
    # 
    # Let me try a different approach: use Strassen directly on the 4x4x4 tensor
    # by viewing it as a 2x2 block product where each block is 2x2, but use 
    # rank-7 Strassen for each 2x2x2 block product.
    # That gives 7*7 = 49. Same as current.
    #
    # For 4x4x8, the current rank is 98. Let me see if we can reduce it.
    # 
    # Alternative: view 4x4x8 as 2x2 blocks in i (rows of A) and k (cols of B):
    # A is 4x4, B is 4x8.
    # i = (i1, i0), k = (k1, k0) where k1 in {0,1} (left/right half of B)
    # 
    # C[i1,i0,k1,k0] = sum_j A[i1,i0,j] * B[j,k1,k0]
    # 
    # Use Strassen in the (i1,k1) block structure:
    # A_top = A[0:2,:], A_bot = A[2:4,:]
    # B_left = B[:,0:4], B_right = B[:,4:8]
    # C_tl = A_top @ B_left, C_tr = A_top @ B_right
    # C_bl = A_bot @ B_left, C_br = A_bot @ B_right
    # 
    # Strassen:
    # S1 = (A_top + A_bot)(B_left + B_right)
    # S2 = (A_bot - A_top)B_left
    # S3 = A_top(B_right - B_left)
    # S4 = (A_top + A_bot)B_right  -- wait, standard Strassen has 7 products
    # 
    # Standard Strassen for 2x2 blocks:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # 
    # Here A00=A_top, A01=A_top (same row block), A10=A_bot, A11=A_bot
    # B00=B_left, B01=B_right, B10=B_left, B11=B_right
    # 
    # M1 = (A_top+A_bot)(B_left+B_right)
    # M2 = (A_bot+A_bot)B_left = 2*A_bot*B_left
    # M3 = A_top(B_left-B_right)
    # M4 = A_bot(B_right-B_left)
    # M5 = (A_top+A_top)B_right = 2*A_top*B_right
    # M6 = (A_bot-A_top)(B_left+B_right)
    # M7 = (A_top-A_bot)(B_left+B_right)
    # 
    # C_top_left = M1 + M4 - M5 + M7
    # C_top_right = M3 + M5
    # C_bot_left = M2 + M4
    # C_bot_right = M1 - M2 + M3 + M6
    # 
    # Each M is a (2x4) @ (4x4) product = 2x4 result.
    # The tensor for (2x4)@(4x4)->(2x4) has dimensions n=2, m=4, p=4.
    # Rank of this? It's the 2x4x4 matmul tensor.
    # 
    # For 2x4x4: C[i,k] = sum_j A[i,j]*B[j,k] with i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # This is a rank-? tensor. The trivial bound is 2*4*4/4 = 8? No.
    # The rank is at most 2*4 = 8 (for each i, we have 4 multiplications A[i,j]*B[j,k]).
    # Actually, it's the direct sum of two 1x4x4 tensors, each of rank 4.
    # So rank(2x4x4) = 8.
    # 
    # Total: 7 * 8 = 56 for 4x4x8! That's better than 98!
    #
    # Let me verify: 7 Strassen products, each is (2x4)@(4x4)->(2x4) with rank 8.
    # Total rank = 7*8 = 56.
    #
    # Let me implement this.

    def build_2x4x4_schoolbook():
        """Schoolbook decomposition for (2x4) @ (4x4) -> (2x4). Rank 8.
        U[i*4+j, r], V[j*4+k, r], W[k*2+i, r]
        n=2, m=4, p=4
        """
        n, m, p = 2, 4, 4
        rank = n * m  # 8
        U = [[0]*rank for _ in range(n*m)]  # (8, 8)
        V = [[0]*rank for _ in range(m*p)]  # (16, 8)
        W = [[0]*rank for _ in range(p*n)]  # (8, 8)
        
        for i in range(n):
            for j in range(m):
                r = i*m + j
                U[i*m + j][r] = 1
                for k in range(p):
                    V[j*p + k][r] = 1
                    W[k*n + i][r] = 1
        return U, V, W
    
    U244, V244, W244 = build_2x4x4_schoolbook()
    # U244: (8, 8), V244: (16, 8), W244: (8, 8)
    
    # Now build the 4x4x8 decomposition using Strassen at the (i,k) block level.
    # A is 4x4: A_top = A[0:2, 0:4], A_bot = A[2:4, 0:4]
    # B is 4x8: B_left = B[0:4, 0:4], B_right = B[0:4, 4:8]
    # C is 4x8: C_top_left = C[0:2, 0:4], C_top_right = C[0:2, 4:8]
    #             C_bot_left = C[2:4, 0:4], C_bot_right = C[2:4, 4:8]
    #
    # Strassen products (each is (2x4)@(4x4)->(2x4)):
    # M1 = (A_top+A_bot)(B_left+B_right)
    # M2 = 2*A_bot*B_left
    # M3 = A_top*(B_left-B_right)
    # M4 = A_bot*(B_right-B_left)
    # M5 = 2*A_top*B_right
    # M6 = (A_bot-A_top)*(B_left+B_right)
    # M7 = (A_top-A_bot)*(B_left+B_right)
    #
    # C_top_left = M1 + M4 - M5 + M7
    # C_top_right = M3 + M5
    # C_bot_left = M2 + M4
    # C_bot_right = M1 - M2 + M3 + M6
    #
    # For each M_r, we use the 2x4x4 schoolbook (rank 8).
    # Total rank: 7*8 = 56.
    #
    # U for 4x4x8: shape (16, 56). U[i*4+j, r]
    #   For M1: U_top[i,j] += 1, U_bot[i,j] += 1 (i in 0,1 for top/bot)
    #   Actually, let me be more careful.
    #
    # A_top[i,j] = A[i,j] for i=0,1; A_bot[i,j] = A[i+2,j] for i=0,1
    # 
    # For M1 = (A_top+A_bot)(B_left+B_right):
    #   A-part: A_top[i,j] + A_bot[i,j] = A[i,j] + A[i+2,j]
    #   B-part: B_left[j,k] + B_right[j,k] = B[j,k] + B[j,k+4]
    #   C-part: contributes to C_top_left and C_bot_right (via the Strassen formula)
    #
    # Let me define the rank-1 terms. For each Strassen product r (0..6) and each
    # schoolbook term s (0..7), we have rank index R = r*8 + s.
    #
    # For schoolbook term s = i_s * 4 + j_s (i_s in {0,1}, j_s in {0,1,2,3}):
    #   The 2x4x4 decomposition gives:
    #   U_244[i_s*4+j_s, s] = 1
    #   V_244[j_s*4+k, s] = 1 for all k
    #   W_244[k*2+i_s, s] = 1 for all k
    #
    # Now for the full 4x4x8:
    #   U[i*4+j, R] depends on the Strassen formula for the A-part
    #   V[j*8+k, R] depends on the Strassen formula for the B-part
    #   W[k*4+i, R] depends on the Strassen formula for the C-part
    #
    # Let me define the Strassen coefficients:
    # For each M_r, we have:
    #   A_coeff[r][i_block][i_local, j] where i_block in {top, bot}
    #   B_coeff[r][j, k_block][k_local] where k_block in {left, right}
    #   C_coeff[r][i_block][i_local, k_block][k_local]
    
    # A coefficients for each Strassen product (for the 2x4 A-matrix):
    # M1: A_top + A_bot -> coeff: top=+1, bot=+1
    # M2: 2*A_bot -> coeff: top=0, bot=+2
    # M3: A_top -> coeff: top=+1, bot=0
    # M4: A_bot -> coeff: top=0, bot=+1
    # M5: 2*A_top -> coeff: top=+2, bot=0
    # M6: A_bot - A_top -> coeff: top=-1, bot=+1
    # M7: A_top - A_bot -> coeff: top=+1, bot=-1
    
    A_coeff = [
        [1, 1],   # M1: top=1, bot=1
        [0, 2],   # M2: top=0, bot=2
        [1, 0],   # M3: top=1, bot=0
        [0, 1],   # M4: top=0, bot=1
        [2, 0],   # M5: top=2, bot=0
        [-1, 1],  # M6: top=-1, bot=1
        [1, -1],  # M7: top=1, bot=-1
    ]
    
    # B coefficients for each Strassen product (for the 4x4 B-matrix, left/right):
    # M1: B_left + B_right -> left=+1, right=+1
    # M2: B_left -> left=+1, right=0
    # M3: B_left - B_right -> left=+1, right=-1
    # M4: B_right - B_left -> left=-1, right=+1
    # M5: B_right -> left=0, right=+1
    # M6: B_left + B_right -> left=+1, right=+1
    # M7: B_left + B_right -> left=+1, right=+1
    
    B_coeff = [
        [1, 1],   # M1: left=1, right=1
        [1, 0],   # M2: left=1, right=0
        [1, -1],  # M3: left=1, right=-1
        [-1, 1],  # M4: left=-1, right=1
        [0, 1],   # M5: left=0, right=1
        [1, 1],   # M6: left=1, right=1
        [1, 1],   # M7: left=1, right=1
    ]
    
    # C coefficients: which C blocks each M_r contributes to, and with what sign
    # C_top_left = M1 + M4 - M5 + M7
    # C_top_right = M3 + M5
    # C_bot_left = M2 + M4
    # C_bot_right = M1 - M2 + M3 + M6
    #
    # C_coeff[r][c_block] where c_block in {tl, tr, bl, br}
    C_coeff = [
        [1, 0, 0, 1],   # M1: tl=+1, tr=0, bl=0, br=+1
        [0, 0, 1, -1],  # M2: tl=0, tr=0, bl=+1, br=-1
        [0, 1, 0, 1],   # M3: tl=0, tr=+1, bl=0, br=+1
        [1, 0, 1, 0],   # M4: tl=+1, tr=0, bl=+1, br=0
        [-1, 1, 0, 0],  # M5: tl=-1, tr=+1, bl=0, br=0
        [0, 0, 0, 1],   # M6: tl=0, tr=0, bl=0, br=+1
        [1, 0, 0, 0],   # M7: tl=+1, tr=0, bl=0, br=0
    ]
    
    rank = 56
    U = [[0]*rank for _ in range(16)]   # (16, 56)
    V = [[0]*rank for _ in range(32)]   # (32, 56)
    W = [[0]*rank for _ in range(32)]   # (32, 56)
    
    for r in range(7):
        a_top = A_coeff[r][0]
        a_bot = A_coeff[r][1]
        b_left = B_coeff[r][0]
        b_right = B_coeff[r][1]
        c_tl = C_coeff[r][0]
        c_tr = C_coeff[r][1]
        c_bl = C_coeff[r][2]
        c_br = C_coeff[r][3]
        
        for s in range(8):
            R = r * 8 + s
            i_s = s // 4  # 0 or 1 (local row in the 2x4 block)
            j_s = s % 4   # 0..3 (column)
            
            # U: A[i,j] contribution
            # For top block: i_global = i_s, for bot block: i_global = 2 + i_s
            if a_top != 0:
                U[i_s * 4 + j_s][R] += a_top
            if a_bot != 0:
                U[(2 + i_s) * 4 + j_s][R] += a_bot
            
            # V: B[j,k] contribution
            # B is 4x8. B[j, k] where k=0..3 is left, k=4..7 is right.
            # For the 2x4x4 schoolbook, V_244[j_s*4+k, s] = 1 for k=0..3
            # In the full B: B[j_s, k] for k=0..3 (left) and B[j_s, k+4] for k=4..7 (right)
            for k in range(4):
                if b_left != 0:
                    V[j_s * 8 + k][R] += b_left
                if b_right != 0:
                    V[j_s * 8 + (4 + k)][R] += b_right
            
            # W: C[i,k] contribution
            # C is 4x8. C[i, k] where i=0,1 is top, i=2,3 is bot.
            # k=0..3 is left, k=4..7 is right.
            # For the 2x4x4 schoolbook, W_244[k*2+i_s, s] = 1 for k=0..3
            # In the full C: 
            #   C_top_left[i_s, k]: i_global=i_s, k_global=k
            #   C_top_right[i_s, k]: i_global=i_s, k_global=4+k
            #   C_bot_left[i_s, k]: i_global=2+i_s, k_global=k
            #   C_bot_right[i_s, k]: i_global=2+i_s, k_global=4+k
            for k in range(4):
                if c_tl != 0:
                    W[k * 4 + i_s][R] += c_tl
                if c_tr != 0:
                    W[(4 + k) * 4 + i_s][R] += c_tr
                if c_bl != 0:
                    W[k * 4 + (2 + i_s)][R] += c_bl
                if c_br != 0:
                    W[(4 + k) * 4 + (2 + i_s)][R] += c_br
    
    # Verify the decomposition (only if time permits)
    if time.time() < _DEADLINE - 5:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: for all i, j, k: sum_r U[i*4+j, r] * V[j*8+k, r] * W[k*4+i, r] == delta(j appears in both)
        # Actually, the matmul tensor: C[i,k] = sum_j A[i,j]*B[j,k]
        # So sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 if j is the 
        # "matching" index, i.e., it should reconstruct the matmul operation.
        # 
        # More precisely: for the decomposition to be correct, we need:
        # For all i, k: C[i,k] = sum_j A[i,j]*B[j,k] = sum_r (sum_j U[i*4+j,r]*A[i,j]) * (sum_j V[j*8+k,r]*B[j,k]) * W[k*4+i,r]
        # 
        # This is equivalent to: the tensor T[i,j,k] = delta (the matmul tensor) satisfies:
        # T[i,j,k] = sum_r U[i*4+j,r] * V[j*8+k,r] * W[k*4+i,r]
        # where T[i,j,k] = 1 if j is the "inner" index connecting i and k, else 0.
        # Actually T[i,j,k] is the standard matmul tensor: T[i,j,k] = 1 always (it's 
        # the structural tensor), and the decomposition means:
        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = delta_{i,j,k} 
        # where delta is 1 when the indices form a valid matmul term.
        
        # Let me just check with random matrices.
        np.random.seed(42)
        A_test = np.random.randn(4, 4)
        B_test = np.random.randn(4, 8)
        C_expected = A_test @ B_test
        
        # Compute C from decomposition
        C_decomp = np.zeros((4, 8))
        for r in range(rank):
            # u_vec = U[:, r] -> shape (16,) = (4,4)
            u_mat = U_np[:, r].reshape(4, 4)
            # v_vec = V[:, r] -> shape (32,) = (4,8)
            v_mat = V_np[:, r].reshape(4, 8)
            # w_vec = W[:, r] -> shape (32,) = (8,4)
            w_val = W_np[:, r]  # (32,) indexed by k*4+i
            
            # C[i,k] += sum_j u_mat[i,j] * v_mat[j,k] * w_val[k*4+i]
            # This is not a simple matrix product, so let's do it element-wise
            for i in range(4):
                for k in range(8):
                    s = 0.0
                    for j in range(4):
                        s += u_mat[i, j] * v_mat[j, k] * w_val[k*4+i]
                    C_decomp[i, k] += s
        
        max_err = np.max(np.abs(C_decomp - C_expected))
        if max_err > 1e-9:
            print(f"WARNING: Verification failed with max error {max_err}")
            # Fallback to schoolbook
            U_fb = [[0]*128 for _ in range(16)]
            V_fb = [[0]*128 for _ in range(32)]
            W_fb = [[0]*128 for _ in range(32)]
            for i in range(4):
                for j in range(4):
                    for k in range(8):
                        r = i*32 + j*8 + k
                        U_fb[i*4+j][r] = 1
                        V_fb[j*8+k][r] = 1
                        W_fb[k*4+i][r] = 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
