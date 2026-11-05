import numpy as np
import time
import random

_START = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill
n, m, p = 4, 5, 5

def construct():
    # We'll build a correct decomposition with rank 85 (Strassen on 4x4 + schoolbook for column/row 4)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    def add_term(i, k, j, coeff_u=1.0, coeff_v=1.0, coeff_w=1.0):
        """
        Adds a term: coeff_u * A[i,k] * coeff_v * B[k,j] -> coeff_w * C[i,j]
        u_vec has coeff_u at i*m+k
        v_vec has coeff_v at k*p+j
        w_vec has coeff_w at j*n+i
        """
        u_vec = np.zeros(n * m)
        v_vec = np.zeros(m * p)
        w_vec = np.zeros(p * n)
        
        u_vec[i * m + k] = coeff_u
        v_vec[k * p + j] = coeff_v
        w_vec[j * n + i] = coeff_w
        
        add_rank1(u_vec, v_vec, w_vec)
    
    # Strassen 2x2 terms - CORRECTED version
    # For 2x2x2: A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = A*B: C00 = A00*B00 + A01*B10, C01 = A00*B01 + A01*B11
    #           C10 = A10*B00 + A11*B10, C11 = A10*B01 + A11*B11
    #
    # Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 - M3 + M6
    
    # Indexing: u_vec[li*2+lk], v_vec[lk*2+lj], w_vec[lj*2+li]
    # For 2x2: li,lk,lj in {0,1}
    
    strassen_2x2 = [
        # M1 = (A00+A11)*(B00+B11): contributes to C00(+1), C11(+1)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # w: C00=idx0, C11=idx3
        # M2 = (A10+A11)*B00: contributes to C10(+1), C11(-1)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # w: C10=idx2, C11=idx3
        # M3 = A00*(B01-B11): contributes to C01(+1), C11(-1)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, -1]),  # w: C01=idx1, C11=idx3
        # M4 = A11*(B10-B00): contributes to C00(-1), C10(+1)
        ([0, 0, 0, 1], [0, -1, 1, 0], [-1, 0, 1, 0]),  # w: C00=idx0, C10=idx2
        # M5 = (A00+A01)*B11: contributes to C00(-1), C01(+1)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # w: C00=idx0, C01=idx1
        # M6 = (A10-A00)*(B00+B01): contributes to C11(+1)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # w: C11=idx3
        # M7 = (A01-A11)*(B10+B11): contributes to C00(+1)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # w: C00=idx0
    ]
    
    # Now use Strassen to do 4x4x4 (i in 0..3, k in 0..3, j in 0..3)
    # Split into 2x2 blocks:
    # A blocks: A00 (i:0-1, k:0-1), A01 (i:0-1, k:2-3), A10 (i:2-3, k:0-1), A11 (i:2-3, k:2-3)
    # B blocks: B00 (k:0-1, j:0-1), B01 (k:0-1, j:2-3), B10 (k:2-3, j:0-1), B11 (k:2-3, j:2-3)
    # C blocks: C00 (i:0-1, j:0-1), C01 (i:0-1, j:2-3), C10 (i:2-3, j:0-1), C11 (i:2-3, j:2-3)
    
    # For each outer Strassen product, we have:
    # A_part: linear combo of A blocks
    # B_part: linear combo of B blocks  
    # C_part: linear combo of C blocks (with signs)
    #
    # The product is: (sum a_s * A_s) * (sum b_t * B_t) = sum_{s,t} a_s * b_t * (A_s * B_t)
    # Each A_s * B_t is a 2x2 * 2x2 = 2x2 matrix multiplication
    # We use inner Strassen for each such product
    
    # Outer Strassen products (same as 2x2 case but with block indices):
    outer_products = [
        # M1 = (A00+A11)*(B00+B11) -> C00 + C11
        {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
        # M2 = (A10+A11)*B00 -> C10 - C11
        {'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1), (3, -1)]},
        # M3 = A00*(B01-B11) -> C01 - C11
        {'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1), (3, -1)]},
        # M4 = A11*(B10-B00) -> -C00 + C10
        {'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, -1), (2, 1)]},
        # M5 = (A00+A01)*B11 -> -C00 + C01
        {'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, -1), (1, 1)]},
        # M6 = (A10-A00)*(B00+B01) -> C11
        {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(3, 1)]},
        # M7 = (A01-A11)*(B10+B11) -> C00
        {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, 1)]},
    ]
    
    # Block to global index mapping:
    # A block b (0-3): b//2 gives row half, b%2 gives col half
    #   block 0: i in 0-1, k in 0-1
    #   block 1: i in 0-1, k in 2-3
    #   block 2: i in 2-3, k in 0-1
    #   block 3: i in 2-3, k in 2-3
    # B block b (0-3): b//2 gives row half (k), b%2 gives col half (j)
    #   block 0: k in 0-1, j in 0-1
    #   block 1: k in 0-1, j in 2-3
    #   block 2: k in 2-3, j in 0-1
    #   block 3: k in 2-3, j in 2-3
    # C block b (0-3): b//2 gives row half (i), b%2 gives col half (j)
    #   block 0: i in 0-1, j in 0-1
    #   block 1: i in 0-1, j in 2-3
    #   block 2: i in 2-3, j in 0-1
    #   block 3: i in 2-3, j in 2-3
    
    def a_block_indices(block):
        """Return list of (i, k) for A block"""
        r = block // 2
        c = block % 2
        return [(r * 2 + di, c * 2 + dk) for di in range(2) for dk in range(2)]
    
    def b_block_indices(block):
        """Return list of (k, j) for B block"""
        r = block // 2
        c = block % 2
        return [(r * 2 + dk, c * 2 + dj) for dk in range(2) for dj in range(2)]
    
    def c_block_indices(block):
        """Return list of (i, j) for C block"""
        r = block // 2
        c = block % 2
        return [(r * 2 + di, c * 2 + dj) for di in range(2) for dj in range(2)]
    
    # For each outer product, generate 7 inner Strassen terms
    for prod in outer_products:
        a_terms = prod['A']  # list of (block, sign)
        b_terms = prod['B']
        c_terms = prod['C']
        
        # For each inner Strassen term
        for u_inner, v_inner, w_inner in strassen_2x2:
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)
            
            # u_inner is indexed by [li*2+lk] for the local 2x2 A
            # For each local (li, lk) in the inner Strassen, we need to map to global
            # The local 2x2 A has elements at positions corresponding to one A block
            # But since we have multiple A blocks with signs, we sum contributions
            
            # Actually, the structure is:
            # The "matrix" being multiplied is A_mat = sum_s a_s * A_s (where A_s is a 2x2 block)
            # Similarly B_mat = sum_t b_t * B_t
            # The product A_mat * B_mat uses Strassen on the 2x2 structure
            # Each inner Strassen term picks certain entries of A_mat and B_mat
            # A_mat[li][lk] = sum_s a_s * A_s[li][lk]
            # where A_s[li][lk] is the element at local position (li, lk) of block s
            
            # So for u_inner[li*2+lk] = val:
            # This means we need val * A_mat[li][lk] = val * sum_s a_s * A_s[li][lk]
            # = sum_s val * a_s * A_s[li][lk]
            # A_s[li][lk] corresponds to global position (block_s_row*2+li, block_s_col*2+lk)
            
            for li in range(2):
                for lk in range(2):
                    local_val = u_inner[li * 2 + lk]
                    if local_val != 0:
                        for block, sign in a_terms:
                            # Get all global (i, k) for this block
                            for (gi, gk) in a_block_indices(block):
                                # Check if this local (li, lk) maps to this global position
                                # Block's local row offset:
                                r_off = block // 2
                                c_off = block % 2
                                if gi == r_off * 2 + li and gk == c_off * 2 + lk:
                                    u_vec[gi * m + gk] += local_val * sign
            
            # Similarly for v_inner[lok*2+loj]
            for lk in range(2):
                for lj in range(2):
                    local_val = v_inner[lk * 2 + lj]
                    if local_val != 0:
                        for block, sign in b_terms:
                            r_off = block // 2
                            c_off = block % 2
                            if r_off * 2 + lk == r_off * 2 + lk and c_off * 2 + lj == c_off * 2 + lj:
                                # This is always true, so just add
                                v_vec[(r_off * 2 + lk) * p + (c_off * 2 + lj)] += local_val * sign
            
            # Similarly for w_inner[loj*2+loi]
            for lj in range(2):
                for li in range(2):
                    local_val = w_inner[lj * 2 + li]
                    if local_val != 0:
                        for block, sign in c_terms:
                            r_off = block // 2
                            c_off = block % 2
                            if r_off * 2 + li == r_off * 2 + li and c_off * 2 + lj == c_off * 2 + lj:
                                w_vec[(c_off * 2 + lj) * n + (r_off * 2 + li)] += local_val * sign
            
            add_rank1(u_vec, v_vec, w_vec)
    
    # Part 2: Handle k=4 or j=4 (schoolbook for the remaining entries)
    # We need: sum_k A[i,k] * B[k,j] for all i in 0..3, j in 0..4
    # The Strassen part handled k in 0..3, j in 0..3
    # Remaining: k=4 (any j in 0..4) OR j=4 (any k in 0..3)
    # But if k=4 AND j=4, it's counted once
    
    # Add terms for k in 0..3, j=4:
    for i in range(n):
        for k in range(4):
            add_term(i, k, 4, 1.0, 1.0, 1.0)
    
    # Add terms for k=4, j in 0..4:
    for i in range(n):
        for j in range(p):
            add_term(i, 4, j, 1.0, 1.0, 1.0)
    
    rank = len(U_cols)
    
    # Verification (optional, can skip for speed)
    # Let's do a quick check with random matrices
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u = U_cols[r]
        v = V_cols[r]
        w = W_cols[r]
        a_mat = u.reshape(n, m)
        b_mat = v.reshape(m, p)
        c_mat = w.reshape(p, n).T  # w is indexed by j*n+i, so reshape and transpose
        prod = a_mat @ b_mat
        C_computed += prod * c_mat
    
    # Check if decomposition is correct
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook
        U_cols = []
        V_cols = []
        W_cols = []
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    add_term(i, k, j, 1.0, 1.0, 1.0)
        rank = len(U_cols)
    
    if rank == 0:
        add_term(0, 0, 0, 1.0, 1.0, 1.0)
        rank = 1
    
    U = np.stack(U_cols, axis=1)
    V = np.stack(V_cols, axis=1)
    W = np.stack(W_cols, axis=1)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
