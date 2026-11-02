import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    Attempts rank reduction via random linear combinations if time permits.
    """
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank))  # 16 x 49
    V = np.zeros((m * p, rank))  # 16 x 49
    W = np.zeros((p * n, rank))  # 16 x 49
    
    # Build the 7 level-1 Strassen terms
    def block_a(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    def block_b(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    def block_c(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    level1_terms = []
    
    # Term 0: (A11+A22)*(B11+B22)
    L0 = []
    R0 = []
    for r in range(2):
        row_l = []
        row_r = []
        for c in range(2):
            ar, ac = block_a(0,0,r,c)
            dr, dc = block_a(1,1,r,c)
            row_l.append([(ar,ac,1),(dr,dc,1)])
            er, ec = block_b(0,0,r,c)
            hr, hc = block_b(1,1,r,c)
            row_r.append([(er,ec,1),(hr,hc,1)])
        L0.append(row_l)
        R0.append(row_r)
    C0 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = block_c(0,0,r,c)
            cr2, cc2 = block_c(1,1,r,c)
            C0[(r,c)] = [(cr,cc,1),(cr2,cc2,1)]
    level1_terms.append((L0, R0, C0))
    
    # Term 1: (A21+A22)*B11
    L1 = []
    R1 = []
    for r in range(2):
        row_l = []
        row_r = []
        for c in range(2):
            ar, ac = block_a(1,0,r,c)
            dr, dc = block_a(1,1,r,c)
            row_l.append([(ar,ac,1),(dr,dc,1)])
            er, ec = block_b(0,0,r,c)
            row_r.append([(er,ec,1)])
        L1.append(row_l)
        R1.append(row_r)
    C1 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = block_c(1,0,r,c)
            cr2, cc2 = block_c(1,1,r,c)
            C1[(r,c)] = [(cr,cc,1),(cr2,cc2,-1)]
    level1_terms.append((L1, R1, C1))
    
    # Term 2: A11*(B12-B22)
    L2 = []
    R2 = []
    for r in range(2):
        row_l = []
        row_r = []
        for c in range(2):
            ar, ac = block_a(0,0,r,c)
            row_l.append([(ar,ac,1)])
            fr, fc = block_b(0,1,r,c)
            hr, hc = block_b(1,1,r,c)
            row_r.append([(fr,fc,1),(hr,hc,-1)])
        L2.append(row_l)
        R2.append(row_r)
    C2 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = block_c(0,1,r,c)
            cr2, cc2 = block_c(1,1,r,c)
            C2[(r,c)] = [(cr,cc,1),(cr2,cc2,1)]
    level1_terms.append((L2, R2, C2))
    
    # Term 3: A22*(B21-B11)
    L3 = []
    R3 = []
    for r in range(2):
        row_l = []
        row_r = []
        for c in range(2):
            dr, dc = block_a(1,1,r,c)
            row_l.append([(dr,dc,1)])
            gr, gc = block_b(1,0,r,c)
            er, ec = block_b(0,0,r,c)
            row_r.append([(gr,gc,1),(er,ec,-1)])
        L3.append(row_l)
        R3.append(row_r)
    C3 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = block_c(0,0,r,c)
            cr2, cc2 = block_c(1,0,r,c)
            C3[(r,c)] = [(cr,cc,1),(cr2,cc2,1)]
    level1_terms.append((L3, R3, C3))
    
    # Term 4: (A11+A12)*B22
    L4 = []
    R4 = []
    for r in range(2):
        row_l = []
        row_r = []
        for c in range(2):
            ar, ac = block_a(0,0,r,c)
            br, bc = block_a(0,1,r,c)
            row_l.append([(ar,ac,1),(br,bc,1)])
            hr, hc = block_b(1,1,r,c)
            row_r.append([(hr,hc,1)])
        L4.append(row_l)
        R4.append(row_r)
    C4 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = block_c(0,0,r,c)
            cr2, cc2 = block_c(0,1,r,c)
            C4[(r,c)] = [(cr,cc,-1),(cr2,cc2,1)]
    level1_terms.append((L4, R4, C4))
    
    # Term 5: (A21-A11)*(B11+B12)
    L5 = []
    R5 = []
    for r in range(2):
        row_l = []
        row_r = []
        for c in range(2):
            cr, cc = block_a(1,0,r,c)
            ar, ac = block_a(0,0,r,c)
            row_l.append([(cr,cc,1),(ar,ac,-1)])
            er, ec = block_b(0,0,r,c)
            fr, fc = block_b(0,1,r,c)
            row_r.append([(er,ec,1),(fr,fc,1)])
        L5.append(row_l)
        R5.append(row_r)
    C5 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = block_c(1,1,r,c)
            C5[(r,c)] = [(cr,cc,1)]
    level1_terms.append((L5, R5, C5))
    
    # Term 6: (A12-A22)*(B21+B22)
    L6 = []
    R6 = []
    for r in range(2):
        row_l = []
        row_r = []
        for c in range(2):
            br, bc = block_a(0,1,r,c)
            dr, dc = block_a(1,1,r,c)
            row_l.append([(br,bc,1),(dr,dc,-1)])
            gr, gc = block_b(1,0,r,c)
            hr, hc = block_b(1,1,r,c)
            row_r.append([(gr,gc,1),(hr,hc,1)])
        L6.append(row_l)
        R6.append(row_r)
    C6 = {}
    for r in range(2):
        for c in range(2):
            cr, cc = block_c(0,0,r,c)
            C6[(r,c)] = [(cr,cc,1)]
    level1_terms.append((L6, R6, C6))
    
    # Strassen level-2 definitions
    strassen_L = [
        [(0,0,1),(1,1,1)],
        [(1,0,1),(1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1),(0,1,1)],
        [(1,0,1),(0,0,-1)],
        [(0,1,1),(1,1,-1)]
    ]
    strassen_R = [
        [(0,0,1),(1,1,1)],
        [(0,0,1)],
        [(0,1,1),(1,1,-1)],
        [(1,0,1),(0,0,-1)],
        [(1,1,1)],
        [(0,0,1),(0,1,1)],
        [(1,0,1),(1,1,1)]
    ]
    strassen_contrib = [
        [(0,0,1),(1,1,1)],
        [(1,0,1),(1,1,-1)],
        [(0,1,1),(1,1,1)],
        [(0,0,1),(1,0,1)],
        [(0,0,-1),(0,1,1)],
        [(1,1,1)],
        [(0,0,1)]
    ]
    
    # Build the 49-term decomposition
    for t in range(7):
        L_t, R_t, C_t = level1_terms[t]
        for s in range(7):
            rank_idx = t * 7 + s
            u_dict = {}
            for (li, lj, sc) in strassen_L[s]:
                for (a_row, a_col, a_coeff) in L_t[li][lj]:
                    u_idx = a_row * 4 + a_col
                    u_dict[u_idx] = u_dict.get(u_idx, 0) + sc * a_coeff
            for u_idx, val in u_dict.items():
                if val != 0:
                    U[u_idx, rank_idx] = val
            
            v_dict = {}
            for (ri, rj, sc) in strassen_R[s]:
                for (b_row, b_col, b_coeff) in R_t[ri][rj]:
                    v_idx = b_row * 4 + b_col
                    v_dict[v_idx] = v_dict.get(v_idx, 0) + sc * b_coeff
            for v_idx, val in v_dict.items():
                if val != 0:
                    V[v_idx, rank_idx] = val
            
            w_dict = {}
            for (zi, zj, zcoeff) in strassen_contrib[s]:
                if (zi, zj) in C_t:
                    for (gc_row, gc_col, c_coeff) in C_t[(zi, zj)]:
                        w_idx = gc_col * 4 + gc_row
                        w_dict[w_idx] = w_dict.get(w_idx, 0) + zcoeff * c_coeff
            for w_idx, val in w_dict.items():
                if val != 0:
                    W[w_idx, rank_idx] = val
    
    # Fast verification using vectorized approach
    def verify_decomposition(U_mat, V_mat, W_mat, n_test=3):
        """Verify decomposition using random tests. Returns True if valid."""
        for seed in range(n_test):
            np.random.seed(seed)
            A_test = np.random.randn(4, 4)
            B_test = np.random.randn(4, 4)
            C_expected = A_test @ B_test
            # Vectorized: C[i,j] = sum_r sum_k U[i*4+k,r] * V[k*4+j,r] * W[j*4+i,r]
            # Reshape: U_mat is (16, rank), V_mat is (16, rank), W_mat is (16, rank)
            # For each r: outer product structure
            # C = sum_r (U_r reshaped as 4x4) * (V_r reshaped as 4x4) weighted by W_r
            # Actually: C[i,j] = sum_r W[j*4+i, r] * (sum_k U[i*4+k,r] * V[k*4+j,r])
            # Let M_r = U[:,r] reshaped to (4,4) [row i, col k] and N_r = V[:,r] reshaped to (4,4) [row k, col j]
            # Then sum_k U[i*4+k,r]*V[k*4+j,r] = (U_r @ V_r)[i,j]
            # So C[i,j] = sum_r W[j*4+i,r] * (U_r @ V_r)[i,j]
            C_computed = np.zeros((4, 4))
            for r in range(U_mat.shape[1]):
                Ur = U_mat[:, r].reshape(4, 4)
                Vr = V_mat[:, r].reshape(4, 4)
                Wr = W_mat[:, r].reshape(4, 4)  # Wr[j, i] = W[j*4+i, r]
                C_computed += Wr.T * (Ur @ Vr)  # Wr.T[i,j] = Wr[j,i]
            if not np.allclose(C_computed, C_expected, atol=1e-8):
                return False
        return True
    
    # Verify base decomposition
    elapsed = time.time() - start_time
    if not verify_decomposition(U, V, W):
        # Should not happen, but return anyway
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Attempt rank reduction: try to find linear combinations of columns that allow
    # removal of one column while preserving the tensor identity.
    # Strategy: for each pair (r1, r2), check if replacing column r2 with a random linear
    # combination of all columns still yields a valid decomposition. If so, we can potentially
    # eliminate r2.
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = 49
    
    # Try to reduce rank by finding redundant columns
    # A column r is redundant if there exists a linear combination of other columns
    # that produces the same tensor contribution.
    
    # More practical approach: try random linear combinations of column pairs
    # and check if the resulting 48-column decomposition is still valid.
    
    np.random.seed(123)
    max_attempts = 2000  # limited by time
    attempt = 0
    
    while time.time() - start_time < DEADLINE - 10 and attempt < max_attempts:
        attempt += 1
        
        # Pick two random columns to try to eliminate
        r1, r2 = np.random.choice(49, 2, replace=False)
        
        # Try: new_col_r2 = alpha * col_r1 + beta * col_r2 (for U, V, W)
        # If we can find alpha, beta such that the resulting 48-col decomposition works,
        # we've reduced rank.
        
        # Actually, a simpler approach: check if any column is a linear combination
        # of others in the "tensor sense". This is complex, so instead:
        # Try random perturbations and see if we can merge columns.
        
        # Alternative: try to find if column r2 can be expressed as a linear combo
        # of all other columns in a way that preserves the identity.
        # This requires solving a system, which is expensive.
        
        # Simpler heuristic: for each column r, try removing it and checking if
        # the remaining 48 columns still span the tensor.
        # We can test this by checking if the tensor residual is zero.
        
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Try removing column r2
        U_test = np.delete(U, r2, axis=1)
        V_test = np.delete(V, r2, axis=1)
        W_test = np.delete(W, r2, axis=1)
        
        # Quick check: is the removed column's contribution zero?
        # (It shouldn't be, but let's check)
        if np.allclose(U[:, r2], 0) and np.allclose(V[:, r2], 0) and np.allclose(W[:, r2], 0):
            best_rank = 48
            best_U, best_V, best_W = U_test, V_test, W_test
            continue
        
        # Check if the tensor is still correctly decomposed without column r2
        # This is expensive, so do a quick spot check
        np.random.seed(99)
        A_spot = np.random.randn(4, 4)
        B_spot = np.random.randn(4, 4)
        C_expected = A_spot @ B_spot
        C_computed = np.zeros((4, 4))
        for r in range(48):
            Ur = U_test[:, r].reshape(4, 4)
            Vr = V_test[:, r].reshape(4, 4)
            Wr = W_test[:, r].reshape(4, 4)
            C_computed += Wr.T * (Ur @ Vr)
        
        if np.allclose(C_computed, C_expected, atol=1e-8):
            best_rank = 48
            best_U, best_V, best_W = U_test, V_test, W_test
    
    # If we found a reduced rank, use it
    if best_rank < 49:
        U, V, W = best_U, best_V, best_W
        rank = best_rank
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
