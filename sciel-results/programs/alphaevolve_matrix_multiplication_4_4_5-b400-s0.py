import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # We will build U, V, W as lists of columns (rank vectors)
    # Then stack them.
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a rank-1 term
    # u: vector of length n*m (16)
    # v: vector of length m*p (20)
    # w: vector of length p*n (20)
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

    # --- Strassen 2x2x2 Decomposition ---
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Local indices for A (2x2): (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # Local indices for B (2x2): (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    # Local indices for C (2x2): (0,0)=c11, (0,1)=c12, (1,0)=c21, (1,1)=c22
    
    # Global mapping function for A
    # A is 4x4. Block (bi, bj) is 2x2.
    # Local (li, lj) -> Global (bi*2+li, bj*2+lj)
    # Index in U (flattened row-major i*m+j): i*m + j
    
    def get_u_idx(gi, gj):
        return gi * m + gj
        
    def get_v_idx(gj, gk):
        return gj * p + gk
        
    def get_w_idx(gk, gi):
        return gk * n + gi

    # Define Strassen terms for a block at A offset (ai, aj) and B offset (bj, bk)
    # ai, aj: row/col start index for A block (0 or 2)
    # bj, bk: row/col start index for B block (0 or 2)
    # Note: B block is 2x2, so bk is start col, width 2.
    
    def add_strassen_2x2_2(ai, aj, bj, bk):
        # a = A[ai, aj], b = A[ai, aj+1], c = A[ai+1, aj], d = A[ai+1, aj+1]
        # e = B[bj, bk], f = B[bj, bk+1], g = B[bj+1, bk], h = B[bj+1, bk+1]
        
        # M1 = (a+d)(e+h)
        u1 = np.zeros(n*m)
        u1[get_u_idx(ai, aj)] = 1
        u1[get_u_idx(ai+1, aj+1)] = 1
        v1 = np.zeros(m*p)
        v1[get_v_idx(bj, bk)] = 1
        v1[get_v_idx(bj+1, bk+1)] = 1
        w1 = np.zeros(p*n)
        w1[get_w_idx(bk, ai)] = 1      # c11 += M1
        w1[get_w_idx(bk+1, ai+1)] = 1  # c22 += M1 (Wait, C22 = M1 - M2 + M3 + M6)
        # Let's be precise with C contributions later.
        # Actually, it's easier to define the U, V, W for each M_i, 
        # and then accumulate them into the final C via W?
        # No, the tensor decomposition is sum_r U_r V_r W_r.
        # The W vector for a term r corresponds to the contribution to C.
        # So for M1, it contributes to C11, C22.
        # C11 index: (ai, bk). C22 index: (ai+1, bk+1).
        
        # Let's re-evaluate.
        # The term is U_vec * V_vec * W_vec.
        # U_vec picks up elements of A.
        # V_vec picks up elements of B.
        # W_vec picks up elements of C.
        
        # M1 = (A_ai,aj + A_ai+1,aj+1) * (B_bj,bk + B_bj+1,bk+1)
        # This term contributes to C_ai,bk and C_ai+1,bk+1.
        # C_ai,bk = ... + M1 ...
        # C_ai+1,bk+1 = ... + M1 ...
        
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj)] = 1
        u[get_u_idx(ai+1, aj+1)] = 1
        
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk)] = 1
        v[get_v_idx(bj+1, bk+1)] = 1
        
        w = np.zeros(p*n)
        w[get_w_idx(bk, ai)] = 1       # C[ai, bk]
        w[get_w_idx(bk+1, ai+1)] = 1   # C[ai+1, bk+1]
        
        add_term(u, v, w)

        # M2 = (c+d)e = (A_ai+1,aj + A_ai+1,aj+1) * B_bj,bk
        # Contributes to C_ai+1,bk (C21) and C_ai+1,bk+1 (C22)?
        # C21 = M2 + M4. C22 = M1 - M2 + M3 + M6.
        # So M2 contributes to C21 with +1 and C22 with -1.
        
        u = np.zeros(n*m)
        u[get_u_idx(ai+1, aj)] = 1
        u[get_u_idx(ai+1, aj+1)] = 1
        
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk)] = 1
        
        w = np.zeros(p*n)
        w[get_w_idx(bk, ai+1)] = 1     # C21
        w[get_w_idx(bk+1, ai+1)] = -1  # C22
        
        add_term(u, v, w)

        # M3 = a(f-h) = A_ai,aj * (B_bj,bk+1 - B_bj+1,bk+1)
        # Contributes to C_ai,bk+1 (C12) with +1 and C_ai+1,bk+1 (C22) with +1.
        # C12 = M3 + M5. C22 = ... + M3 ...
        
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj)] = 1
        
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk+1)] = 1
        v[get_v_idx(bj+1, bk+1)] = -1
        
        w = np.zeros(p*n)
        w[get_w_idx(bk+1, ai)] = 1     # C12
        w[get_w_idx(bk+1, ai+1)] = 1   # C22
        
        add_term(u, v, w)

        # M4 = d(g-e) = A_ai+1,aj+1 * (B_bj+1,bk - B_bj,bk)
        # Contributes to C_ai,bk (C11) with +1 and C_ai+1,bk (C21) with +1.
        # C11 = M1 + M4 ... C21 = M2 + M4
        
        u = np.zeros(n*m)
        u[get_u_idx(ai+1, aj+1)] = 1
        
        v = np.zeros(m*p)
        v[get_v_idx(bj+1, bk)] = 1
        v[get_v_idx(bj, bk)] = -1
        
        w = np.zeros(p*n)
        w[get_w_idx(bk, ai)] = 1       # C11
        w[get_w_idx(bk, ai+1)] = 1     # C21
        
        add_term(u, v, w)

        # M5 = (a+b)h = (A_ai,aj + A_ai,aj+1) * B_bj+1,bk+1
        # Contributes to C_ai,bk+1 (C12) with +1 and C_ai,bk (C11) with -1.
        # C11 = ... - M5 ... C12 = M3 + M5
        
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj)] = 1
        u[get_u_idx(ai, aj+1)] = 1
        
        v = np.zeros(m*p)
        v[get_v_idx(bj+1, bk+1)] = 1
        
        w = np.zeros(p*n)
        w[get_w_idx(bk, ai)] = -1      # C11
        w[get_w_idx(bk+1, ai)] = 1     # C12
        
        add_term(u, v, w)

        # M6 = (c-a)(e+f) = (A_ai+1,aj - A_ai,aj) * (B_bj,bk + B_bj,bk+1)
        # Contributes to C_ai+1,bk+1 (C22) with +1.
        # C22 = ... + M6
        
        u = np.zeros(n*m)
        u[get_u_idx(ai+1, aj)] = 1
        u[get_u_idx(ai, aj)] = -1
        
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk)] = 1
        v[get_v_idx(bj, bk+1)] = 1
        
        w = np.zeros(p*n)
        w[get_w_idx(bk+1, ai+1)] = 1   # C22
        
        add_term(u, v, w)

        # M7 = (b-d)(g+h) = (A_ai,aj+1 - A_ai+1,aj+1) * (B_bj+1,bk + B_bj+1,bk+1)
        # Contributes to C_ai,bk (C11) with +1.
        # C11 = ... + M7
        
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj+1)] = 1
        u[get_u_idx(ai+1, aj+1)] = -1
        
        v = np.zeros(m*p)
        v[get_v_idx(bj+1, bk)] = 1
        v[get_v_idx(bj+1, bk+1)] = 1
        
        w = np.zeros(p*n)
        w[get_w_idx(bk, ai)] = 1       # C11
        
        add_term(u, v, w)

    # --- 2x2x3 Decomposition (Strassen for first 2 cols, Naive for 3rd) ---
    # A is 2x2. B is 2x3.
    # B = [[e, f, g], [h, i, j]]
    # We use Strassen on B_a = [[e, f], [h, i]] (cols 0,1)
    # We use Naive on B_b = [[g], [j]] (col 2)
    
    def add_2x2x3(ai, aj, bj, bk):
        # A offset (ai, aj), B offset (bj, bk)
        # B block is 2 rows (bj, bj+1) and 3 cols (bk, bk+1, bk+2)
        
        # Part 1: Strassen on cols bk, bk+1
        # This is identical to add_strassen_2x2_2 but W indices are for C cols bk, bk+1
        # We can just call add_strassen_2x2_2? 
        # No, add_strassen_2x2_2 assumes B is 2x2 and C is 2x2.
        # Here C is 2x3. The Strassen part only affects C cols bk and bk+1.
        # So we can reuse the logic, but we must ensure W indices are correct.
        # In add_strassen_2x2_2, W indices were (bk, ai), (bk+1, ai+1) etc.
        # This matches the first two columns of the 2x3 C block.
        # So we can just call add_strassen_2x2_2(ai, aj, bj, bk)
        add_strassen_2x2_2(ai, aj, bj, bk)
        
        # Part 2: Naive for col bk+2
        # C_col2 = A * B_col2
        # C_ai, bk+2 = A_ai,aj * B_bj,bk+2 + A_ai,aj+1 * B_bj+1,bk+2
        # C_ai+1, bk+2 = A_ai+1,aj * B_bj,bk+2 + A_ai+1,aj+1 * B_bj+1,bk+2
        
        # Term 1: A_ai,aj * B_bj,bk+2
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk+2)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk+2, ai)] = 1
        add_term(u, v, w)
        
        # Term 2: A_ai,aj+1 * B_bj+1,bk+2
        u = np.zeros(n*m)
        u[get_u_idx(ai, aj+1)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj+1, bk+2)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk+2, ai)] = 1
        add_term(u, v, w)
        
        # Term 3: A_ai+1,aj * B_bj,bk+2
        u = np.zeros(n*m)
        u[get_u_idx(ai+1, aj)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj, bk+2)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk+2, ai+1)] = 1
        add_term(u, v, w)
        
        # Term 4: A_ai+1,aj+1 * B_bj+1,bk+2
        u = np.zeros(n*m)
        u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p)
        v[get_v_idx(bj+1, bk+2)] = 1
        w = np.zeros(p*n)
        w[get_w_idx(bk+2, ai+1)] = 1
        add_term(u, v, w)

    # --- Construct the full decomposition ---
    
    # A blocks:
    # A00: ai=0, aj=0
    # A01: ai=0, aj=2
    # A10: ai=2, aj=0
    # A11: ai=2, aj=2
    
    # B blocks:
    # B00: bj=0, bk=0 (2x2)
    # B01: bj=0, bk=2 (2x3)
    # B10: bj=2, bk=0 (2x2)
    # B11: bj=2, bk=2 (2x3)
    
    # C00 = A00 B00 + A01 B10
    # C00 is 2x2.
    # A00 B00 is 2x2x2. Add Strassen for (0,0) and (0,0).
    add_strassen_2x2_2(0, 0, 0, 0)
    # A01 B10 is 2x2x2. Add Strassen for (0,2) and (2,0).
    add_strassen_2x2_2(0, 2, 2, 0)
    
    # C01 = A00 B01 + A01 B11
    # C01 is 2x3.
    # A00 B01 is 2x2x3. Add 2x2x3 for (0,0) and (0,2).
    add_2x2x3(0, 0, 0, 2)
    # A01 B11 is 2x2x3. Add 2x2x3 for (0,2) and (2,2).
    add_2x2x3(0, 2, 2, 2)
    
    # C10 = A10 B00 + A11 B10
    # C10 is 2x2.
    # A10 B00 is 2x2x2. Add Strassen for (2,0) and (0,0).
    add_strassen_2x2_2(2, 0, 0, 0)
    # A11 B10 is 2x2x2. Add Strassen for (2,2) and (2,0).
    add_strassen_2x2_2(2, 2, 2, 0)
    
    # C11 = A10 B01 + A11 B11
    # C11 is 2x3.
    # A10 B01 is 2x2x3. Add 2x2x3 for (2,0) and (0,2).
    add_2x2x3(2, 0, 0, 2)
    # A11 B11 is 2x2x3. Add 2x2x3 for (2,2) and (2,2).
    add_2x2x3(2, 2, 2, 2)

    # Stack into matrices
    if not U_cols:
        # Fallback to schoolbook if something went wrong (shouldn't happen)
        rank = n*m*p
        U = np.eye(rank).reshape(n*m, rank)
        V = np.eye(rank).reshape(m*p, rank)
        W = np.eye(rank).reshape(p*n, rank)
        # Wait, schoolbook needs specific U,V,W.
        # U[i*m+j, r] = 1 if r == (i*m+j)*p + k? No.
        # Let's just return the constructed one.
        pass

    U_mat = np.array(U_cols).T  # Shape (16, rank)
    V_mat = np.array(V_cols).T  # Shape (20, rank)
    W_mat = np.array(W_cols).T  # Shape (20, rank)
    
    # Verify
    # The task requires integer/half-integer entries.
    # Our entries are -1, 0, 1. This is fine.
    
    # Check rank
    rank = U_mat.shape[1]
    
    # Quick verification
    # T_ijk = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # We can check a few points.
    
    return {
        "U": U_mat.tolist(),
        "V": V_mat.tolist(),
        "W": W_mat.tolist()
    }
# EVOLVE-BLOCK-END
