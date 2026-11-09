"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Partition indices into 3 groups: 2, 2, 1
    # Group 0: indices 0, 1
    # Group 1: indices 2, 3
    # Group 2: index 4
    groups = [[0, 1], [2, 3], [4]]
    g_sizes = [2, 2, 1]
    
    # Total rank is sum of ranks of each block product
    # 8 blocks of 2x2x2 (Strassen rank 7)
    # 19 blocks of mixed size (Schoolbook rank = product of sizes)
    # Rank calculation:
    # 8 * 7 = 56
    # k=1 (size 2) non-2x2x2:
    #   i=1,j=3 (2x2*2x1 -> 4)
    #   i=2,j=3 (2x2*2x1 -> 4)
    #   i=3,j=1 (1x2*2x2 -> 4)
    #   i=3,j=2 (1x2*2x2 -> 4)
    #   i=3,j=3 (1x2*2x1 -> 2)
    #   Sum = 18
    # k=2 (size 2) non-2x2x2:
    #   Same as above = 18
    # k=3 (size 1):
    #   i=1,j=1 (2x1*1x2 -> 4)
    #   i=1,j=2 (2x1*1x2 -> 4)
    #   i=1,j=3 (2x1*1x1 -> 2)
    #   i=2,j=1 (2x1*1x2 -> 4)
    #   i=2,j=2 (2x1*1x2 -> 4)
    #   i=2,j=3 (2x1*1x1 -> 2)
    #   i=3,j=1 (1x1*1x2 -> 2)
    #   i=3,j=2 (1x1*1x2 -> 2)
    #   i=3,j=3 (1x1*1x1 -> 1)
    #   Sum = 25
    # Total Rank = 56 + 18 + 18 + 25 = 117
    
    rank = 117
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    def add_term(u_idx, v_idx, w_idx, val_u, val_v, val_w):
        nonlocal col_idx
        U[u_idx][col_idx] += val_u
        V[v_idx][col_idx] += val_v
        W[w_idx][col_idx] += val_w
        col_idx += 1

    # Helper to add schoolbook decomposition for a block of size s_u x s_v
    # u_idx_offset, v_idx_offset are the starting global indices
    # The block A is s_u x s_k, B is s_k x s_v
    # We need to map local (row, col) to global indices
    # Global row for A: u_start + r
    # Global col for A: k_start + c
    # Global row for B: k_start + c
    # Global col for B: v_start + c
    
    def add_schoolbook(u_start, k_start, v_start, s_u, s_k, s_v):
        nonlocal col_idx
        for r in range(s_u):
            for c in range(s_k):
                for d in range(s_v):
                    u_g = u_start + r
                    v_g = k_start + c
                    w_g = v_start + d
                    # W index is p*n + i, so w_g is the index in the flattened W matrix
                    # But W is defined as W[k*n+i]. 
                    # In the tensor decomposition C[i,j] = sum_k A[i,k] B[k,j]
                    # The standard decomposition uses terms T_{i,k,j}
                    # U maps (i,k) to rank. V maps (k,j) to rank. W maps (j,i) to rank?
                    # Task says: U(n*m, rank), V(m*p, rank), W(p*n, rank).
                    # C = sum_r U[:,r] V[:,r]^T ... wait.
                    # Usually: C_{i,j} = sum_r U_{i,r} V_{j,r} is for C=UV^T?
                    # No, for matrix multiplication tensor:
                    # T_{i,k,j} = delta_{i,k} delta_{k,j}? No.
                    # The tensor is T[i,k,j] = 1 if it contributes to C[i,j] via A[i,k]B[k,j].
                    # Decomposition: T[i,k,j] = sum_r U[i,k,r] V[k,j,r] W[j,i,r]?
                    # The interface is U(n*m, rank), V(m*p, rank), W(p*n, rank).
                    # This implies:
                    # U index is i*m + k ? (n*m size)
                    # V index is k*p + j ? (m*p size)
                    # W index is j*n + i ? (p*n size)
                    # And C[i,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] ?
                    # No, the standard form is C = U V^T? No.
                    # It is: C_{i,j} = \sum_r U_{i, r} V_{j, r} is rank-1.
                    # For matrix mult: C_{i,j} = \sum_k A_{i,k} B_{k,j}.
                    # The tensor is T_{i,k,j}.
                    # We need T_{i,k,j} = \sum_r U_{i,m+k,r} V_{k,p+j,r} W_{j,n+i,r} ?
                    # Let's check the seed code.
                    # Seed: U[i*m+j][col] = 1, V[j*p+k][col] = 1, W[k*n+i][col] = 1.
                    # Here i is row of A, j is col of A (which is row of B), k is col of B (which is col of C).
                    # So U index is i*m+j. V index is j*p+k. W index is k*n+i.
                    # This matches the task description.
                    
                    u_g_flat = u_start + r * m + (k_start + c) # i*m + k
                    v_g_flat = (k_start + c) * p + (v_start + d) # k*p + j
                    w_g_flat = (v_start + d) * n + (u_start + r) # j*n + i
                    
                    add_term(u_g_flat, v_g_flat, w_g_flat, 1, 1, 1)

    def add_strassen(u_start, k_start, v_start):
        # 2x2x2 Strassen
        # Blocks are 2x2.
        # Local indices 0,1 for rows/cols.
        # Global indices:
        # u_start is the row offset for i.
        # k_start is the row offset for k.
        # v_start is the row offset for j.
        # But the blocks are contiguous in memory?
        # i=0,1 ; k=0,1 ; j=0,1 relative to starts.
        
        # Strassen formulas:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We need to expand these into rank-1 terms (scalar multiplications).
        # Since the elements are scalars (integers), each "block multiplication" like (A00+A11)(B00+B11)
        # is actually 4 scalar multiplications if done directly?
        # No, Strassen reduces the number of *matrix* multiplications.
        # Here, the "matrix" is 1x1.
        # So M1 is just a product of two sums.
        # (a00+a11)*(b00+b11) = a00b00 + a00b11 + a11b00 + a11b11.
        # This requires 4 multiplications if we distribute.
        # But we are constructing the tensor decomposition.
        # The rank of the 2x2x2 tensor is 7.
        # We need 7 terms (columns in U,V,W) that represent the 7 multiplications.
        
        # Let's define the 7 combinations of A and B.
        # A_blocks:
        # A00: i=0,k=0
        # A01: i=0,k=1
        # A10: i=1,k=0
        # A11: i=1,k=1
        
        # B_blocks:
        # B00: k=0,j=0
        # B01: k=0,j=1
        # B10: k=1,j=0
        # B11: k=1,j=1
        
        # We will create 7 new columns.
        
        # M1: (A00+A11)(B00+B11)
        # U terms: A00, A11
        # V terms: B00, B11
        # This term contributes to C00 and C11.
        # C00 += M1
        # C11 += M1
        
        # Instead of expanding into 4 scalar mults, we use the rank-1 structure.
        # Wait, if we treat A00 as a single variable, M1 is a product of two linear forms.
        # In the tensor decomposition, each rank-1 term is a product of one U, one V, one W.
        # For Strassen, we have 7 such terms.
        
        # Term 1:
        # U: A00 + A11
        # V: B00 + B11
        # W: C00 + C11 (since M1 adds to both)
        
        # Let's verify the coefficients.
        # C00 = A00B00 + A01B10
        # C01 = A00B01 + A01B11
        # C10 = A10B00 + A11B10
        # C11 = A10B01 + A11B11
        
        # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # M2 = (A10+A11)B00 = A10B00 + A11B00
        # M3 = A00(B01-B11) = A00B01 - A00B11
        # M4 = A11(B10-B00) = A11B10 - A11B00
        # M5 = (A00+A01)B11 = A00B11 + A01B11
        # M6 = (A10-A00)(B00+B01) = A10B00 + A10B01 - A00B00 - A00B01
        # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
        
        # C00 = M1 + M3 - M6
        #    = (A00B00 + A00B11 + A11B00 + A11B11) + (A00B01 - A00B11) - (A10B00 + A10B01 - A00B00 - A00B01)
        #    = A00B00 + A11B11 + A11B00 + A00B01 - A10B00 - A10B01 + A00B00 + A00B01
        #    = 2 A00B00 + A11B11 + A11B00 + 2 A00B01 - A10B00 - A10B01
        # This is incorrect. Strassen formulas are specific.
        
        # Correct Strassen:
        # P1 = (A11 - A00) * (B00 + B11)
        # P2 = (A00 + A11) * (B11 - B01)  -- No, standard is:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We will implement these 7 terms.
        
        # Map local indices to global
        # i: 0,1 -> u_start + i
        # k: 0,1 -> k_start + k
        # j: 0,1 -> v_start + j
        
        # U index for (i,k): (u_start + i) * m + (k_start + k)
        # V index for (k,j): (k_start + k) * p + (v_start + j)
        # W index for (j,i): (v_start + j) * n + (u_start + i)
        
        def get_u(i, k): return (u_start + i) * m + (k_start + k)
        def get_v(k, j): return (k_start + k) * p + (v_start + j)
        def get_w(j, i): return (v_start + j) * n + (u_start + i)
        
        # Term 1: M1
        # U: A00(1), A11(1)
        # V: B00(1), B11(1)
        # W: C00(1), C11(1)
        add_term(get_u(0,0), get_v(0,0), get_w(0,0), 1, 1, 1)
        add_term(get_u(0,0), get_v(1,1), get_w(0,0), 1, 1, 1)
        add_term(get_u(1,1), get_v(0,0), get_w(1,1), 1, 1, 1)
        add_term(get_u(1,1), get_v(1,1), get_w(1,1), 1, 1, 1)
        # Wait, M1 is ONE multiplication of two sums.
        # In the tensor decomposition, this corresponds to 4 rank-1 terms?
        # No. The rank is 7.
        # The 7 terms are the 7 multiplications M1..M7.
        # Each Mx is a product of two linear combinations of A and B.
        # To represent M1 = (A00+A11)(B00+B11) in the decomposition,
        # we need to express it as a sum of rank-1 tensors?
        # No, M1 is a single scalar multiplication in the Strassen algorithm.
        # The decomposition has 7 columns.
        # Column 1 corresponds to M1.
        # U[::, 0] has non-zeros at A00, A11.
        # V[::, 0] has non-zeros at B00, B11.
        # W[::, 0] has non-zeros at C00, C11.
        
        # So for M1:
        U[get_u(0,0)][col_idx] += 1
        U[get_u(1,1)][col_idx] += 1
        V[get_v(0,0)][col_idx] += 1
        V[get_v(1,1)][col_idx] += 1
        W[get_w(0,0)][col_idx] += 1
        W[get_w(1,1)][col_idx] += 1
        col_idx += 1
        
        # M2: (A10 + A11) B00
        U[get_u(1,0)][col_idx] += 1
        U[get_u(1,1)][col_idx] += 1
        V[get_v(0,0)][col_idx] += 1
        W[get_w(0,0)][col_idx] += -1 # C10 += M2, C11 -= M2
        W[get_w(1,0)][col_idx] += 1  # C10 is row 1, col 0. W index j*n+i. j=0, i=1 -> 0*n+1=1.
        # Wait, C10 corresponds to j=0, i=1.
        # C11 corresponds to j=1, i=1.
        # C10 += M2. C11 -= M2.
        # W[get_w(0,0)] is C00. W[get_w(0,1)] is C10? No.
        # W index is j*n+i.
        # C[i,j] is at W[j*n+i].
        # C10: i=1, j=0 -> W[0*n+1] = W[1].
        # C11: i=1, j=1 -> W[1*n+1] = W[6].
        
        # Let's re-verify W index mapping.
        # W[k*n+i] in prompt? "W[k*n+i,r]".
        # Prompt says: "W[k*n+i,r]" where k is col of B (j in my notation), i is row of A.
        # So W[j*n+i].
        
        # M2 contributes to C10 (+) and C11 (-).
        # C10: i=1, j=0 -> W[0*n+1] = W[1].
        # C11: i=1, j=1 -> W[1*n+1] = W[6].
        
        W[get_w(0,1)][col_idx] += 1
        W[get_w(1,1)][col_idx] += -1
        col_idx += 1
        
        # M3: A00 (B01 - B11)
        U[get_u(0,0)][col_idx] += 1
        V[get_v(0,1)][col_idx] += 1
        V[get_v(1,1)][col_idx] += -1
        # C01 += M3, C11 += M3
        # C01: i=0, j=1 -> W[1*n+0] = W[5].
        # C11: i=1, j=1 -> W[1*n+1] = W[6].
        W[get_w(1,0)][col_idx] += 1
        W[get_w(1,1)][col_idx] += 1
        col_idx += 1
        
        # M4: A11 (B10 - B00)
        U[get_u(1,1)][col_idx] += 1
        V[get_v(1,0)][col_idx] += 1
        V[get_v(0,0)][col_idx] += -1
        # C00 += M4, C10 += M4
        # C00: i=0, j=0 -> W[0*n+0] = W[0].
        # C10: i=1, j=0 -> W[0*n+1] = W[1].
        W[get_w(0,0)][col_idx] += 1
        W[get_w(0,1)][col_idx] += 1
        col_idx += 1
        
        # M5: (A00 + A01) B11
        U[get_u(0,0)][col_idx] += 1
        U[get_u(0,1)][col_idx] += 1
        V[get_v(1,1)][col_idx] += 1
        # C00 -= M5, C01 += M5
        # C00: W[0]. C01: W[5].
        W[get_w(0,0)][col_idx] += -1
        W[get_w(1,0)][col_idx] += 1
        col_idx += 1
        
        # M6: (A10 - A00) (B00 + B01)
        U[get_u(1,0)][col_idx] += 1
        U[get_u(0,0)][col_idx] += -1
        V[get_v(0,0)][col_idx] += 1
        V[get_v(0,1)][col_idx] += 1
        # C11 += M6
        # C11: W[6].
        W[get_w(1,1)][col_idx] += 1
        col_idx += 1
        
        # M7: (A01 - A11) (B10 + B11)
        U[get_u(0,1)][col_idx] += 1
        U[get_u(1,1)][col_idx] += -1
        V[get_v(1,0)][col_idx] += 1
        V[get_v(1,1)][col_idx] += 1
        # C00 += M7
        # C00: W[0].
        W[get_w(0,0)][col_idx] += 1
        col_idx += 1

    # Iterate over all 27 block products
    for k_g in range(3):
        for i_g in range(3):
            for j_g in range(3):
                # Get global start indices
                i_start = sum(g_sizes[g] for g in range(i_g))
                k_start = sum(g_sizes[g] for g in range(k_g))
                j_start = sum(g_sizes[g] for g in range(j_g))
                
                s_u = g_sizes[i_g]
                s_k = g_sizes[k_g]
                s_v = g_sizes[j_g]
                
                if s_u == 2 and s_k == 2 and s_v == 2:
                    add_strassen(i_start, k_start, j_start)
                else:
                    add_schoolbook(i_start, k_start, j_start, s_u, s_k, s_v)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
