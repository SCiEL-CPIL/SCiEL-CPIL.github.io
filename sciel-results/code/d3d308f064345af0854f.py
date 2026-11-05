"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
_start_time = time.time()
TIME_LIMIT = 120  # seconds

DIMENSIONS = [3, 4, 4]

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct."""
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i*m + j, r] * V_arr[j*p + k, r] * W_arr[k*n + i, r]
                error += abs(val - 1.0)
    return error < 1e-10

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 44
    
    # Initialize U, V, W as lists of lists
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add terms to the matrices
    # We will accumulate terms in a list of tuples first to avoid index management issues
    # Tuple: (u_idx, u_val), (v_idx, v_val), (w_idx, w_val)
    # Actually, since rank is fixed, we can just fill a 3D array or list of columns.
    # Let's use a list of columns: col = (u_vec, v_vec, w_vec)
    # u_vec is dict {idx: val} or list of length n*m
    
    cols = []
    
    def add_term(u_entries, v_entries, w_entries):
        """
        u_entries: list of (idx, val)
        v_entries: list of (idx, val)
        w_entries: list of (idx, val)
        """
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        
        for idx, val in u_entries:
            u_vec[idx] = val
        for idx, val in v_entries:
            v_vec[idx] = val
        for idx, val in w_entries:
            w_vec[idx] = val
            
        cols.append((u_vec, v_vec, w_vec))
        
    def add_strassen(a,b,c,d, e,f,g,h, w11,w12,w21,w22):
        """
        Adds 7 terms for 2x2x2 Strassen decomposition.
        a,b,c,d are U indices for A block.
        e,f,g,h are V indices for B block.
        w11,w12,w21,w22 are W indices for C block.
        """
        # M1 = (a+d)(e+h)
        add_term([(a,1),(d,1)], [(e,1),(h,1)], [(w11,1),(w22,1)])
        
        # M2 = (c+d)e
        # Contributes to C21 (+1) and C22 (-1)
        add_term([(c,1),(d,1)], [(e,1)], [(w21,1),(w22,-1)])
        
        # M3 = a(f-h)
        # Contributes to C12 (+1) and C22 (+1)
        add_term([(a,1)], [(f,1),(h,-1)], [(w12,1),(w22,1)])
        
        # M4 = d(g-e)
        # Contributes to C11 (+1) and C21 (+1)
        add_term([(d,1)], [(g,1),(e,-1)], [(w11,1),(w21,1)])
        
        # M5 = (a+b)h
        # Contributes to C12 (+1) and C11 (-1)
        add_term([(a,1),(b,1)], [(h,1)], [(w12,1),(w11,-1)])
        
        # M6 = (c-a)(e+f)
        # Contributes to C22 (+1)
        add_term([(c,1),(a,-1)], [(e,1),(f,1)], [(w22,1)])
        
        # M7 = (b-d)(g+h)
        # Contributes to C11 (+1)
        add_term([(b,1),(d,-1)], [(g,1),(h,1)], [(w11,1)])
        
    def add_rank4(a,b, c,d,e,f, w1,w2):
        """
        Adds 4 terms for 1x2x2 multiplication.
        a,b are U indices.
        c,d,e,f are V indices (flattened 2x2 B: c=d? No. B=[[c,d],[e,f]]).
        w1,w2 are W indices.
        C1 = a*c + b*e
        C2 = a*d + b*f
        """
        add_term([(a,1)], [(c,1)], [(w1,1)])
        add_term([(b,1)], [(e,1)], [(w1,1)])
        add_term([(a,1)], [(d,1)], [(w2,1)])
        add_term([(b,1)], [(f,1)], [(w2,1)])

    # Indices for A (3x4)
    # Row 0: 0, 1, 2, 3
    # Row 1: 4, 5, 6, 7
    # Row 2: 8, 9, 10, 11
    
    # Indices for B (4x4)
    # Row 0: 0, 1, 2, 3
    # Row 1: 4, 5, 6, 7
    # Row 2: 8, 9, 10, 11
    # Row 3: 12, 13, 14, 15
    
    # Indices for C (3x4) -> W (k*3+i)
    # Col 0: 0, 1, 2
    # Col 1: 3, 4, 5
    # Col 2: 6, 7, 8
    # Col 3: 9, 10, 11
    
    # Case 1: A_LL (R0,1 C0,1) * B_LL1 (R0,1 C0,1) -> C(R0,1 C0,1)
    add_strassen(
        0, 1, 4, 5,   # A
        0, 1, 4, 5,   # B
        0, 3, 1, 4    # W: C[0,0]=0, C[0,1]=3, C[1,0]=1, C[1,1]=4
    )
    
    # Case 2: A_LL (R0,1 C0,1) * B_LL2 (R0,1 C2,3) -> C(R0,1 C2,3)
    add_strassen(
        0, 1, 4, 5,   # A
        2, 3, 6, 7,   # B
        6, 9, 7, 10   # W: C[0,2]=6, C[0,3]=9, C[1,2]=7, C[1,3]=10
    )
    
    # Combined Case 3+7: A row 2, cols 0..3 × B rows 0..1, cols 0..1 -> C[2, 0..1]
    # This is 1x4 x 2x2 -> 1x2. Use rank-7 Strassen on the 2x2 part with A split.
    # A[2,0]=8, A[2,1]=9, A[2,2]=10, A[2,3]=11
    # B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # C[2,0]=2, C[2,1]=5
    # 
    # Split A into (A[2,0]+A[2,1]) and (A[2,2]+A[2,3]) won't work directly.
    # Instead, treat as two separate 1x2 x 2x2 blocks but share some terms.
    # 
    # Actually, let's use a different strategy: merge the four rank-4 blocks into
    # two 1x4 x 2x2 -> 1x2 problems, each solvable in rank 7 with Strassen-like.
    #
    # For 1x4 x 2x2 -> 1x2: C[j] = sum_{i=0}^{3} A[i] * B[i//2, j%2] for appropriate grouping
    # This doesn't have a simple Strassen structure. Let me try rank-4 for each and see
    # if I can save by sharing.
    
    # Keep the four rank-4 blocks but try to reduce total rank by 2.
    # Strategy: Use a rank-6 decomposition for the combined 1x4 x 2x2 -> 1x2 problem.
    
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[0,0] + A[2,3]*B[1,0]
    #         = (A[2,0]+A[2,2])*B[0,0] + (A[2,1]+A[2,3])*B[1,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[0,1] + A[2,3]*B[1,1]
    #         = (A[2,0]+A[2,2])*B[0,1] + (A[2,1]+A[2,3])*B[1,1]
    
    # Let P = A[2,0]+A[2,2] = A[8]+A[10], Q = A[2,1]+A[2,3] = A[9]+A[11]
    # Then C[2,0] = P*B[0,0] + Q*B[1,0], C[2,1] = P*B[0,1] + Q*B[1,1]
    # This is a 1x2 x 2x2 -> 1x2 problem! Rank 4.
    
    # Similarly for B rows 2..3:
    # C[2,0] += A[2,2]*B[2,0] + A[2,3]*B[3,0]  -- wait, this is wrong.
    
    # Let me re-examine. B is 4x4. Rows 0,1 are B[0..1, 0..1] for Case 3.
    # B[0,0]=idx 0, B[0,1]=idx 1, B[1,0]=idx 4, B[1,1]=idx 5
    # For Case 7: B rows 2,3: B[2,0]=idx 8, B[2,1]=idx 9, B[3,0]=idx 12, B[3,1]=idx 13
    
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0] + A[2,3]*B[3,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1] + A[2,3]*B[3,1]
    
    # Group: P = A[2,0]+A[2,2], Q = A[2,1]+A[2,3]
    # C[2,0] = P*(B[0,0]+B[2,0]) + Q*(B[1,0]+B[3,0])
    # C[2,1] = P*(B[0,1]+B[2,1]) + Q*(B[1,1]+B[3,1])
    
    # This is 1x2 x 2x2 -> 1x2, rank 4. But we need 4 terms.
    # Term 1: P * (B[0,0]+B[2,0]) -> C[2,0]
    # Term 2: Q * (B[1,0]+B[3,0]) -> C[2,0]  
    # Term 3: P * (B[0,1]+B[2,1]) -> C[2,1]
    # Term 4: Q * (B[1,1]+B[3,1]) -> C[2,1]
    
    # That's still rank 4 for this combined block. No savings yet.
    # Let me just keep the original four rank-4 blocks for now and verify correctness.
    
    add_rank4(
        8, 9,         # A
        0, 1, 4, 5,   # B
        2, 5          # W: C[2,0]=2, C[2,1]=5
    )
    
    add_rank4(
        8, 9,         # A
        2, 3, 6, 7,   # B
        8, 11         # W: C[2,2]=8, C[2,3]=11
    )
    
    add_strassen(
        2, 3, 6, 7,   # A
        8, 9, 12, 13, # B
        0, 3, 1, 4    # W: C[0,0]=0, C[0,1]=3, C[1,0]=1, C[1,1]=4
    )
    
    add_strassen(
        2, 3, 6, 7,   # A
        10, 11, 14, 15, # B
        6, 9, 7, 10   # W: C[0,2]=6, C[0,3]=9, C[1,2]=7, C[1,3]=10
    )
    
    add_rank4(
        10, 11,       # A
        8, 9, 12, 13, # B
        2, 5          # W: C[2,0]=2, C[2,1]=5
    )
    
    add_rank4(
        10, 11,       # A
        10, 11, 14, 15, # B
        8, 11         # W: C[2,2]=8, C[2,3]=11
    )
    
    # Fill U, V, W
    for r in range(rank):
        u_vec, v_vec, w_vec = cols[r]
        for i in range(n * m):
            U[i][r] = u_vec[i]
        for j in range(m * p):
            V[j][r] = v_vec[j]
        for k in range(p * n):
            W[k][r] = w_vec[k]
            
    # Verify decomposition
    try:
        import numpy as np
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Build target tensor
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1.0
        
        # Check decomposition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal T[i,j,k]
        # W index: k*n + i where k is column of C, i is row of C
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # W[k*n + i, r] corresponds to C[i,k]
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_arr[i*m + j, r] * V_arr[j*p + k, r] * W_arr[k*n + i, r]
                    error += abs(val - T[i, j, k])
        
        if error > 1e-10:
            # Fallback: return schoolbook
            rank_school = n * m * p
            U_s = [[0] * rank_school for _ in range(n * m)]
            V_s = [[0] * rank_school for _ in range(m * p)]
            W_s = [[0] * rank_school for _ in range(p * n)]
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_s[i*m + j][idx] = 1
                        V_s[j*p + k][idx] = 1
                        W_s[k*n + i][idx] = 1
                        idx += 1
            return {"U": U_s, "V": V_s, "W": W_s}
    except:
        pass
    
    if time.time() - _start_time > TIME_LIMIT - 5:
        # Emergency fallback
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m + j][idx] = 1
                    V_s[j*p + k][idx] = 1
                    W_s[k*n + i][idx] = 1
                    idx += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
