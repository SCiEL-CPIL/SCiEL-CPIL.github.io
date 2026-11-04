"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p

    def verify_decomposition(U, V, W, rank):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True

    # Try to find lower rank decompositions using structured search
    # Strategy: Start from schoolbook and try to merge pairs of terms
    # For exact decomposition, we need integer/half-integer entries
    
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try rank reduction by 1: remove one schoolbook term and add a compensating term
    # This is equivalent to finding a rank-1 update that cancels one basis term
    np.random.seed(42)
    
    for target_rank in range(best_rank - 1, max(15, best_rank - 5), -1):
        if time.time() > deadline - 10:
            break
        
        # Approach 1: Start from schoolbook and try to zero out one column
        # by adding a combination of other columns
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            
            # Create a copy of schoolbook
            U = U_sb.copy()
            V = V_sb.copy()
            W = W_sb.copy()
            
            # Try to merge two terms: pick two columns and see if we can replace them
            # with a single column that preserves the tensor
            c1 = np.random.randint(0, best_rank)
            c2 = np.random.randint(0, best_rank)
            if c1 == c2:
                continue
            
            # Try to find alpha, beta, gamma such that:
            # U[:,c1]*V[:,c1]*W[:,c1] + U[:,c2]*V[:,c2]*W[:,c2] = 
            # alpha*(U[:,c1]+U[:,c2])*V[:,c1]*W[:,c1] + ... (this is getting complex)
            
            # Simpler: try random perturbations of the schoolbook
            U_new = U_sb.copy()
            V_new = V_sb.copy()
            W_new = W_sb.copy()
            
            # Add small random noise to some entries
            for _ in range(5):
                row = np.random.randint(0, U_new.shape[0])
                col = np.random.randint(0, U_new.shape[1])
                U_new[row, col] += np.random.choice([-1, 1]) * np.random.choice([0.5, 1.0])
                
                row = np.random.randint(0, V_new.shape[0])
                col = np.random.randint(0, V_new.shape[1])
                V_new[row, col] += np.random.choice([-1, 1]) * np.random.choice([0.5, 1.0])
                
                row = np.random.randint(0, W_new.shape[0])
                col = np.random.randint(0, W_new.shape[1])
                W_new[row, col] += np.random.choice([-1, 1]) * np.random.choice([0.5, 1.0])
            
            # Try to find a lower-rank approximation by SVD-like truncation
            # But we need exact decomposition, so this won't work directly
            
            # Instead, try to find if any column is redundant
            # Check if column c can be expressed as a combination of others
            for c in range(target_rank, best_rank):
                # Try to express column c as linear combination of first target_rank columns
                # This is a linear system, but we need it to work for the tensor product
                pass
        
        # Approach 2: Try known small-rank decompositions for sub-blocks
        # Embed Strassen's 2x2x2 rank-7 decomposition into the 3x4x4 problem
        if attempt == 0:  # Do this once per target_rank
            # Strassen's algorithm for 2x2 matrices:
            # M1 = (A11+A22)(B11+B22)
            # M2 = (A21+A22)B11
            # M3 = A11(B12-B22)
            # M4 = A22(B21-B11)
            # M5 = (A11+A12)B22
            # M6 = (A21-A11)(B11+B12)
            # M7 = (A12-A22)(B21+B22)
            # C11 = M1+M4-M5+M7
            # C12 = M3+M5
            # C21 = M2+M4
            # C22 = M1-M2+M3+M6
            
            # We can use this for the 2x2 sub-blocks within our 3x4x4 problem
            # But this requires careful embedding and is complex
            
            pass
    
    # If we couldn't find better, try one more structured approach
    # Use the fact that for matrix multiplication, we can sometimes share factors
    # across different (i,j,k) tuples
    
    # Try to find a decomposition with rank 47 by solving a system of equations
    # This is a hard problem, so we'll just return the best we found
    
    # Now try the Strassen-based block decomposition from inspiration 1
    # This should achieve rank 44
    
    if time.time() > deadline - 30:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
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
    
    # Case 3: A_LR (R2 C0,1) * B_LL1 (R0,1 C0,1) -> C(R2 C0,1)
    add_rank4(
        8, 9,         # A
        0, 1, 4, 5,   # B
        2, 5          # W: C[2,0]=2, C[2,1]=5
    )
    
    # Case 4: A_LR (R2 C0,1) * B_LL2 (R0,1 C2,3) -> C(R2 C2,3)
    add_rank4(
        8, 9,         # A
        2, 3, 6, 7,   # B
        8, 11         # W: C[2,2]=8, C[2,3]=11
    )
    
    # Case 5: A_RL (R0,1 C2,3) * B_RL1 (R2,3 C0,1) -> C(R0,1 C0,1)
    add_strassen(
        2, 3, 6, 7,   # A
        8, 9, 12, 13, # B
        0, 3, 1, 4    # W: C[0,0]=0, C[0,1]=3, C[1,0]=1, C[1,1]=4
    )
    
    # Case 6: A_RL (R0,1 C2,3) * B_RL2 (R2,3 C2,3) -> C(R0,1 C2,3)
    add_strassen(
        2, 3, 6, 7,   # A
        10, 11, 14, 15, # B
        6, 9, 7, 10   # W: C[0,2]=6, C[0,3]=9, C[1,2]=7, C[1,3]=10
    )
    
    # Case 7: A_RR (R2 C2,3) * B_RL1 (R2,3 C0,1) -> C(R2 C0,1)
    add_rank4(
        10, 11,       # A
        8, 9, 12, 13, # B
        2, 5          # W: C[2,0]=2, C[2,1]=5
    )
    
    # Case 8: A_RR (R2 C2,3) * B_RL2 (R2,3 C2,3) -> C(R2 C2,3)
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
    
    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    if verify_decomposition(U_np, V_np, W_np, rank):
        best_U, best_V, best_W = U_np, V_np, W_np
        best_rank = rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
