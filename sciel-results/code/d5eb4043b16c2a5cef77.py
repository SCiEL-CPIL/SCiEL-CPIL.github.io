"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x4 matrix multiplication.
    Returns dict with U, V, W factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = 3, 3, 4
    schoolbook_rank = n * m * p  # 36
    
    # Build the target tensor T[i,j,k] = 1 if we need A[i,j]*B[j,k]
    # The tensor has shape (n*m, m*p, p*n) = (9, 12, 12)
    # T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k) triples
    
    def build_target_tensor():
        """Build the 3x3x4 matrix multiplication tensor."""
        T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m + j, j*p + k, k*n + i] = 1.0
        return T
    
    T = build_target_tensor()
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that U @ diag @ V^T approximates T for each W column."""
        # For each rank element r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # should sum to T
        rank = U.shape[1]
        reconstructed = np.zeros_like(T, dtype=np.float64)
        for r in range(rank):
            reconstructed += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.max(np.abs(reconstructed - T)) < tol
    
    def to_half_integer_list(arr):
        """Convert numpy array to list of lists with half-integer values."""
        return [[float(x) for x in row] for row in arr]
    
    # Strategy 1: Strassen-based block decomposition
    # Split 3x3x4 into blocks:
    # A = [[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]] (3x3)
    # B = [[B11, B12, B13, B14], [B21, B22, B23, B24], [B31, B32, B33, B34]] (3x4)
    # 
    # We can group columns of B: first 2 columns form a 3x2 submatrix, last 2 form another
    # Or: use Strassen on 2x2 blocks and handle the rest
    #
    # Better approach: Use the known fact that M(3,3,4) can be decomposed as:
    # - A 2x2x2 Strassen block (rank 7) for the top-left 2x2 of A times first 2 cols of B
    # - Remaining terms
    #
    # Actually, let's try a direct approach: enumerate possible rank values and search
    
    # Start with schoolbook as fallback
    best_result = None
    best_rank = schoolbook_rank
    
    # Schoolbook decomposition
    U_sb = np.zeros((n*m, schoolbook_rank), dtype=np.float64)
    V_sb = np.zeros((m*p, schoolbook_rank), dtype=np.float64)
    W_sb = np.zeros((p*n, schoolbook_rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    best_rank = schoolbook_rank
    
    # Try structured decompositions
    # Approach: Use Strassen for 2x2x2 sub-blocks
    # 
    # Let's define the tensor indices more carefully:
    # A is 3x3: indices (i,j) for i in 0..2, j in 0..2
    # B is 3x4: indices (j,k) for j in 0..2, k in 0..3
    # C = A*B is 3x4: indices (i,k)
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Tensor T has T[i,j,k] = 1 meaning the term A[i,j]*B[j,k] contributes to C[i,k]
    # In the factorization: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # where U is indexed by (i,j), V by (j,k), W by (k,i)
    
    # Try rank values from low to high
    # Known: M(2,2,2) = 7 (Strassen)
    # M(3,3,3) <= 21 (Winograd/Toom)
    # M(3,3,4) - let's try to find a good decomposition
    
    # Method: Construct decomposition by combining smaller blocks
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (2x2x2, Strassen rank 7)
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] (2x2x2, rank 4)
    # Block 3: A[0:2, 2:3] * B[2:3, 0:4] -> C[0:2, 0:4] (2x1x4, rank 8)
    # Block 4: A[2:3, 0:3] * B[0:3, 0:4] -> C[2:3, 0:4] (1x3x4, rank 12)
    # Total: 7 + 4 + 8 + 12 = 31
    
    # But we can do better with overlapping Strassen blocks
    
    # Let's try a more systematic approach:
    # Use the fact that we can share computation between blocks
    
    # Attempt: Construct a rank-30 decomposition manually
    # This is complex, so let's try a search approach instead
    
    # Random local search: start from schoolbook, try to merge/eliminate terms
    # This is expensive, so let's try structured approaches first
    
    # Structured approach: Use Strassen on 2x2x2 blocks
    # Partition:
    # Rows of A: group {0,1} and {2}
    # Cols of A / Rows of B: group {0,1} and {2}  
    # Cols of B: group {0,1} and {2,3}
    
    # This gives 4 blocks:
    # (2x2) x (2x2) -> 2x2 : Strassen rank 7
    # (2x2) x (2x2) -> 2x2 : rank 4 (simple)
    # (2x1) x (1x4) -> 2x4 : rank 8
    # (1x3) x (3x4) -> 1x4 : rank 12
    # Total: 7 + 4 + 8 + 12 = 31
    
    # Let's build this decomposition
    
    def build_strassen_2x2x2():
        """Return Strassen 2x2x2 decomposition as (U, V, W) with shape (4, 7), (4, 7), (4, 7)"""
        # Strassen's 7 products for 2x2 * 2x2:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)*f
        # M3 = a*(f-g)
        # M4 = d*(h-g)
        # M5 = (a+b)*(e+f)
        # M6 = (c-d)*(g+h)
        # M7 = (a-c)*(e+d)
        # 
        # U factors (for A elements):
        # M1: a + d -> [1, 0, 0, 1]
        # M2: c + d -> [0, 0, 1, 1]
        # M3: a     -> [1, 0, 0, 0]
        # M4: d     -> [0, 0, 0, 1]
        # M5: a + b -> [1, 1, 0, 0]
        # M6: c - d -> [0, 0, 1, -1]
        # M7: a - c -> [1, 0, -1, 0]
        
        U = np.array([
            [1, 0, 1, 0, 1, 0, 1],   # a
            [0, 0, 0, 0, 1, 0, 0],   # b
            [0, 1, -1, 0, 0, 1, -1], # c
            [1, 1, 0, 1, 0, -1, 0]   # d
        ], dtype=np.float64)
        
        # V factors (for B elements): e,f,g,h
        # M1: e + h -> [1, 0, 0, 1]
        # M2: f     -> [0, 1, 0, 0]
        # M3: f - g -> [0, 1, -1, 0]
        # M4: h - g -> [0, 0, -1, 1]
        # M5: e + f -> [1, 1, 0, 0]
        # M6: g + h -> [0, 0, 1, 1]
        # M7: e + d -> [1, 0, 0, 1]  (wait, this should be e+d for M7? Let me recheck)
        # 
        # Actually M7 = (a-c)*(e+d), so V factor is [1, 0, 0, 1]
        # But M1 also uses [1,0,0,1]. That's fine, they're different products.
        
        V = np.array([
            [1, 0, 0, 0, 1, 0, 1],   # e
            [0, 1, 1, 0, 1, 0, 0],   # f
            [0, 0, -1, -1, 0, 1, 0], # g
            [1, 0, 0, 1, 0, 1, 1]    # h
        ], dtype=np.float64)
        
        # W factors (for C elements): p,q,r,s where C = [[p,q],[r,s]]
        # C[0,0] = M1 - M4 + M6 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M3 + M5 - M6
        
        W = np.array([
            [1, 0, 0, 0, 1, 0, 1],   # p = C[0,0]
            [0, 0, 1, 0, 1, 0, 0],   # q = C[0,1]
            [0, 1, 0, 1, 0, 0, 0],   # r = C[1,0]
            [1, 0, -1, 0, -1, -1, 1] # s = C[1,1]
        ], dtype=np.float64)
        
        return U, V, W
    
    # Verify Strassen
    U_s, V_s, W_s = build_strassen_2x2x2()
    T_s = np.zeros((4, 4, 4), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T_s[i*2+j, j*2+k, k*2+i] = 1.0
    recon_s = np.einsum('ar,br,cr->abc', U_s, V_s, W_s)
    strassen_valid = np.max(np.abs(recon_s - T_s)) < 1e-9
    
    if not strassen_valid:
        print("Strassen verification failed, using schoolbook only")
        return {"U": to_half_integer_list(U_sb), "V": to_half_integer_list(V_sb), "W": to_half_integer_list(W_sb)}
    
    # Build the block decomposition
    # Block structure:
    # A rows: [0,1] (Strassen) and [2]
    # A cols / B rows: [0,1] (Strassen) and [2]
    # B cols: [0,1] and [2,3]
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] : Strassen rank 7
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] : simple rank 4
    # Block 3: A[0:2, 2:3] * B[2:3, 0:4] -> C[0:2, 0:4] : rank 8
    # Block 4: A[2:3, 0:3] * B[0:3, 0:4] -> C[2:3, 0:4] : rank 12
    # Total: 31
    
    # Let's build this
    rank1 = 7   # Strassen
    rank2 = 4   # Simple 2x2x2
    rank3 = 8   # 2x1x4
    rank4 = 12  # 1x3x4
    total_rank = rank1 + rank2 + rank3 + rank4  # 31
    
    U_block = np.zeros((n*m, total_rank), dtype=np.float64)
    V_block = np.zeros((m*p, total_rank), dtype=np.float64)
    W_block = np.zeros((p*n, total_rank), dtype=np.float64)
    
    # Block 1: Strassen for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # A indices: (0,0),(0,1),(1,0),(1,1) -> flat: 0,1,2,3
    # B indices: (0,0),(0,1),(1,0),(1,1) -> flat: 0,1,2,3
    # C indices: (0,0),(0,1),(1,0),(1,1) -> flat: 0,1,2,3 (since C[k*n+i] with k in 0,1 and i in 0,1)
    
    offset = 0
    for r in range(7):
        # U: A[0:2, 0:2]
        U_block[0, offset+r] = U_s[0, r]  # A[0,0]
        U_block[1, offset+r] = U_s[1, r]  # A[0,1]
        U_block[2, offset+r] = U_s[2, r]  # A[1,0]
        U_block[3, offset+r] = U_s[3, r]  # A[1,1]
        
        # V: B[0:2, 0:2]
        V_block[0, offset+r] = V_s[0, r]  # B[0,0]
        V_block[1, offset+r] = V_s[1, r]  # B[0,1]
        V_block[2, offset+r] = V_s[2, r]  # B[1,0]
        V_block[3, offset+r] = V_s[3, r]  # B[1,1]
        
        # W: C[0:2, 0:2] -> C[k*n+i]
        W_block[0, offset+r] = W_s[0, r]  # C[0,0]
        W_block[1, offset+r] = W_s[1, r]  # C[0,1]
        W_block[2, offset+r] = W_s[2, r]  # C[1,0]
        W_block[3, offset+r] = W_s[3, r]  # C[1,1]
    
    offset += rank1
    
    # Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    # B cols 2,3: B[0,2],B[0,3],B[1,2],B[1,3] -> flat: 2,3,4,5 (since B[j*p+k], j=0,1; k=2,3)
    # C cols 2,3: C[0,2],C[0,3],C[1,2],C[1,3] -> flat: 6,7,8,9 (since C[k*n+i], k=2,3; i=0,1)
    
    # Simple multiplication: 4 terms
    b_cols = [2, 3]  # k indices
    for j in range(2):  # j = 0,1
        for k_idx, k in enumerate(b_cols):
            col = offset + j*2 + k_idx
            # U: A[0:2, j]
            U_block[0*m + j, col] = 1.0  # A[0,j]
            U_block[1*m + j, col] = 1.0  # A[1,j]
            # V: B[j, k]
            V_block[j*p + k, col] = 1.0
            # W: C[k, 0:2]
            W_block[k*n + 0, col] = 1.0  # C[k,0]
            W_block[k*n + 1, col] = 1.0  # C[k,1]
    
    offset += rank2
    
    # Block 3: A[0:2, 2] * B[2, 0:4] -> C[0:2, 0:4]
    # A col 2: A[0,2], A[1,2] -> flat: 2, 3
    # B row 2: B[2,0],B[2,1],B[2,2],B[2,3] -> flat: 8,9,10,11
    # C: C[0:2, 0:4] -> C[k*n+i] for k in 0..3, i in 0,1
    
    for i in range(2):  # i = 0,1
        for k in range(4):  # k = 0,1,2,3
            col = offset + i*4 + k
            U_block[i*m + 2, col] = 1.0  # A[i,2]
            V_block[2*p + k, col] = 1.0  # B[2,k]
            W_block[k*n + i, col] = 1.0  # C[k,i]
    
    offset += rank3
    
    # Block 4: A[2, 0:3] * B[0:3, 0:4] -> C[2, 0:4]
    # A row 2: A[2,0],A[2,1],A[2,2] -> flat: 6,7,8
    # B: all 3x4 -> 12 elements
    # C: C[2, 0:4] -> C[k*n+2] for k in 0..3
    
    for j in range(3):
        for k in range(4):
            col = offset + j*4 + k
            U_block[2*m + j, col] = 1.0  # A[2,j]
            V_block[j*p + k, col] = 1.0  # B[j,k]
            W_block[k*n + 2, col] = 1.0  # C[k,2]
    
    # Verify the block decomposition
    if verify_decomposition(U_block, V_block, W_block):
        best_result = (U_block, V_block, W_block)
        best_rank = total_rank
    
    # Try to improve: Can we do better than 31?
    # Let's try a different block structure
    
    # Alternative: Use Strassen more aggressively
    # Or try to find a rank-30 decomposition
    
    # For now, if we have a valid rank-31, that's better than 36
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 10.0:
        # Not enough time for more search
        U_final, V_final, W_final = best_result
        return {"U": to_half_integer_list(U_final), "V": to_half_integer_list(V_final), "W": to_half_integer_list(W_final)}
    
    # Try a few more structured approaches
    # Approach: Use rank-21 for 3x3x3 and add one column
    # M(3,3,3) rank <= 21, then M(3,3,4) = M(3,3,3) + M(3,3,1)
    # M(3,3,1) rank = 9 (since it's 3*3*1 = 9 simple terms)
    # Total: 21 + 9 = 30
    
    # But we need an actual rank-21 decomposition for 3x3x3
    # This is complex to implement correctly
    
    # Let's try: Can we modify the rank-31 to get rank-30?
    # This requires finding linear dependencies among the 31 terms
    
    # For now, return the best we have
    U_final, V_final, W_final = best_result
    
    # Final verification
    if not verify_decomposition(U_final, V_final, W_final):
        # Fallback to schoolbook
        U_final, V_final, W_final = U_sb, V_sb, W_sb
    
    return {"U": to_half_integer_list(U_final), "V": to_half_integer_list(V_final), "W": to_half_integer_list(W_final)}
# EVOLVE-BLOCK-END
