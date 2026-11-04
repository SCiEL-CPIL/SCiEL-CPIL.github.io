"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # We'll build the decomposition using block partitioning with Strassen-like tricks.
    # Strategy: 
    # - Partition A (n x m) into row blocks and col blocks
    # - Partition B (m x p) into row blocks and col blocks  
    # - For each pair of sub-blocks, use either schoolbook or Strassen
    
    # Let's use a recursive approach:
    # Base case: if any dimension is 1, use schoolbook
    # Otherwise, split into halves and use Strassen for 2x2 block structure
    
    # For 4x5 x 5x5:
    # Split A: rows 2+2, cols 3+2
    # Split B: rows 3+2, cols 3+2
    # This gives 4 output blocks, each is sum of 2 products
    
    # Let me just do a direct construction with known rank reductions.
    
    # Actually, let me try: use the fact that we can combine some multiplications.
    # For now, let's implement a Strassen-based decomposition.
    
    # I'll use a helper to add a rank-1 term
    rank_terms = []  # list of (U_vec, V_vec, W_vec) where each is a dict or list
    
    # U has shape (n*m,), V has shape (m*p,), W has shape (p*n,)
    # Each rank term is a triple (u, v, w) where u[i*m+j], v[j*p+k], w[k*n+i]
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        """Add a rank-1 term with coefficient val at given indices."""
        rank_terms.append((u_idx, v_idx, w_idx, val))
    
    # Schoolbook baseline: rank = 100
    # Let's try to do better.
    
    # Approach: For the 4x5 x 5x5 multiplication, partition as follows:
    # A = [A1; A2] where A1 is 2x5, A2 is 2x5
    # B = [B1 | B2] where... no, B is 5x5.
    
    # Let me partition:
    # A (4x5): split rows into 2+2
    #   A_top (2x5), A_bot (2x5)
    # B (5x5): split cols into 3+2
    #   B_left (5x3), B_right (5x2)
    # C (4x5): 
    #   C_top_left (2x3) = A_top @ B_left
    #   C_top_right (2x2) = A_top @ B_right
    #   C_bot_left (2x3) = A_bot @ B_left
    #   C_bot_right (2x2) = A_bot @ B_right
    
    # Each of these is a matrix multiplication. Let's handle each:
    # C_top_left: 2x3 = (2x5) @ (5x3) -> 30 scalar mults schoolbook
    # C_top_right: 2x2 = (2x5) @ (5x2) -> 20 scalar mults
    # C_bot_left: 2x3 = (2x5) @ (5x3) -> 30 scalar mults
    # C_bot_right: 2x2 = (2x5) @ (5x2) -> 20 scalar mults
    # Total: 100
    
    # To reduce, I can further partition. Let's split the 5 in the middle:
    # A (4x5): rows 2+2, cols 3+2
    #   A11 (2x3), A12 (2x2)
    #   A21 (2x3), A22 (2x2)
    # B (5x5): rows 3+2, cols 3+2
    #   B11 (3x3), B12 (3x2)
    #   B21 (2x3), B22 (2x2)
    # C (4x5):
    #   C11 (2x3) = A11@B11 + A12@B21
    #   C12 (2x2) = A11@B12 + A12@B22
    #   C21 (2x3) = A21@B11 + A22@B21
    #   C22 (2x2) = A21@B12 + A22@B22
    
    # Now each sub-product:
    # A11@B11: (2x3)@(3x3) = 18 mults
    # A12@B21: (2x2)@(2x3) = 12 mults
    # A11@B12: (2x3)@(3x2) = 12 mults
    # A12@B22: (2x2)@(2x2) = 4 mults
    # A21@B11: (2x3)@(3x3) = 18 mults
    # A22@B21: (2x2)@(2x3) = 12 mults
    # A21@B12: (2x3)@(3x2) = 12 mults
    # A22@B22: (2x2)@(2x2) = 4 mults
    # Total: 18+12+12+4+18+12+12+4 = 92
    
    # That's already 8 less! Now I can apply Strassen to the 2x2 blocks:
    # A12@B22: (2x2)@(2x2) can be done in 7 mults instead of 4? No, Strassen is for square matrices of same size.
    # Actually Strassen works for any size if you pad, but let's see.
    
    # For (2x2)@(2x2): schoolbook is 4, Strassen is 7. That's worse!
    # Strassen is only beneficial for larger matrices.
    
    # For (2x3)@(3x3): 18 mults. Can I do better? 
    # For (2x3)@(3x2): 12 mults.
    
    # Hmm, Strassen's method is primarily for square matrices. For rectangular, it's less clear.
    
    # Let me try a different partition that allows Strassen.
    
    # Alternative: Make everything square-ish.
    # 4x5 x 5x5: pad A to 4x6? No, that changes the problem.
    
    # Let me think about this differently. The minimum rank for 4x5 x 5x5 is known.
    # It's at most 4*5*5 = 100 (schoolbook).
    # Lower bounds: at least max(4,5,5)^2 = 25? No, that's for square.
    
    # Actually, for m x n x p, the rank is at least max(m,n,p)^2 in some cases, but not always.
    
    # Let me just try to get a decomposition with rank < 100.
    
    # New approach: Use the fact that 5 = 3+2, and handle the 3 and 2 parts separately.
    # For the 2x2 subproblems, use schoolbook (4 mults each).
    # For the 2x3 and 3x3 subproblems, see if we can share multiplications.
    
    # Actually, let me just implement the 92-rank decomposition from above, which is already better than 100.
    
    # Let me code the 92-rank decomposition:
    
    # Indexing:
    # A[i,j] where i in 0..3, j in 0..4
    # B[j,k] where j in 0..4, k in 0..4
    # C[i,k] where i in 0..3, k in 0..4
    
    # Blocks:
    # A11: i in 0..1, j in 0..2  (2x3)
    # A12: i in 0..1, j in 3..4  (2x2)
    # A21: i in 2..3, j in 0..2  (2x3)
    # A22: i in 2..3, j in 3..4  (2x2)
    
    # B11: j in 0..2, k in 0..2  (3x3)
    # B12: j in 0..2, k in 3..4  (3x2)
    # B21: j in 3..4, k in 0..2  (2x3)
    # B22: j in 3..4, k in 3..4  (2x2)
    
    # C11: i in 0..1, k in 0..2  (2x3) = A11@B11 + A12@B21
    # C12: i in 0..1, k in 3..4  (2x2) = A11@B12 + A12@B22
    # C21: i in 2..3, k in 0..2  (2x3) = A21@B11 + A22@B21
    # C22: i in 2..3, k in 3..4  (2x2) = A21@B12 + A22@B22
    
    # For each sub-product, use schoolbook:
    
    terms = []  # (U_index, V_index, W_index)
    
    def add_schoolbook(a_i_range, a_j_range, b_j_range, b_k_range, c_i_range, c_k_range):
        """Add schoolbook terms for A[a_i_range, a_j_range] @ B[b_j_range, b_k_range] contributing to C[c_i_range, c_k_range]."""
        for i in c_i_range:
            for j in a_j_range:
                for k in b_k_range:
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    terms.append((u_idx, v_idx, w_idx))
    
    # A11@B11 -> C11
    add_schoolbook([0,1], [0,1,2], [0,1,2], [0,1,2], [0,1], [0,1,2])
    # A12@B21 -> C11
    add_schoolbook([0,1], [3,4], [3,4], [0,1,2], [0,1], [0,1,2])
    # A11@B12 -> C12
    add_schoolbook([0,1], [0,1,2], [0,1,2], [3,4], [0,1], [3,4])
    # A12@B22 -> C12
    add_schoolbook([0,1], [3,4], [3,4], [3,4], [0,1], [3,4])
    # A21@B11 -> C21
    add_schoolbook([2,3], [0,1,2], [0,1,2], [0,1,2], [2,3], [0,1,2])
    # A22@B21 -> C21
    add_schoolbook([2,3], [3,4], [3,4], [0,1,2], [2,3], [0,1,2])
    # A21@B12 -> C22
    add_schoolbook([2,3], [0,1,2], [0,1,2], [3,4], [2,3], [3,4])
    # A22@B22 -> C22
    add_schoolbook([2,3], [3,4], [3,4], [3,4], [2,3], [3,4])
    
    rank = len(terms)
    # Should be 92
    
    # Build U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_idx, v_idx, w_idx) in enumerate(terms):
        U[u_idx][r] = 1
        V[v_idx][r] = 1
        W[w_idx][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
