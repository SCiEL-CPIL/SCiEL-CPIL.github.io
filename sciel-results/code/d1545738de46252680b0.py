"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    rank = 16
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 16)
    V = [[0] * rank for _ in range(m * p)]  # (10, 16)
    W = [[0] * rank for _ in range(p * n)]  # (10, 16)
    
    # Strassen block 1: columns 0-1 of B
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B indices for cols 0-1: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C indices: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    # W index: W[k*n + i, r] = W[k*2 + i, r]
    
    # M1 = (A[1,1]) * (B[0,0] - B[1,1])
    # M2 = (A[0,1] + A[1,1]) * (B[1,1])
    # M3 = (A[0,0] - A[1,1]) * (B[0,0] + B[0,1])
    # M4 = (A[0,0]) * (B[1,1] - B[0,1])
    # M5 = (A[0,0] + A[0,1]) * (B[0,0])
    # M6 = (A[1,0] - A[0,0]) * (B[0,1] + B[1,1])
    # M7 = (A[0,1] - A[1,0]) * (B[1,0] - B[0,0])
    
    # C[0,0] = M5 + M4 - M2 + M6
    # C[0,1] = M1 + M5
    # C[1,0] = M3 + M4
    # C[1,1] = M1 - M3 + M2 + M7
    
    # Column 0 of U/V/W for M1
    U[3][0] = 1          # A[1,1]
    V[0][0] = 1          # B[0,0]
    V[3][0] = -1         # B[1,1]
    W[0][0] = 1          # C[0,1] -> k=1, i=0 -> W[1*2+0]=W[2]... wait
    # W[k*n+i, r]: k is column of C, i is row of C
    # C[0,1]: i=0, k=1 -> W[1*2+0] = W[2]
    # C[1,1]: i=1, k=1 -> W[1*2+1] = W[3]
    W[2][0] = 1          # C[0,1]
    W[3][0] = 1          # C[1,1]
    
    # Column 1 for M2
    U[1][1] = 1          # A[0,1]
    U[3][1] = 1          # A[1,1]
    V[3][1] = 1          # B[1,1]
    W[0][1] = -1         # C[0,0]: k=0,i=0 -> W[0]
    W[3][1] = 1          # C[1,1]: k=1,i=1 -> W[3]
    
    # Column 2 for M3
    U[0][2] = 1          # A[0,0]
    U[3][2] = -1         # A[1,1]
    V[0][2] = 1          # B[0,0]
    V[1][2] = 1          # B[0,1]
    W[2][2] = 1          # C[1,0]: k=0,i=1 -> W[0*2+1]=W[1]... 
    
    # Wait, let me reclarify. C is 2x5. C[i][k] where i in {0,1}, k in {0,1,2,3,4}
    # W[k*n + i, r] = W[k*2 + i, r]
    # C[0,0]: i=0, k=0 -> W[0]
    # C[0,1]: i=0, k=1 -> W[2]
    # C[0,2]: i=0, k=2 -> W[4]
    # C[0,3]: i=0, k=3 -> W[6]
    # C[0,4]: i=0, k=4 -> W[8]
    # C[1,0]: i=1, k=0 -> W[1]
    # C[1,1]: i=1, k=1 -> W[3]
    # C[1,2]: i=1, k=2 -> W[5]
    # C[1,3]: i=1, k=3 -> W[7]
    # C[1,4]: i=1, k=4 -> W[9]
    
    # Let me redo this properly.
    
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_idx, v_indices, w_indices, col):
        """Add a rank-1 term with coefficients."""
        for u_i, coeff in u_idx:
            U[u_i][col] += coeff
        for v_i, coeff in v_indices:
            V[v_i][col] += coeff
        for w_i, coeff in w_indices:
            W[w_i][col] += coeff
    
    # Strassen Block 1: A (2x2) * B[:, 0:2] (2x2) -> C[:, 0:2] (2x2)
    # A: [0,1; 2,3] -> A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B[:,0:2]: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C[:,0:2]: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    # W indices: C[0,0]->W[0], C[0,1]->W[2], C[1,0]->W[1], C[1,1]->W[3]
    
    # M1 = A[1,1] * (B[0,0] - B[1,1])
    add_term([(3, 1)], [(0, 1), (3, -1)], [(2, 1), (3, 1)], 0)
    
    # M2 = (A[0,1] + A[1,1]) * B[1,1]
    add_term([(1, 1), (3, 1)], [(3, 1)], [(0, -1), (3, 1)], 1)
    
    # M3 = (A[0,0] - A[1,1]) * (B[0,0] + B[0,1])
    add_term([(0, 1), (3, -1)], [(0, 1), (1, 1)], [(1, 1), (3, -1)], 2)
    
    # M4 = A[0,0] * (B[1,1] - B[0,1])
    add_term([(0, 1)], [(3, 1), (1, -1)], [(0, 1), (1, 1)], 3)
    
    # M5 = (A[0,0] + A[0,1]) * B[0,0]
    add_term([(0, 1), (1, 1)], [(0, 1)], [(0, 1), (2, 1)], 4)
    
    # M6 = (A[1,0] - A[0,0]) * (B[0,1] + B[1,1])
    add_term([(2, 1), (0, -1)], [(1, 1), (3, 1)], [(0, 1)], 5)
    
    # M7 = (A[0,1] - A[1,0]) * (B[1,0] - B[0,0])
    add_term([(1, 1), (2, -1)], [(2, 1), (0, -1)], [(3, 1)], 6)
    
    # Strassen Block 2: A (2x2) * B[:, 2:4] (2x2) -> C[:, 2:4] (2x2)
    # B[:,2:4]: B[0,2]=4, B[0,3]=5, B[1,2]=6, B[1,3]=7
    # C[:,2:4]: C[0,2]=4, C[0,3]=5, C[1,2]=6, C[1,3]=7
    # W indices: C[0,2]->W[4], C[0,3]->W[6], C[1,2]->W[5], C[1,3]->W[7]
    
    # M8 = A[1,1] * (B[0,2] - B[1,3])
    add_term([(3, 1)], [(4, 1), (7, -1)], [(6, 1), (7, 1)], 7)
    
    # M9 = (A[0,1] + A[1,1]) * B[1,3]
    add_term([(1, 1), (3, 1)], [(7, 1)], [(4, -1), (7, 1)], 8)
    
    # M10 = (A[0,0] - A[1,1]) * (B[0,2] + B[0,3])
    add_term([(0, 1), (3, -1)], [(4, 1), (5, 1)], [(5, 1), (7, -1)], 9)
    
    # M11 = A[0,0] * (B[1,3] - B[0,3])
    add_term([(0, 1)], [(7, 1), (5, -1)], [(4, 1), (5, 1)], 10)
    
    # M12 = (A[0,0] + A[0,1]) * B[0,2]
    add_term([(0, 1), (1, 1)], [(4, 1)], [(4, 1), (6, 1)], 11)
    
    # M13 = (A[1,0] - A[0,0]) * (B[0,3] + B[1,3])
    add_term([(2, 1), (0, -1)], [(5, 1), (7, 1)], [(4, 1)], 12)
    
    # M14 = (A[0,1] - A[1,0]) * (B[1,2] - B[0,2])
    add_term([(1, 1), (2, -1)], [(6, 1), (4, -1)], [(7, 1)], 13)
    
    # Schoolbook for column 4: A (2x2) * B[:,4] (2x1) -> C[:,4] (2x1)
    # B[0,4]=8, B[1,4]=9
    # C[0,4]=10, C[1,4]=11
    # W indices: C[0,4]->W[8], C[1,4]->W[9]
    
    # C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4]
    # C[1,4] = A[1,0]*B[0,4] + A[1,1]*B[1,4]
    
    # Term 1: A[0,0] * B[0,4]
    add_term([(0, 1)], [(8, 1)], [(8, 1)], 14)
    
    # Term 2: A[0,1] * B[1,4]
    add_term([(1, 1)], [(9, 1)], [(8, 1)], 15)
    
    # Wait, I need C[1,4] too. Let me reconsider.
    # C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4]  -> two terms
    # C[1,4] = A[1,0]*B[0,4] + A[1,1]*B[1,4]  -> two terms
    # That's 4 terms, not 2. I was wrong about rank 2.
    
    # Actually for 2x2 * 2x1, the rank is 2 (it's a bilinear map from R^2 x R^2 -> R^2)
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # This can be done with rank 2:
    # T1 = A[0,0]*B[0] + A[1,0]*B[0] = (A[0,0]+A[1,0])*B[0]  -- no, that doesn't work directly
    
    # Actually, the rank of the 2x2x1 tensor is 2. We can write:
    # C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4]
    # C[1,4] = A[1,0]*B[0,4] + A[1,1]*B[1,4]
    
    # Rank 2 decomposition:
    # Term 14: (A[0,0] + A[1,0]) * B[0,4] contributes to both C[0,4] and C[1,4]
    # But that gives C[0,4] += (A[0,0]+A[1,0])*B[0,4] and C[1,4] += (A[0,0]+A[1,0])*B[0,4]
    # That's not right.
    
    # Let me think again. The matrix multiplication 2x2 * 2x1 is equivalent to:
    # [a b] [e]   [ae+bf]
    # [c d] [f] = [ce+df]
    
    # Rank 2: 
    # T1: a*e -> contributes to C[0]
    # T2: (b*f + c*e + d*f - a*e)... no.
    
    # Actually, the standard result is that the rank of <2,2,1> is 2.
    # C[0] = a*e + b*f
    # C[1] = c*e + d*f
    # = (a*e + c*e) + (b*f + d*f) - ... no.
    
    # Let me just use rank 4 (schoolbook) for safety, or find the rank 2.
    # Actually, I recall that rank(<2,2,1>) = 2.
    # C[0] = a*e + b*f
    # C[1] = c*e + d*f
    # Let x = a + c, y = b + d, z = a - c, w = b - d
    # Hmm, let me try:
    # T1 = (a + c) * e -> contributes (a*e + c*e) to... no, it's a single scalar.
    
    # I think I'm overcomplicating. Let me just use rank 4 for the last column.
    # Total rank = 7 + 7 + 4 = 18.
    
    # But wait, the knowledge says rank 18 was achieved. Let me use that.
    
    # Actually, let me reconsider. For 2x2 * 2x1:
    # The tensor has entries T[i,j,k] where i in {0,1} (row of A), j in {0,1} (col of A = row of B), k in {0} (col of B = col of C)
    # T[0,0,0]=1, T[0,1,0]=1, T[1,0,0]=1, T[1,1,0]=1
    # This is the all-ones 2x2x1 tensor, which has rank 2.
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # Rank 2: T1 = A[0,0]*B[0], T2 = A[0,1]*B[1] gives C[0] but not C[1].
    # We need: C[0] = u1*v1*w1 + u2*v2*w2, C[1] = u1'*v1'*w1' + u2'*v2'*w2'
    # With 2 terms: 
    # Term 1: (A[0,0] + A[1,0]) * B[0] * 1 -> but this gives same value to C[0] and C[1]
    
    # I'll just use rank 4 for simplicity. Total = 18.
    
    # Let me restart with rank 18.
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_idx, v_indices, w_indices, col):
        for u_i, coeff in u_idx:
            U[u_i][col] += coeff
        for v_i, coeff in v_indices:
            V[v_i][col] += coeff
        for w_i, coeff in w_indices:
            W[w_i][col] += coeff
    
    # Strassen Block 1: cols 0-1
    add_term([(3, 1)], [(0, 1), (3, -1)], [(2, 1), (3, 1)], 0)   # M1
    add_term([(1, 1), (3, 1)], [(3, 1)], [(0, -1), (3, 1)], 1)    # M2
    add_term([(0, 1), (3, -1)], [(0, 1), (1, 1)], [(1, 1), (3, -1)], 2)  # M3
    add_term([(0, 1)], [(3, 1), (1, -1)], [(0, 1), (1, 1)], 3)    # M4
    add_term([(0, 1), (1, 1)], [(0, 1)], [(0, 1), (2, 1)], 4)     # M5
    add_term([(2, 1), (0, -1)], [(1, 1), (3, 1)], [(0, 1)], 5)    # M6
    add_term([(1, 1), (2, -1)], [(2, 1), (0, -1)], [(3, 1)], 6)   # M7
    
    # Strassen Block 2: cols 2-3
    add_term([(3, 1)], [(4, 1), (7, -1)], [(6, 1), (7, 1)], 7)    # M8
    add_term([(1, 1), (3, 1)], [(7, 1)], [(4, -1), (7, 1)], 8)    # M9
    add_term([(0, 1), (3, -1)], [(4, 1), (5, 1)], [(5, 1), (7, -1)], 9)   # M10
    add_term([(0, 1)], [(7, 1), (5, -1)], [(4, 1), (5, 1)], 10)   # M11
    add_term([(0, 1), (1, 1)], [(4, 1)], [(4, 1), (6, 1)], 11)    # M12
    add_term([(2, 1), (0, -1)], [(5, 1), (7, 1)], [(4, 1)], 12)   # M13
    add_term([(1, 1), (2, -1)], [(6, 1), (4, -1)], [(7, 1)], 13)  # M14
    
    # Schoolbook for col 4: 4 terms
    # C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4]
    # C[1,4] = A[1,0]*B[0,4] + A[1,1]*B[1,4]
    add_term([(0, 1)], [(8, 1)], [(8, 1)], 14)   # A[0,0]*B[0,4] -> C[0,4]
    add_term([(1, 1)], [(9, 1)], [(8, 1)], 15)   # A[0,1]*B[1,4] -> C[0,4]
    add_term([(2, 1)], [(8, 1)], [(9, 1)], 16)   # A[1,0]*B[0,4] -> C[1,4]
    add_term([(3, 1)], [(9, 1)], [(9, 1)], 17)   # A[1,1]*B[1,4] -> C[1,4]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
