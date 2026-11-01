import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    # We will construct a rank-34 decomposition based on block decomposition.
    # A is 2x3, B is 3x7, C is 2x7.
    # Split A into A1 (2x2) and A2 (2x1).
    # Split B into B1 (3x3), B2 (3x3), B3 (3x1) along the output dimension k.
    # Wait, the split of B was along the column dimension of B (which is the output dimension of C).
    # B is 3x7.
    # C = A B.
    # C = A1 B_top + A2 B_bottom.
    # A1 is 2x2 (cols 0,1 of A). B_top is 2x7 (rows 0,1 of B).
    # A2 is 2x1 (col 2 of A). B_bottom is 1x7 (row 2 of B).
    
    # Part 1: A1 (2x2) * B_top (2x7).
    # Split B_top (2x7) into B1 (2x3), B2 (2x3), B3 (2x1).
    # C1 = A1 B1 + A1 B2 + A1 B3.
    # A1 B1: 2x2 * 2x3.
    #   Split B1 (2x3) into B1a (2x2), B1b (2x1).
    #   A1 B1a: 2x2 * 2x2. Rank 7 (Strassen).
    #   A1 B1b: 2x2 * 2x1. Rank 2.
    #   Total 9.
    # A1 B2: 2x2 * 2x3. Rank 9.
    # A1 B3: 2x2 * 2x1. Rank 2.
    # Total Part 1: 9 + 9 + 2 = 20.
    
    # Part 2: A2 (2x1) * B_bottom (1x7).
    # C2 = A2 B_bottom.
    # This is outer product of 2x1 and 1x7.
    # Rank 2*7 = 14.
    
    # Total Rank: 20 + 14 = 34.
    
    R = 34
    U = np.zeros((n*m, R), dtype=np.int64)
    V = np.zeros((m*p, R), dtype=np.int64)
    W = np.zeros((p*n, R), dtype=np.int64)
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        nonlocal col
        U[u_idx, col] = u_val
        V[v_idx, col] = v_val
        W[w_idx, col] = w_val
        col += 1
    
    # Strassen decomposition for 2x2 * 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = c(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(f+g)
    # M7 = (b-d)(e+g)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def add_strassen(a_base, b_base, c_base, a_idx_map, b_idx_map, c_idx_map):
        # a_idx_map: maps logical 2x2 index (i,j) to actual A index
        # b_idx_map: maps logical 2x2 index (j,k) to actual B index
        # c_idx_map: maps logical 2x2 index (i,k) to actual C index
        
        # We need to form linear combinations of A and B entries.
        # Since we are building a tensor decomposition, we cannot pre-compute sums.
        # We must add terms for each multiplication.
        # This means we add 7 terms.
        
        # M1 = (a+d)(e+h) = ae + ah + de + dh
        # We can add 4 terms:
        add_term(a_idx_map[0,0], b_idx_map[0,0], c_idx_map[0,0], 1, 1, 1) # ae -> C11
        add_term(a_idx_map[0,0], b_idx_map[0,1], c_idx_map[0,1], 1, 1, 1) # ah -> C11? No.
        
        # Wait, Strassen is a decomposition of the tensor.
        # The tensor T for 2x2*2x2 has 8 terms.
        # Strassen replaces them with 7 terms.
        # The 7 terms are rank-1 tensors in the space of A, B, C.
        # But A, B, C are flattened.
        # We need to express the Strassen intermediates as linear combinations of the input vectors.
        
        # This approach is tricky to implement directly as a rank-1 sum without expanding the linear combinations.
        # If we expand, we get back to the schoolbook or worse.
        # To use Strassen, we must treat the linear combinations as part of the U, V, W matrices.
        
        # U column r has entries for A.
        # V column r has entries for B.
        # W column r has entries for C.
        
        # For M1 = (a+d)(e+h):
        # This is not a single rank-1 term in the original A, B space.
        # It is (a+d) * (e+h).
        # We can realize this by adding 4 terms:
        # a*e, a*h, d*e, d*h.
        # But then we have 8 terms again.
        
        # The magic of Strassen is that the 7 terms are:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # ...
        # And the output is a linear combination of these 7 values.
        # C11 = M1 + M4 - M5 + M7.
        
        # In the tensor decomposition T = sum U_r V_r W_r,
        # each term U_r V_r W_r contributes to the output.
        # If we use M1 as a "virtual" multiplication, we need to map the inputs to it and the output from it.
        
        # Input A for M1: a+d. So U_r has 1 at a and 1 at d.
        # Input B for M1: e+h. So V_r has 1 at e and 1 at h.
        # Output C for M1: M1 contributes to C11 and C22.
        # C11 = ... + M1 ...
        # C22 = ... + M1 ...
        # So W_r has 1 at C11 and 1 at C22.
        
        # This works!
        
        # Let's define the 7 Strassen terms for a 2x2 block.
        # Indices:
        # A: 00, 01, 10, 11
        # B: 00, 01, 10, 11
        # C: 00, 01, 10, 11
        
        # M1: A: 00, 10. B: 00, 11. C: 00, 11.
        # M2: A: 10, 11. B: 00. C: 10, 11.
        # M3: A: 00. B: 01, 11. C: 01. (with sign -1 for B11)
        # M4: A: 10. B: 10, 00. C: 00, 10. (with sign -1 for B00)
        # M5: A: 00, 01. B: 11. C: 00, 01.
        # M6: A: 10, 11. B: 01, 10. C: 11. (with signs)
        # M7: A: 01, 11. B: 00, 10. C: 00, 11. (with signs)
        
        # We will add these 7 terms.
        
        # M1
        U[a_idx_map[0,0], col] = 1
        U[a_idx_map[1,0], col] = 1
        V[b_idx_map[0,0], col] = 1
        V[b_idx_map[1,1], col] = 1
        W[c_idx_map[0,0], col] = 1
        W[c_idx_map[1,1], col] = 1
        col += 1
        
        # M2
        U[a_idx_map[1,0], col] = 1
        U[a_idx_map[1,1], col] = 1
        V[b_idx_map[0,0], col] = 1
        W[c_idx_map[1,0], col] = 1
        W[c_idx_map[1,1], col] = -1
        col += 1
        
        # M3
        U[a_idx_map[0,0], col] = 1
        V[b_idx_map[0,1], col] = 1
        V[b_idx_map[1,1], col] = -1
        W[c_idx_map[0,1], col] = 1
        col += 1
        
        # M4
        U[a_idx_map[1,0], col] = 1
        V[b_idx_map[1,0], col] = 1
        V[b_idx_map[0,0], col] = -1
        W[c_idx_map[0,0], col] = 1
        W[c_idx_map[1,0], col] = 1
        col += 1
        
        # M5
        U[a_idx_map[0,0], col] = 1
        U[a_idx_map[0,1], col] = 1
        V[b_idx_map[1,1], col] = 1
        W[c_idx_map[0,0], col] = -1
        W[c_idx_map[0,1], col] = 1
        col += 1
        
        # M6
        U[a_idx_map[1,0], col] = 1
        U[a_idx_map[1,1], col] = -1
        V[b_idx_map[0,1], col] = 1
        V[b_idx_map[1,0], col] = 1
        W[c_idx_map[1,1], col] = 1
        col += 1
        
        # M7
        U[a_idx_map[0,1], col] = 1
        U[a_idx_map[1,1], col] = -1
        V[b_idx_map[0,0], col] = 1
        V[b_idx_map[1,0], col] = 1
        W[c_idx_map[0,0], col] = 1
        W[c_idx_map[1,1], col] = -1
        col += 1
        
    # Helper for 2x2 * 2x1 (Rank 2)
    def add_2x2x1(a_base, b_base, c_base, a_idx_map, b_idx_map, c_idx_map):
        # A 2x2, B 2x1, C 2x1
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        # Term 1: A00 B0 -> C0
        # Term 2: A01 B1 -> C0
        # Term 3: A10 B0 -> C1
        # Term 4: A11 B1 -> C1
        # Wait, rank 2?
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        # This is 4 terms.
        # Can we do it with 2?
        # No, the flattening rank is 2, but the tensor rank?
        # T_{a,b,c}.
        # a in {00, 01, 10, 11}, b in {0, 1}, c in {0, 1}.
        # It is known that R(2,2,1) = 2?
        # Let's check.
        # C = A B.
        # This is a linear map.
        # If we fix B, C is linear in A.
        # The tensor is separable?
        # Actually, R(2,2,1) is 2.
        # M1 = (A00 + A10) B0
        # M2 = (A01 + A11) B1
        # C0 = M1 + M2? No.
        # C0 = A00 B0 + A01 B1.
        # C1 = A10 B0 + A11 B1.
        # This is not separable into 2 terms.
        # It requires 4 terms?
        # Let's assume 4 terms for safety, or 2 if I find a trick.
        # Actually, R(2,2,1) = 2 is incorrect. It is 4?
        # No, R(2,2,1) is the rank of the map A, B -> C.
        # It is equivalent to 2 dot products of length 2.
        # Rank 2?
        # Let's just use 4 terms. 2x2x1 is small.
        
        # Term 1
        U[a_idx_map[0,0], col] = 1
        V[b_idx_map[0], col] = 1
        W[c_idx_map[0], col] = 1
        col += 1
        
        # Term 2
        U[a_idx_map[0,1], col] = 1
        V[b_idx_map[1], col] = 1
        W[c_idx_map[0], col] = 1
        col += 1
        
        # Term 3
        U[a_idx_map[1,0], col] = 1
        V[b_idx_map[0], col] = 1
        W[c_idx_map[1], col] = 1
        col += 1
        
        # Term 4
        U[a_idx_map[1,1], col] = 1
        V[b_idx_map[1], col] = 1
        W[c_idx_map[1], col] = 1
        col += 1

    # Helper for 2x1 * 1x1 (Rank 2)
    def add_2x1x1(a_base, b_base, c_base, a_idx_map, b_idx_map, c_idx_map):
        # A 2x1, B 1x1, C 2x1
        # C0 = A0 B0
        # C1 = A1 B0
        # Term 1
        U[a_idx_map[0], col] = 1
        V[b_idx_map[0], col] = 1
        W[c_idx_map[0], col] = 1
        col += 1
        # Term 2
        U[a_idx_map[1], col] = 1
        V[b_idx_map[0], col] = 1
        W[c_idx_map[1], col] = 1
        col += 1

    # Helper for 2x1 * 1xk (Rank 2k)
    def add_2x1xk(a_base, b_base, c_base, a_idx_map, b_idx_map, c_idx_map, k):
        # A 2x1, B 1xk, C 2xk
        # C_i, j = A_i B_j
        for i in range(2):
            for j in range(k):
                U[a_idx_map[i], col] = 1
                V[b_idx_map[j], col] = 1
                W[c_idx_map[i, j], col] = 1
                col += 1

    # Mapping functions
    # A is 2x3. Index i*m+j.
    # B is 3x7. Index j*p+k.
    # C is 2x7. Index k*n+i.
    
    # Part 1: A1 (2x2) * B_top (2x7)
    # A1 uses A[0,0], A[0,1], A[1,0], A[1,1]
    # B_top uses B[0,0..6], B[1,0..6]
    
    # Split B_top into B1 (2x3), B2 (2x3), B3 (2x1)
    # B1: k=0,1,2. B2: k=3,4,5. B3: k=6.
    
    # A1 B1 (2x2 * 2x3)
    # Split B1 into B1a (2x2, k=0,1) and B1b (2x1, k=2)
    
    # A1 B1a (2x2 * 2x2) -> Strassen
    a_map_1 = lambda i, j: i*3 + j # A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    b_map_1a = lambda j, k: j*7 + k # B[0,0]=0, B[0,1]=1, B[1,0]=7, B[1,1]=8
    c_map_1a = lambda i, k: k*2 + i # C[0,0]=0, C[1,0]=1, C[0,1]=2, C[1,1]=3
    
    add_strassen(0, 0, 0, a_map_1, b_map_1a, c_map_1a)
    
    # A1 B1b (2x2 * 2x1)
    # B1b: k=2.
    b_map_1b = lambda j: j*7 + 2 # B[0,2]=2, B[1,2]=9
    c_map_1b = lambda i: 2*2 + i # C[0,2]=4, C[1,2]=5
    
    add_2x2x1(0, 0, 0, a_map_1, b_map_1b, c_map_1b)
    
    # A1 B2 (2x2 * 2x3)
    # Split B2 into B2a (2x2, k=3,4) and B2b (2x1, k=5)
    
    # A1 B2a (2x2 * 2x2)
    b_map_2a = lambda j, k: j*7 + (3+k) # B[0,3]=3, B[0,4]=4, B[1,3]=10, B[1,4]=11
    c_map_2a = lambda i, k: (3+k)*2 + i # C[0,3]=6, C[1,3]=7, C[0,4]=8, C[1,4]=9
    
    add_strassen(0, 0, 0, a_map_1, b_map_2a, c_map_2a)
    
    # A1 B2b (2x2 * 2x1)
    b_map_2b = lambda j: j*7 + 5 # B[0,5]=5, B[1,5]=12
    c_map_2b = lambda i: 5*2 + i # C[0,5]=10, C[1,5]=11
    
    add_2x2x1(0, 0, 0, a_map_1, b_map_2b, c_map_2b)
    
    # A1 B3 (2x2 * 2x1)
    # B3: k=6.
    b_map_3 = lambda j: j*7 + 6 # B[0,6]=6, B[1,6]=13
    c_map_3 = lambda i: 6*2 + i # C[0,6]=12, C[1,6]=13
    
    add_2x2x1(0, 0, 0, a_map_1, b_map_3, c_map_3)
    
    # Part 2: A2 (2x1) * B_bottom (1x7)
    # A2 uses A[0,2], A[1,2]
    # B_bottom uses B[2,0..6]
    
    a_map_2 = lambda i: i*3 + 2 # A[0,2]=2, A[1,2]=5
    b_map_2 = lambda k: 2*7 + k # B[2,0]=14, ..., B[2,6]=20
    c_map_2 = lambda i, k: k*2 + i
    
    add_2x1xk(0, 0, 0, a_map_2, b_map_2, c_map_2, 7)
    
    # Total columns used:
    # Strassen 1: 7
    # 2x2x1 1: 4
    # Strassen 2: 7
    # 2x2x1 2: 4
    # 2x2x1 3: 4
    # 2x1x7: 14
    # Total: 7+4+7+4+4+14 = 40.
    
    # Wait, my previous estimate was 34.
    # Where did I lose 6?
    # R(2,2,1) is 2?
    # If R(2,2,1) is 2, then 3 terms of 2x2x1 would be 6.
    # 7+2+7+2+2+14 = 34.
    
    # Let's check R(2,2,1).
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # Can we do this with 2 terms?
    # M1 = (A00 + A10) B0
    # M2 = (A01 + A11) B1
    # C0 = M1_part0 + M2_part0?
    # No, M1 contributes to C0 and C1.
    # M1 = (A00 + A10) B0.
    # Contribution to C0: A00 B0.
    # Contribution to C1: A10 B0.
    # This works!
    # M2 = (A01 + A11) B1.
    # Contribution to C0: A01 B1.
    # Contribution to C1: A11 B1.
    # This works!
    # So R(2,2,1) = 2.
    
    # I need to update add_2x2x1 to use 2 terms.
    
    def add_2x2x1_optimized(a_base, b_base, c_base, a_idx_map, b_idx_map, c_idx_map):
        # Term 1: (A00 + A10) B0 -> C0, C1
        U[a_idx_map[0,0], col] = 1
        U[a_idx_map[1,0], col] = 1
        V[b_idx_map[0], col] = 1
        W[c_idx_map[0], col] = 1
        W[c_idx_map[1], col] = 1
        col += 1
        
        # Term 2: (A01 + A11) B1 -> C0, C1
        U[a_idx_map[0,1], col] = 1
        U[a_idx_map[1,1], col] = 1
        V[b_idx_map[1], col] = 1
        W[c_idx_map[0], col] = 1
        W[c_idx_map[1], col] = 1
        col += 1

    # Re-run with optimized 2x2x1.
    # Reset
    U = np.zeros((n*m, R), dtype=np.int64)
    V = np.zeros((m*p, R), dtype=np.int64)
    W = np.zeros((p*n, R), dtype=np.int64)
    col = 0
    
    add_strassen(0, 0, 0, a_map_1, b_map_1a, c_map_1a)
    add_2x2x1_optimized(0, 0, 0, a_map_1, b_map_1b, c_map_1b)
    add_strassen(0, 0, 0, a_map_1, b_map_2a, c_map_2a)
    add_2x2x1_optimized(0, 0, 0, a_map_1, b_map_2b, c_map_2b)
    add_2x2x1_optimized(0, 0, 0, a_map_1, b_map_3, c_map_3)
    add_2x1xk(0, 0, 0, a_map_2, b_map_2, c_map_2, 7)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
