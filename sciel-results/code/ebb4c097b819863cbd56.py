import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 49 for M_{4,4,4} (Strassen)
    rank_1 = 49
    # Rank 48 for M_{4,4,3} (Schoolbook)
    rank_2 = 48
    total_rank = rank_1 + rank_2
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # --- Part 1: M_{4,4,4} ---
    # Indices for A: i in 0..3, j in 0..3
    # Indices for B: j in 0..3, k in 0..3
    # Indices for C: k in 0..3, i in 0..3
    
    # Strassen 4x4
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each M_i is a 2x2x2 multiplication.
    # We need the Strassen decomposition of 2x2x2.
    
    # Helper to add a 2x2x2 Strassen decomposition
    # a_block: 2x2 indices in A (i1, i2, j1, j2)
    # b_block: 2x2 indices in B (j1, j2, k1, k2)
    # c_block: 2x2 indices in C (k1, k2, i1, i2)
    # offset: column offset in U, V, W
    
    def add_strassen_2x2(a_i, a_j, b_j, b_k, c_k, c_i, offset):
        # a_i, a_j, b_j, b_k, c_k, c_i are tuples of 2 indices
        # e.g., a_i = (0, 1)
        
        # Strassen 2x2 terms
        # S1 = (A11+A22)(B11+B22)
        # S2 = (A21+A22)B11
        # S3 = A11(B12-B22)
        # S4 = A22(B21-B11)
        # S5 = (A11+A12)B22
        # S6 = (A21-A11)(B11+B12)
        # S7 = (A12-A22)(B21+B22)
        
        # U, V, W for each S_k
        # U[i, j] for A
        # V[j, k] for B
        # W[k, i] for C
        
        # S1
        # U: A11+A22 -> (0,0)=1, (1,1)=1
        # V: B11+B22 -> (0,0)=1, (1,1)=1
        # W: C11+=1, C12+=0, C21+=0, C22+=1? No.
        # C11 = S1 + S4 - S5 + S7
        # C12 = S3 + S5
        # C21 = S2 + S4
        # C22 = S1 - S2 + S3 + S6
        
        # We need to map the 2x2 indices to the global indices.
        # A index: a_i[0], a_i[1], a_j[0], a_j[1]
        # B index: b_j[0], b_j[1], b_k[0], b_k[1]
        # C index: c_k[0], c_k[1], c_i[0], c_i[1]
        
        # S1
        U[a_i[0]*m + a_j[0], offset] += 1
        U[a_i[1]*m + a_j[1], offset] += 1
        V[b_j[0]*p + b_k[0], offset] += 1
        V[b_j[1]*p + b_k[1], offset] += 1
        W[c_k[0]*n + c_i[0], offset] += 1
        W[c_k[1]*n + c_i[1], offset] += 1
        
        # S2
        U[a_i[1]*m + a_j[0], offset+1] += 1
        U[a_i[1]*m + a_j[1], offset+1] += 1
        V[b_j[0]*p + b_k[0], offset+1] += 1
        W[c_k[0]*n + c_i[1], offset+1] += 1
        
        # S3
        U[a_i[0]*m + a_j[0], offset+2] += 1
        V[b_j[0]*p + b_k[1], offset+2] += 1
        V[b_j[1]*p + b_k[1], offset+2] -= 1
        W[c_k[1]*n + c_i[0], offset+2] += 1
        
        # S4
        U[a_i[1]*m + a_j[1], offset+3] += 1
        V[b_j[0]*p + b_k[0], offset+3] -= 1
        V[b_j[1]*p + b_k[0], offset+3] += 1
        W[c_k[0]*n + c_i[1], offset+3] += 1
        
        # S5
        U[a_i[0]*m + a_j[0], offset+4] += 1
        U[a_i[0]*m + a_j[1], offset+4] += 1
        V[b_j[1]*p + b_k[1], offset+4] += 1
        W[c_k[1]*n + c_i[0], offset+4] -= 1
        W[c_k[1]*n + c_i[1], offset+4] += 1 # Wait, C22 has +S5? No.
        # C12 = S3 + S5. C12 is c_k[1], c_i[0].
        # C22 = S1 - S2 + S3 + S6.
        # So S5 only contributes to C12.
        W[c_k[1]*n + c_i[0], offset+4] += 1
        
        # S6
        U[a_i[1]*m + a_j[0], offset+5] += 1
        U[a_i[0]*m + a_j[0], offset+5] -= 1
        V[b_j[0]*p + b_k[0], offset+5] += 1
        V[b_j[0]*p + b_k[1], offset+5] += 1
        W[c_k[1]*n + c_i[1], offset+5] += 1
        
        # S7
        U[a_i[0]*m + a_j[1], offset+6] += 1
        U[a_i[1]*m + a_j[1], offset+6] -= 1
        V[b_j[0]*p + b_k[0], offset+6] += 1
        V[b_j[1]*p + b_k[1], offset+6] += 1
        W[c_k[0]*n + c_i[0], offset+6] += 1
        W[c_k[1]*n + c_i[1], offset+6] += 1 # Wait, C11 has +S7. C11 is c_k[0], c_i[0].
        # C11 = S1 + S4 - S5 + S7.
        # So S7 contributes to C11.
        W[c_k[0]*n + c_i[0], offset+6] += 1

    # Define blocks for 4x4
    # A11: i=0,1, j=0,1
    # A12: i=0,1, j=2,3
    # A21: i=2,3, j=0,1
    # A22: i=2,3, j=2,3
    
    # B11: j=0,1, k=0,1
    # B12: j=0,1, k=2,3
    # B21: j=2,3, k=0,1
    # B22: j=2,3, k=2,3
    
    # C11: k=0,1, i=0,1
    # C12: k=0,1, i=2,3
    # C21: k=2,3, i=0,1
    # C22: k=2,3, i=2,3
    
    # M1 = (A11+A22)(B11+B22)
    # This is not a single 2x2x2 multiplication.
    # (A11+A22) is a 2x2 matrix.
    # (B11+B22) is a 2x2 matrix.
    # So M1 is a 2x2x2 multiplication.
    
    add_strassen_2x2((0,1), (0,1), (0,1), (0,1), (0,1), (0,1), 0) # M1? No.
    # M1 = (A11+A22)(B11+B22)
    # A part: A11+A22. Indices: (0,1) for i, (0,1) for j.
    # B part: B11+B22. Indices: (0,1) for j, (0,1) for k.
    # C part: C11 += M1. Indices: (0,1) for k, (0,1) for i.
    # But M1 is a 2x2 matrix.
    # The result of M1 is added to C11.
    # So the C block for M1 is C11.
    
    # Wait, my add_strassen_2x2 function assumes the A and B blocks are contiguous 2x2 blocks.
    # For M1, A is A11+A22. This is not a contiguous block in memory, but the indices are (0,1) and (2,3).
    # My function `add_strassen_2x2` takes `a_i` and `a_j` as tuples.
    # For A11+A22:
    # A11 is i=0,1, j=0,1.
    # A22 is i=2,3, j=2,3.
    # So `a_i` should be (0,2)? No.
    # The linear combination is $L_A = A_{11} + A_{22}$.
    # $L_A[i,j] = A[i,j] + A[i+2, j+2]$.
    # This is not a simple permutation of indices.
    # My `add_strassen_2x2` function sets U entries to 1 or -1.
    # It can handle this if I pass the correct indices.
    # For $L_A = A_{11} + A_{22}$:
    # $U[i,j]$ for $i \in \{0,1\}, j \in \{0,1\}$ is 1.
    # $U[i,j]$ for $i \in \{2,3\}, j \in \{2,3\}$ is 1.
    # But my function only sets 4 entries in U (for the 2x2 block).
    # It doesn't support summing two disjoint blocks.
    
    # I need a more flexible function or to expand the logic.
    # Let's create a helper to add a term $L_A \cdot L_B$ to the decomposition.
    # $L_A$ is a 2x2 matrix defined by a list of (i, j, coeff).
    # $L_B$ is a 2x2 matrix defined by a list of (j, k, coeff).
    # The product is added to a 2x2 block of C.
    
    def add_term(a_terms, b_terms, c_terms, offset):
        # a_terms: list of (i, j, coeff)
        # b_terms: list of (j, k, coeff)
        # c_terms: list of (k, i, coeff)
        
        # This adds ONE rank-1 term? No.
        # $L_A \cdot L_B$ is a 2x2 matrix product.
        # It has rank 4 (schoolbook) or 7 (Strassen).
        # I want to use Strassen for the inner 2x2.
        # So I should call `add_strassen_2x2` with the appropriate indices?
        # No, `add_strassen_2x2` assumes the blocks are contiguous.
        
        # Let's just use schoolbook for the inner 2x2?
        # Rank 4.
        # Total rank 7 * 4 = 28.
        # This is better than 49!
        # Wait, is rank 28 valid for M_{4,4,4}?
        # If I use schoolbook for the inner 2x2, I am effectively using a non-recursive Strassen.
        # Is it valid? Yes.
        # Is it minimal? No, 49 is better? No, 28 < 49.
        # Why did I think Strassen 4x4 is 49?
        # Because Strassen is recursive.
        # But I can use a "hybrid" approach.
        # Use Strassen for the outer 4x4 (7 terms), and Schoolbook for the inner 2x2 (4 terms each).
        # Total rank 28.
        
        # Let's verify if 28 is valid.
        # The decomposition must be exact.
        # Schoolbook 2x2 is exact.
        # So 28 is a valid rank for M_{4,4,4}.
        
        # Let's implement `add_term_schoolbook` for 2x2x2.
        
        # U, V, W for 2x2 schoolbook
        # Terms: (0,0), (0,1), (1,0), (1,1)
        
        # For each term (u, v) in 2x2:
        # U[i, j] = delta(i,u) delta(j,v) * coeff_A
        # V[j, k] = delta(j,v) delta(k,w) * coeff_B
        # W[k, i] = delta(k,w) delta(i,x) * coeff_C
        
        # This is getting complicated.
        # Let's stick to the known rank 49 for M_{4,4,4} using full Strassen.
        # It's safer.
        
        pass

    # Let's just use the schoolbook decomposition for the whole 4x4x7.
    # Rank 112.
    # It's simple and correct.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
