"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 140.0
    
    # Use Strassen's algorithm recursively:
    # 4x4x4: split into 2x2 blocks of 2x2, use Strassen rank 7 on blocks -> rank 49
    # 4x4x1: rank 16
    # Total: 65
    
    # But let me think about the indexing carefully.
    
    # A is 4x4, B is 4x5, C is 4x5
    # Split A into 4 blocks of 2x2: A11, A12, A21, A22
    # Split B into blocks: B is 4x5, so B11 (2x5), B21 (2x5)? 
    # No, for Strassen we need square blocks.
    
    # Let me split B into B_left (4x4) and B_right (4x1)
    # C_left = A * B_left (4x4), C_right = A * B_right (4x1)
    
    # For C_left = A * B_left, use Strassen recursively.
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B_left = [[B11, B12], [B21, B22]] where each is 2x2
    
    # Strassen:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B21-B22)
    # M4 = A22*(B12-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Each Mi is a 2x2 matrix product, which has rank 4 (schoolbook) or 7 (Strassen).
    # If we use schoolbook for the 2x2 products: total rank = 7*4 = 28 for 4x4x4
    # If we use Strassen for the 2x2 products: total rank = 7*7 = 49 for 4x4x4
    
    # Using schoolbook for inner 2x2: rank 28 for 4x4x4, plus 16 for 4x4x1 = 44
    # Using Strassen for inner 2x2: rank 49 for 4x4x4, plus 16 for 4x4x1 = 65
    
    # So using schoolbook for inner 2x2 is better: rank 44!
    
    # Let me implement rank 44: Strassen outer (7 terms) x schoolbook inner (4 terms each) = 28
    # Plus 16 for the 4x4x1 part.
    
    # Total rank = 28 + 16 = 44
    
    rank_total = 44
    U = [[0] * rank_total for _ in range(16)]  # n*m = 16
    V = [[0] * rank_total for _ in range(20)]  # m*p = 20
    W = [[0] * rank_total for _ in range(20)]  # p*n = 20
    
    col = 0
    
    # Strassen outer terms for 4x4x4
    # A blocks: A11=(0,0),(0,1),(1,0),(1,1); A12=(0,2),(0,3),(1,2),(1,3)
    #           A21=(2,0),(2,1),(3,0),(3,1); A22=(2,2),(2,3),(3,2),(3,3)
    # B blocks: B11=(0,0),(0,1),(1,0),(1,1); B12=(0,2),(0,3),(1,2),(1,3)
    #           B21=(2,0),(2,1),(3,0),(3,1); B22=(2,2),(2,3),(3,2),(3,3)
    # C blocks: C11=(0,0),(0,1),(1,0),(1,1); C12=(0,2),(0,3),(1,2),(1,3)
    #           C21=(2,0),(2,1),(3,0),(3,1); C22=(2,2),(2,3),(3,2),(3,3)
    
    # For each Strassen term, we have a linear combination of A blocks times a linear combination of B blocks
    # Then we do schoolbook multiplication of the resulting 2x2 blocks.
    
    # Let me define the Strassen terms as (A_coeff, B_coeff, C_coeff)
    # where A_coeff is a 4x4 matrix (coefficients for A entries), etc.
    
    # Actually, let me think of it differently. Each Strassen term Mi = Ai * Bi where
    # Ai is a 2x2 matrix (linear combo of A blocks) and Bi is a 2x2 matrix (linear combo of B blocks).
    # The product Mi is a 2x2 matrix, and we add it to certain C blocks with coefficients.
    
    # For the tensor decomposition, each "inner" schoolbook term corresponds to one (i,j,k) triple
    # within a 2x2x2 subproblem.
    
    # Let me define the 7 Strassen terms:
    # M1: A = A11+A22, B = B11+B22, contributes to C11(+), C22(+)
    # M2: A = A21+A22, B = B11, contributes to C21(+), C22(-)
    # M3: A = A11, B = B21-B22, contributes to C12(+), C22(+)
    # M4: A = A22, B = B12-B11, contributes to C11(+), C21(+)
    # M5: A = A11+A12, B = B22, contributes to C11(-), C12(+)
    # M6: A = A21-A11, B = B11+B12, contributes to C22(+)
    # M7: A = A12-A22, B = B21+B22, contributes to C11(+)
    
    # For each Mi, the 2x2x2 schoolbook has 8 terms, but we only need 4 (one per output position).
    # Wait, 2x2x2 schoolbook has rank 8 (not 4). Let me reconsider.
    
    # 2x2x2 schoolbook: C[i,k] = sum_j A[i,j]*B[j,k], 8 terms total.
    # So each Mi (a 2x2x2 product) has rank 8.
    # 7 Strassen terms * 8 inner terms = 56 for 4x4x4. That's worse than 28.
    
    # I made an error. Let me reconsider.
    
    # The 4x4x4 tensor can be decomposed as:
    # For each of the 7 Strassen terms, we have a 2x2x2 sub-tensor.
    # Each 2x2x2 sub-tensor has rank 8 (schoolbook) or 7 (Strassen).
    # Using schoolbook: 7*8 = 56. Using Strassen: 7*7 = 49.
    
    # So the best I can get with recursive Strassen is 49 for 4x4x4.
    # Plus 16 for 4x4x1 = 65 total.
    
    # That's better than 80 but not as good as the theoretical 22+16=38.
    
    # Let me implement rank 65: recursive Strassen for 4x4x4 + schoolbook for 4x4x1.
    
    rank_total = 65
    U = [[0] * rank_total for _ in range(16)]
    V = [[0] * rank_total for _ in range(20)]
    W = [[0] * rank_total for _ in range(20)]
    
    col = 0
    
    # Define 2x2x2 Strassen terms
    # For a 2x2x2 product, Strassen's 7 terms:
    # Each term has (A_entries, B_entries, C_entries)
    # A_entries: dict from (i,j) to coeff
    # B_entries: dict from (j,k) to coeff  
    # C_entries: dict from (k,i) to coeff
    
    strassen_2x2 = [
        # M1: (A11+A22)*(B11+B22) -> C11(+), C22(+)
        {'a': {(0,0):1, (1,1):1}, 'b': {(0,0):1, (1,1):1}, 'c': {(0,0):1, (1,1):1}},
        # M2: (A21+A22)*B11 -> C21(+), C22(-)
        {'a': {(1,0):1, (1,1):1}, 'b': {(0,0):1}, 'c': {(0,1):1, (1,1):-1}},
        # M3: A11*(B21-B22) -> C12(+), C22(+)
        {'a': {(0,0):1}, 'b': {(1,0):1, (1,1):-1}, 'c': {(1,0):1, (1,1):1}},
        # M4: A22*(B12-B11) -> C11(+), C21(+)
        {'a': {(1,1):1}, 'b': {(0,1):1, (0,0):-1}, 'c': {(0,0):1, (0,1):1}},
        # M5: (A11+A12)*B22 -> C11(-), C12(+)
        {'a': {(0,0):1, (0,1):1}, 'b': {(1,1):1}, 'c': {(0,0):-1, (1,0):1}},
        # M6: (A21-A11)*(B11+B12) -> C22(+)
        {'a': {(1,0):1, (0,0):-1}, 'b': {(0,0):1, (0,1):1}, 'c': {(1,1):1}},
        # M7: (A12-A22)*(B21+B22) -> C11(+)
        {'a': {(0,1):1, (1,1):-1}, 'b': {(1,0):1, (1,1):1}, 'c': {(0,0):1}},
    ]
    
    # For 4x4x4, we have 4 quadrants of 2x2 blocks.
    # Block (bi, bj) where bi, bj in {0,1}
    # A block (bi,bj) has entries A[2*bi+i, 2*bj+j] for i,j in {0,1}
    # B block (bj,bk) has entries B[2*bj+j, 2*bk+k] for j,k in {0,1}
    # C block (bi,bk) has entries C[2*bi+i, 2*bk+k] for i,k in {0,1}
    
    # For each pair of outer blocks (bi,bj) for A and (bj,bk) for B,
    # we apply Strassen's 7 terms to the 2x2 sub-problems.
    
    # Actually, the Strassen algorithm for 4x4x4 works as follows:
    # Treat A as 2x2 blocks of 2x2 matrices, same for B and C.
    # Apply Strassen's 7-multiply algorithm at the block level.
    # Each "multiply" is a 2x2x2 matrix multiplication.
    
    # So we have 7 block-level terms, each involving 2x2x2 sub-multiplications.
    # Each 2x2x2 sub-multiplication uses Strassen (rank 7) or schoolbook (rank 8).
    
    # Using Strassen for inner: 7*7 = 49
    # Using schoolbook for inner: 7*8 = 56
    
    # Let me use Strassen for inner: rank 49 for 4x4x4.
    
    # For each of the 7 outer terms, and each of the 7 inner terms, we get one rank-1 term.
    # Total: 49 terms for 4x4x4.
    
    # Plus 16 terms for 4x4x1.
    # Total: 65.
    
    # Let me implement this.
    
    for outer_idx, outer_term in enumerate(strassen_2x2):
        for inner_idx, inner_term in enumerate(strassen_2x2):
            # This term corresponds to outer block multiplication and inner 2x2x2
            # The outer term specifies which block combinations
            # The inner term specifies the 2x2x2 Strassen decomposition
            
            # U entries: for each (i,j) in outer A and (ii,jj) in inner A
            # A global index: (2*outer_i + inner_i, 2*outer_j + inner_j)
            # But outer A is a linear combo of blocks, so:
            # For outer A entry (oi, oj) with coeff ca, and inner A entry (ii, ij) with coeff ci:
            # Global A index: (2*oi + ii, 2*oj + ij), coeff = ca * ci
            
            u_entries = {}
            for (oi, oj), ca in outer_term['a'].items():
                for (ii, ij), ci in inner_term['a'].items():
                    gi = 2*oi + ii
                    gj = 2*oj + ij
                    u_entries[(gi, gj)] = u_entries.get((gi, gj), 0) + ca * ci
            
            v_entries = {}
            for (oj, ok), cb in outer_term['b'].items():
                for (ji, jj), cj in inner_term['b'].items():
                    gj = 2*oj + ji
                    gk = 2*ok + jj
                    v_entries[(gj, gk)] = v_entries.get((gj, gk), 0) + cb * cj
            
            w_entries = {}
            for (ok, oi), cc in outer_term['c'].items():
                for (ki, ki2), ci in inner_term['c'].items():
                    gk = 2*ok + ki
                    gi = 2*oi + ki2
                    w_entries[(gk, gi)] = w_entries.get((gk, gi), 0) + cc * ci
            
            # Add to U, V, W
            for (gi, gj), coeff in u_entries.items():
                if coeff != 0:
                    U[gi*4 + gj][col] += coeff
            for (gj, gk), coeff in v_entries.items():
                if coeff != 0:
                    # gk should be 0..3 for the 4x4x4 part
                    if gk < 4:
                        V[gj*4 + gk][col] += coeff
            for (gk, gi), coeff in w_entries.items():
                if coeff != 0:
                    if gk < 4:
                        W[gk*4 + gi][col] += coeff
            
            col += 1
    
    # Now add 4x4x1 part: rank 16
    # C[i,4] = sum_j A[i,j] * B[j,4]
    for i in range(4):
        for j in range(4):
            U[i*4 + j][col] = 1
            V[j*4 + 4][col] = 1  # B[j,4]
            W[4*4 + i][col] = 1  # C[i,4], W[4*4+i]
            col += 1
    
    assert col == rank_total, f"Expected {rank_total} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
