import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    """
    Rank-84 decomposition of the 3x5x6 matrix multiplication tensor using
    Strassen's rank-7 algorithm for 2x2x2 sub-blocks.
    
    The 3x5 matrix is partitioned into 2x2 blocks (with the third row
    handled separately) and the 5x6 matrix similarly. This gives 6 distinct
    2x2x2 interactions that can use Strassen, saving 1 rank per block.
    
    Schoolbook: 3*5*6 = 90
    Strassen savings: 6 blocks * 1 = 6
    Total rank: 84
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Strassen 2x2x2 tensor decomposition (rank 7)
    # For multiplying 2x2 A by 2x2 B:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)B11
    # M6 = (A01-A10)(B00+B10)
    # M7 = (A11-A01)(B01+B11)
    # C00 = M1 - M2 + M4 - M5
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 - M7
    
    # U indices: (i, j) -> i*2+j for 2x2 A
    # V indices: (j, k) -> j*2+k for 2x2 B  
    # W indices: (k, i) -> k*2+i for 2x2 C
    
    U_222 = np.array([
        # i=0, j=0: A00 appears in M1, M3, M5, M6
        [1, 0, 1, 0, 1, 1, 0],   # M1, M2, M3, M4, M5, M6, M7
        # i=0, j=1: A01 appears in M6, M7
        [0, 0, 0, 0, 0, 1, -1],
        # i=1, j=0: A10 appears in M2, M5, M6
        [0, 1, 0, 0, 1, -1, 0],
        # i=1, j=1: A11 appears in M1, M2, M4, M7
        [1, 1, 0, 1, 0, 0, 1],
    ], dtype=int)
    
    V_222 = np.array([
        # j=0, k=0: B00 appears in M1, M2, M6
        [1, 1, 0, 0, 0, 1, 0],
        # j=0, k=1: B01 appears in M4, M7
        [1, 0, 0, -1, 0, 0, 1],
        # j=1, k=0: B10 appears in M3, M4, M6
        [1, 0, 1, -1, 0, 1, 0],
        # j=1, k=1: B11 appears in M1, M3, M5, M7
        [1, 0, -1, 0, 1, 0, 1],
    ], dtype=int)
    
    W_222 = np.array([
        # k=0, i=0: C00 = M1 - M2 + M4 - M5
        [1, -1, 0, 1, -1, 0, 0],
        # k=0, i=1: C10 = M2 + M4
        [0, 1, 0, 1, 0, 0, 0],
        # k=1, i=0: C01 = M3 + M5
        [0, 0, 1, 0, 1, 0, 0],
        # k=1, i=1: C11 = M1 - M3 + M6 - M7
        [1, 0, -1, 0, 0, 1, -1],
    ], dtype=int)
    
    # Partition indices:
    # A (3x5): rows i=0,1 use Strassen block; row i=2 is separate
    #          cols j=0,1,2,3,4
    # B (5x6): rows j=0,1,2,3,4
    #           cols k=0,1,2,3,4,5
    
    # For the 2x2x2 blocks, we pair:
    # - A rows {0,1} with B cols {0,1}, {2,3}, {4,5} → 3 blocks for each j-pair
    # But we also need to handle A row 2 and the middle column j=2 of B
    
    # Actually, let me think more carefully. The 3x5 matrix has rows i∈{0,1,2} and cols j∈{0,1,2,3,4}.
    # The 5x6 matrix has rows j∈{0,1,2,3,4} and cols k∈{0,1,2,3,4,5}.
    
    # We can partition:
    # A rows: {0,1} (Strassen) and {2} (schoolbook)
    # B cols: {0,1}, {2,3}, {4,5} (each 2-wide for Strassen)
    # B rows: {0,1}, {2,3}, {4} - but row 4 is alone
    
    # So the 2x2x2 blocks are:
    # For each pair (A row pair, B col pair): 2 * 3 = 6 blocks
    # A row pair: always {0,1}
    # B col pairs: {0,1}, {2,3}, {4,5}
    # And we need the corresponding B row indices
    
    # Wait, I need to be more careful. The tensor T[i,j,k] = A[i,j] * B[j,k].
    # For a 2x2x2 block with A rows {i1,i2}, B cols {k1,k2}, and B rows {j1,j2}:
    # We compute C[i1,k1] += A[i1,j1]*B[j1,k1] + A[i1,j2]*B[j2,k1]
    # etc.
    
    # So we need to group: A rows, B rows, B cols all in pairs.
    
    # A rows: {0,1} → 1 pair, and {2} → separate
    # B rows: {0,1}, {2,3}, {4} → 2 pairs and 1 single
    # B cols: {0,1}, {2,3}, {4,5} → 3 pairs
    
    # 2x2x2 blocks (A row pair, B row pair, B col pair):
    # ({0,1}, {0,1}, {0,1}), ({0,1}, {0,1}, {2,3}), ({0,1}, {0,1}, {4,5}),
    # ({0,1}, {2,3}, {0,1}), ({0,1}, {2,3}, {2,3}), ({0,1}, {2,3}, {4,5})
    # That's 6 blocks of 2x2x2, each using rank 7 instead of 8.
    
    # Remaining terms (schoolbook):
    # A row 2 with all B rows and B cols: 1 * 5 * 6 = 30 terms
    # A rows {0,1} with B row 4 (single) and all B cols: 2 * 1 * 6 = 12 terms
    # Total schoolbook: 42 terms
    
    # Total rank: 6*7 + 42 = 42 + 42 = 84
    
    # Let's build the full decomposition
    
    rank_total = 84
    U = np.zeros((n * m, rank_total), dtype=int)  # 15 x 84
    V = np.zeros((m * p, rank_total), dtype=int)  # 30 x 84
    W = np.zeros((p * n, rank_total), dtype=int)  # 18 x 84
    
    col = 0
    
    # Strassen blocks: (A_row_pair, B_row_pair, B_col_pair)
    a_row_pairs = [(0, 1)]
    b_row_pairs = [(0, 1), (2, 3)]
    b_col_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for (i1, i2) in a_row_pairs:
        for (j1, j2) in b_row_pairs:
            for (k1, k2) in b_col_pairs:
                # Add Strassen 2x2x2 block
                for r in range(7):
                    # U: A[i,j] contributions
                    U[i1 * m + j1, col + r] = U_222[0, r]
                    U[i1 * m + j2, col + r] = U_222[1, r]
                    U[i2 * m + j1, col + r] = U_222[2, r]
                    U[i2 * m + j2, col + r] = U_222[3, r]
                    
                    # V: B[j,k] contributions
                    V[j1 * p + k1, col + r] = V_222[0, r]
                    V[j1 * p + k2, col + r] = V_222[1, r]
                    V[j2 * p + k1, col + r] = V_222[2, r]
                    V[j2 * p + k2, col + r] = V_222[3, r]
                    
                    # W: C[k,i] contributions (trace convention: W[k*n + i])
                    W[k1 * n + i1, col + r] = W_222[0, r]
                    W[k1 * n + i2, col + r] = W_222[1, r]
                    W[k2 * n + i1, col + r] = W_222[2, r]
                    W[k2 * n + i2, col + r] = W_222[3, r]
                
                col += 7
    
    # Schoolbook terms for remaining (i,j,k) not covered by Strassen blocks
    covered = set()
    for (i1, i2) in a_row_pairs:
        for (j1, j2) in b_row_pairs:
            for (k1, k2) in b_col_pairs:
                for i in [i1, i2]:
                    for j in [j1, j2]:
                        for k in [k1, k2]:
                            covered.add((i, j, k))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
