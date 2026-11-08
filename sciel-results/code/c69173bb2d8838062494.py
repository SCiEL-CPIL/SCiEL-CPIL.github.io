import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition block by block.
    # Total rank is the sum of ranks of the blocks.
    
    # Block 1: A1 (2x2) * B11 (2x2) -> C1 (2x2)
    # Uses Strassen (Rank 7)
    # Block 2: A2 (2x3) * B21 (3x2) -> C1 (2x2)
    # Split A2 into A2a (2x2) and A2b (2x1)
    # Split B21 into B21a (2x2) and B21b (1x2)
    # A2a * B21a (2x2x2) -> Strassen (Rank 7)
    # A2b * B21b (2x1x2) -> Schoolbook (Rank 4)
    # Total Block 2 Rank: 11
    # Block 3: A1 (2x2) * B12 (2x3) -> C2 (2x3)
    # Split B12 into B12a (2x2) and B12b (2x1)
    # A1 * B12a (2x2x2) -> Strassen (Rank 7)
    # A1 * B12b (2x2x1) -> Schoolbook (Rank 4)
    # Total Block 3 Rank: 11
    # Block 4: A2 (2x3) * B22 (3x3) -> C2 (2x3)
    # Split A2 into A2a (2x2) and A2b (2x1)
    # Split B22 into B22a (2x3) and B22b (1x3)
    # A2a * B22a (2x2x3) -> 
    #   Split B22a into B22a1 (2x2) and B22a2 (2x1)
    #   A2a * B22a1 (2x2x2) -> Strassen (Rank 7)
    #   A2a * B22a2 (2x2x1) -> Schoolbook (Rank 4)
    #   Total: 11
    # A2b * B22b (2x1x3) -> Schoolbook (Rank 6)
    # Total Block 4 Rank: 17
    
    total_rank = 7 + 11 + 11 + 17
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    def add_term(i, j, k, coeff, rank_idx):
        # i in [0, n), j in [0, m), k in [0, p)
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        U[i * m + j][rank_idx] += coeff
        V[j * p + k][rank_idx] += coeff
        W[k * n + i][rank_idx] += coeff

    def add_strassen_block(i_offset, j_offset, k_offset, rank_start):
        # Adds a 2x2x2 Strassen block
        # i_offset: 0 (since n=2, i is always 0,1)
        # j_offset: start index of j
        # k_offset: start index of k
        # Returns new rank offset
        # Strassen terms:
        # M1: (A00+A11)(B00+B11)
        # M2: (A10+A11)B00
        # M3: A00(B01-B11)
        # M4: A11(B10-B00)
        # M5: (A00+A01)B11
        # M6: (A10-A00)(B00+B01)
        # M7: (A01-A11)(B10+B11)
        
        # We need to map local (li, lj, lk) to global (i, j, k)
        # li in {0,1}, lj in {0,1}, lk in {0,1}
        # i = li (since i_offset is 0)
        # j = j_offset + lj
        # k = k_offset + lk
        
        # M1
        add_term(0, j_offset + 0, k_offset + 0, 1, rank_start + 0)
        add_term(1, j_offset + 1, k_offset + 0, 1, rank_start + 0)
        add_term(0, j_offset + 0, k_offset + 0, 1, rank_start + 0) # Wait, U, V, W indices
        # U[i*m+j], V[j*p+k], W[k*n+i]
        # For M1:
        # U terms: (0,0), (1,1)
        # V terms: (0,0), (0,1), (1,0), (1,1) ? No, B00 is (0,0), B11 is (1,1)
        # B00: j=0, k=0. B11: j=1, k=1.
        # So V terms: (0,0), (1,1)
        # W terms: (0,0), (1,1)
        
        # Let's list the (i, j, k) tuples for each M
        # M1: (A00+A11)(B00+B11)
        # Pairs: (0,0) and (1,1) for A, (0,0) and (1,1) for B.
        # Tuples: (0,0,0), (0,0,1)? No.
        # A00: i=0, j=0. B00: j=0, k=0. Term: (0,0,0)
        # A00: i=0, j=0. B11: j=1, k=1. Term: (0,1,1)
        # A11: i=1, j=1. B00: j=0, k=0. Term: (1,0,0)
        # A11: i=1, j=1. B11: j=1, k=1. Term: (1,1,1)
        # Coeff 1 for all.
        
        tuples_M1 = [(0,0,0), (0,1,1), (1,0,0), (1,1,1)]
        for t in tuples_M1:
            add_term(t[0], j_offset + t[1], k_offset + t[2], 1, rank_start + 0)

        # M2: (A10+A11)B00
        # A10: i=1, j=0. A11: i=1, j=1. B00: j=0, k=0.
        # Tuples: (1,0,0), (1,1,0)
        tuples_M2 = [(1,0,0), (1,1,0)]
        for t in tuples_M2:
            add_term(t[0], j_offset + t[1], k_offset + t[2], 1, rank_start + 1)

        # M3: A00(B01-B11)
        # A00: i=0, j=0. B01: j=0, k=1. B11: j=1, k=1.
        # Tuples: (0,0,1) coeff 1, (0,1,1) coeff -1
        add_term(0, j_offset + 0, k_offset + 1, 1, rank_start + 2)
        add_term(0, j_offset + 1, k_offset + 1, -1, rank_start + 2)

        # M4: A11(B10-B00)
        # A11: i=1, j=1. B10: j=1, k=0. B00: j=0, k=0.
        # Tuples: (1,1,0) coeff 1, (1,0,0) coeff -1
        add_term(1, j_offset + 1, k_offset + 0, 1, rank_start + 3)
        add_term(1, j_offset + 0, k_offset + 0, -1, rank_start + 3)

        # M5: (A00+A01)B11
        # A00: i=0, j=0. A01: i=0, j=1. B11: j=1, k=1.
        # Tuples: (0,0,1), (0,1,1)
        tuples_M5 = [(0,0,1), (0,1,1)]
        for t in tuples_M5:
            add_term(t[0], j_offset + t[1], k_offset + t[2], 1, rank_start + 4)

        # M6: (A10-A00)(B00+B01)
        # A10: i=1, j=0. A00: i=0, j=0. B00: j=0, k=0. B01: j=0, k=1.
        # Tuples:
        # A10*B00: (1,0,0) coeff 1
        # A10*B01: (1,0,1) coeff 1
        # A00*B00: (0,0,0) coeff -1
        # A00*B01: (0,0,1) coeff -1
        add_term(1, j_offset + 0, k_offset + 0, 1, rank_start + 5)
        add_term(1, j_offset + 0, k_offset + 1, 1, rank_start + 5)
        add_term(0, j_offset + 0, k_offset + 0, -1, rank_start + 5)
        add_term(0, j_offset + 0, k_offset + 1, -1, rank_start + 5)

        # M7: (A01-A11)(B10+B11)
        # A01: i=0, j=1. A11: i=1, j=1. B10: j=1, k=0. B11: j=1, k=1.
        # Tuples:
        # A01*B10: (0,1,0) coeff 1
        # A01*B11: (0,1,1) coeff 1
        # A11*B10: (1,1,0) coeff -1
        # A11*B11: (1,1,1) coeff -1
        add_term(0, j_offset + 1, k_offset + 0, 1, rank_start + 6)
        add_term(0, j_offset + 1, k_offset + 1, 1, rank_start + 6)
        add_term(1, j_offset + 1, k_offset + 0, -1, rank_start + 6)
        add_term(1, j_offset + 1, k_offset + 1, -1, rank_start + 6)
        
        return rank_start + 7

    def add_schoolbook_block(i_offset, j_offset, k_offset, n_j, n_k, rank_start):
        # Adds a 2 x n_j x n_k schoolbook block
        # i is always 0,1
        for lj in range(n_j):
            for lk in range(n_k):
                for li in range(2):
                    add_term(li, j_offset + lj, k_offset + lk, 1, rank_start + (lj * n_k + lk))
        return rank_start + 2 * n_j * n_k

    # Block 1: A1 (j 0-1) * B11 (j 0-1, k 0-1)
    current_rank = add_strassen_block(0, 0, 0, current_rank)
    
    # Block 2: A2 (j 2-4) * B21 (j 2-4, k 0-1)
    # Split A2 into A2a (j 2-3) and A2b (j 4)
    # Split B21 into B21a (j 2-3, k 0-1) and B21b (j 4, k 0-1)
    # A2a * B21a: 2x2x2 Strassen
    current_rank = add_strassen_block(0, 2, 0, current_rank)
    # A2b * B21b: 2x1x2 Schoolbook
    current_rank = add_schoolbook_block(0, 4, 0, 1, 2, current_rank)
    
    # Block 3: A1 (j 0-1) * B12 (j 0-1, k 2-4)
    # Split B12 into B12a (j 0-1, k 2-3) and B12b (j 0-1, k 4)
    # A1 * B12a: 2x2x2 Strassen
    current_rank = add_strassen_block(0, 0, 2, current_rank)
    # A1 * B12b: 2x2x1 Schoolbook
    current_rank = add_schoolbook_block(0, 0, 4, 2, 1, current_rank)
    
    # Block 4: A2 (j 2-4) * B22 (j 2-4, k 2-4)
    # Split A2 into A2a (j 2-3) and A2b (j 4)
    # Split B22 into B22a (j 2-3, k 2-4) and B22b (j 4, k 2-4)
    # A2a * B22a: 2x2x3
    #   Split B22a into B22a1 (j 2-3, k 2-3) and B22a2 (j 2-3, k 4)
    #   A2a * B22a1: 2x2x2 Strassen
    current_rank = add_strassen_block(0, 2, 2, current_rank)
    #   A2a * B22a2: 2x2x1 Schoolbook
    current_rank = add_schoolbook_block(0, 2, 4, 2, 1, current_rank)
    # A2b * B22b: 2x1x3 Schoolbook
    current_rank = add_schoolbook_block(0, 4, 2, 1, 3, current_rank)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
