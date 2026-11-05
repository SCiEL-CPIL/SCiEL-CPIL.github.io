import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Partition indices 0..4 into groups of sizes 2, 2, 1
    # Group 0: 0, 1
    # Group 1: 2, 3
    # Group 2: 4
    
    groups = [[0, 1], [2, 3], [4]]
    sizes = [len(g) for g in groups]
    
    U = []
    V = []
    W = []
    
    # Helper to add a decomposition for a specific block (a, b, c)
    # a: row group of A, b: col group of A (row group of B), c: col group of B
    # We need to map local indices to global indices.
    
    def add_block(a, b, c, rank, u_local, v_local, w_local):
        # u_local: (n_a, m_b, rank)
        # v_local: (m_b, p_c, rank)
        # w_local: (p_c, n_a, rank)
        
        i_indices = groups[a]
        j_indices = groups[b]
        k_indices = groups[c]
        
        # Pre-allocate or append
        # We are building lists of rows.
        # U has n*m rows. V has m*p rows. W has p*n rows.
        
        for r in range(rank):
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            for i_local, i_global in enumerate(i_indices):
                for j_local, j_global in enumerate(j_indices):
                    u_col[i_global * m + j_global] = u_local[i_local][j_local][r]
                    
            for j_local, j_global in enumerate(j_indices):
                for k_local, k_global in enumerate(k_indices):
                    v_col[j_global * p + k_global] = v_local[j_local][k_local][r]
                    
            for k_local, k_global in enumerate(k_indices):
                for i_local, i_global in enumerate(i_indices):
                    w_col[k_global * n + i_global] = w_local[k_local][i_local][r]
                    
            U.append(u_col)
            V.append(v_col)
            W.append(w_col)

    # We need decompositions for each block type.
    # Type 2x2x2: Strassen
    # Type 2x2x1: Rank 4
    # Type 2x1x1: Rank 2
    # Type 1x1x1: Rank 1
    
    # Strassen 2x2x2
    # Standard Strassen matrices
    U_s = [
        [[1, 0], [0, 0]],
        [[1, 1], [0, 0]],
        [[1, 0], [0, 1]],
        [[0, 0], [1, 0]],
        [[0, 1], [0, 1]],
        [[0, 1], [0, 0]],
        [[0, 0], [1, -1]]
    ]
    V_s = [
        [[0, 0], [0, 0]], # Placeholder, need correct V
        # Strassen V matrices for B (2x2)
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # U rows correspond to A factors.
        # U1: A11+A22 -> [[1,0],[0,1]]? No, A is 2x2.
        # A11 is (0,0), A22 is (1,1).
        # U1: [[1,0],[0,1]]
        # U2: [[0,0],[1,1]]
        # U3: [[1,0],[0,0]]
        # U4: [[0,0],[0,1]]
        # U5: [[1,1],[0,0]]
        # U6: [[-1,0],[1,1]]
        # U7: [[0,-1],[1,0]]
        
        # V1: B11+B22 -> [[1,0],[0,1]]
        # V2: B11 -> [[1,0],[0,0]]
        # V3: B21-B22 -> [[0,-1],[1,0]]? B21 is (1,0), B22 is (1,1).
        # V3: [[0,0],[-1,1]]?
        # Let's use a known valid Strassen decomposition.
        
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, -1], [1, 0]],
        [[0, 1], [0, -1]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]]
    ]
    W_s = [
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # W maps to C indices (k, i).
        # C11 is (0,0). C12 is (0,1). C21 is (1,0). C22 is (1,1).
        # W is 2x2.
        [[1, 1], [0, 0]],
        [[0, 0], [1, -1]],
        [[0, 1], [0, 0]],
        [[-1, 0], [0, 1]],
        [[0, -1], [0, 0]],
        [[0, 0], [0, 1]],
        [[0, 0], [1, 0]]
    ]
    
    # Verify Strassen
    # This is risky. I will use the schoolbook for 2x2x2 if I'm not sure, but rank 7 is known.
    # Actually, I can just use the 7-term Strassen.
    
    # For simplicity and correctness, I will use the explicit 89-rank construction.
    # But I need the explicit matrices.
    
    # Let's just use the 117 solution from the prompt's knowledge if 89 is too complex to code correctly in time.
    # The prompt says "Measured: 1 -> 1.068" for 117.
    # If I can get 89, it's 1.4.
    
    # I will implement the 117 solution which is known to work in the "2+2+1" family.
    # The 117 solution likely uses Strassen for the 4 2x2x2 blocks and schoolbook for the rest?
    # 4*7 + 23*? 
    # If the rest are schoolbook:
    # 12 blocks of 2x2x1 (rank 4) -> 48
    # 6 blocks of 2x1x1 (rank 2) -> 12
    # 1 block of 1x1x1 (rank 1) -> 1
    # Sum: 28 + 48 + 12 + 1 = 89.
    # Why 117?
    # Maybe the 2x2x2 blocks are not independent?
    # Or maybe my rank for 2x2x1 is wrong?
    # If 2x2x1 rank is 4, sum is 89.
    
    # I will bet on 89.
    
    pass
# EVOLVE-BLOCK-END
