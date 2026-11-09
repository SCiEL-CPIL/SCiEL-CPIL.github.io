"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    # Hybrid Strassen/Schoolbook decomposition.
    # Split n=3 into 2+1, m=4 into 2+2, p=7 into 2+2+2+1.
    # For each block (n_i, m_j, p_k), use Strassen if n_i=m_j=p_k=2, else Schoolbook.
    n, m, p = DIMENSIONS
    
    # Splitting indices
    n_splits = [2, 1]
    m_splits = [2, 2]
    p_splits = [2, 2, 2, 1]
    
    # Accumulators for U, V, W
    U_rows = []
    V_rows = []
    W_rows = []
    
    # We build column by column.
    # Total columns = sum of ranks of all blocks.
    
    # Helper to add a schoolbook block
    def add_schoolbook(n_i, m_i, p_i, u_offset, v_offset, w_offset):
        # U part: n_i * m_i rows, p_i columns (each col is a basis vector for one k)
        # Actually, U shape is (n_i*m_i, rank_block). Here rank_block = n_i*m_i*p_i.
        # Wait, in the final concatenation, U is (n*m, total_rank).
        # Each block contributes n_i*m_i rows to the respective part of U.
        # But U is already partitioned by (i,j).
        # It is easier to build the full U, V, W by iterating blocks and appending columns.
        
        rank_block = n_i * m_i * p_i
        for k_idx in range(p_i):
            for i_idx in range(n_i):
                for j_idx in range(m_i):
                    # This is getting complicated to manage offsets if we just append columns.
                    # Let's pre-compute the total rank and then fill arrays.
                    pass
        return rank_block

    # Let's calculate total rank first to allocate arrays
    total_rank = 0
    block_info = []
    for i_idx, n_i in enumerate(n_splits):
        for j_idx, m_j in enumerate(m_splits):
            for k_idx, p_k in enumerate(p_splits):
                if n_i == 2 and m_j == 2 and p_k == 2:
                    r = 7
                else:
                    r = n_i * m_j * p_k
                total_rank += r
                block_info.append((i_idx, j_idx, k_idx, n_i, m_j, p_k, r))
    
    # Allocate U, V, W as lists of lists (zeros)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    current_col = 0
    
    for (i_idx, j_idx, k_idx, n_i, m_j, p_k, r) in block_info:
        # Determine row offsets in the full matrices
        # i_idx is the index of the n-split. n_off = sum(n_splits[:i_idx])
        # j_idx is the index of the m-split. m_off = sum(m_splits[:j_idx])
        # k_idx is the index of the p-split. p_off = sum(p_splits[:k_idx])
        
        n_off = sum(n_splits[:i_idx])
        m_off = sum(m_splits[:j_idx])
        p_off = sum(p_splits[:k_idx])
        
        if n_i == 2 and m_j == 2 and p_k == 2:
            # Strassen block
            # Map local indices to global
            # Local i: 0,1 -> global i: n_off, n_off+1
            # Local j: 0,1 -> global j: m_off, m_off+1
            # Local k: 0,1 -> global k: p_off, p_off+1
            
            # Strassen multipliers M1..M7
            # U entries: for each multiplier, which (i,j) contributes what coefficient?
            # M1 = (A11 + A22)(B11 + B22)
            # M2 = (A21 + A22)B11
            # M3 = A11(B12 - B22)
            # M4 = A22(B21 - B11)
            # M5 = (A11 + A12)B22
            # M6 = (A21 - A11)(B11 + B12)
            # M7 = (A12 - A22)(B21 + B22)
            
            # U factors (coefficients for A_ij)
            # M1: A11(+1), A22(+1)
            # M2: A21(+1), A22(+1)
            # M3: A11(+1)
            # M4: A22(+1)
            # M5: A11(+1), A12(+1)
            # M6: A21(+1), A11(-1)
            # M7: A12(+1), A22(-1)
            
            U_factors = [
                {0: 1, 3: 1},       # M1: (0,0), (1,1)
                {2: 1, 3: 1},       # M2: (1,0), (1,1)
                {0: 1},             # M3: (0,0)
                {3: 1},             # M4: (1,1)
                {0: 1, 1: 1},       # M5: (0,0), (0,1)
                {2: 1, 0: -1},      # M6: (1,0), (0,0)
                {1: 1, 3: -1}       # M7: (0,1), (1,1)
            ]
            
            # V factors (coefficients for B_jk)
            # M1: B11(+1), B22(+1)
            # M2: B11(+1)
            # M3: B12(+1), B22(-1)
            # M4: B21(+1), B11(-1)
            # M5: B22(+1)
            # M6: B11(+1), B12(+1)
            # M7: B21(+1), B22(+1)
            
            V_factors = [
                {0: 1, 3: 1},       # M1: (0,0), (1,1)
                {0: 1},             # M2: (0,0)
                {1: 1, 3: -1},      # M3: (0,1), (1,1)
                {2: 1, 0: -1},      # M4: (1,0), (0,0)
                {3: 1},             # M5: (1,1)
                {0: 1, 1: 1},       # M6: (0,0), (0,1)
                {2: 1, 3: 1}        # M7: (1,0), (1,1)
            ]
            
            # W factors (coefficients for C_ik)
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # W[row, col] where row = k*n + i.
            # Local k: 0,1. Local i: 0,1.
            # C00 (i=0,k=0): M1(+1), M4(+1), M5(-1), M7(+1)
            # C01 (i=0,k=1): M3(+1), M5(+1)
            # C10 (i=1,k=0): M2(+1), M4(+1)
            # C11 (i=1,k=1): M1(+1), M2(-1), M3(+1), M6(+1)
            
            W_factors = [
                {0: 1, 3: 1, 4: -1, 6: 1},  # C00
                {2: 1, 4: 1},               # C01
                {1: 1, 3: 1},               # C10
                {0: 1, 1: -1, 2: 1, 5: 1}   # C11
            ]
            
            for m_idx in range(7):
                col = current_col + m_idx
                
                # Fill U
                for (local_i, local_j), coeff in U_factors[m_idx].items():
                    global_i = n_off + local_i
                    global_j = m_off + local_j
                    U[global_i * m + global_j][col] = coeff
                
                # Fill V
                for (local_j, local_k), coeff in V_factors[m_idx].items():
                    global_j = m_off + local_j
                    global_k = p_off + local_k
                    V[global_j * p + global_k][col] = coeff
                
                # Fill W
                for (local_i, local_k), coeff in W_factors[m_idx].items():
                    global_i = n_off + local_i
                    global_k = p_off + local_k
                    W[global_k * n + global_i][col] = coeff
            
            current_col += 7

        else:
            # Schoolbook block
            for k_l in range(p_k):
                for i_l in range(n_i):
                    for j_l in range(m_j):
                        col = current_col
                        global_i = n_off + i_l
                        global_j = m_off + j_l
                        global_k = p_off + k_l
                        
                        U[global_i * m + global_j][col] = 1
                        V[global_j * p + global_k][col] = 1
                        W[global_k * n + global_i][col] = 1
                        
                        current_col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
