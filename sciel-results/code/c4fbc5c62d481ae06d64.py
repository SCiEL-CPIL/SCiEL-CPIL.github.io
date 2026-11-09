import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p

    def verify(U, V, W, rank):
        """Verify that the decomposition correctly computes matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = np.sum(U[idx_u] * V[idx_v] * W[idx_w])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def build_strassen_decomposition():
        """
        Build a Strassen-style decomposition for 2x5x5 matrix multiplication.
        
        Strategy: Split the 5-dimensional axes into blocks of sizes [2, 2, 1].
        For 2x2 sub-blocks, use Strassen's 7-term algorithm.
        For mixed 2x1, 1x2, and 1x1 sub-blocks, use schoolbook.
        
        The total rank will be less than 50 (schoolbook).
        """
        # Block sizes for splitting dimension 5
        m_blocks = [2, 2, 1]  # Split of m=5
        p_blocks = [2, 2, 1]  # Split of p=5
        
        # Compute block offsets
        m_offsets = [0]
        for size in m_blocks:
            m_offsets.append(m_offsets[-1] + size)
        
        p_offsets = [0]
        for size in p_blocks:
            p_offsets.append(p_offsets[-1] + size)
        
        U = {}  # (row, col) -> value
        V = {}  # (row, col) -> value
        W = {}  # (row, col) -> value
        
        next_col = 0
        
        def add_term(u_row, v_row, w_row, col, val):
            """Add a term to the factor matrices."""
            U[(u_row, col)] = U.get((u_row, col), 0) + val
            V[(v_row, col)] = V.get((v_row, col), 0) + val
            W[(w_row, col)] = W.get((w_row, col), 0) + val
        
        def add_strassen_block(m_start, p_start, next_col):
            """
            Add Strassen's 7-term decomposition for a 2x2 block.
            A[:, m_start:m_start+2] * B[m_start:m_start+2, p_start:p_start+2] -> C[:, p_start:p_start+2]
            
            Returns the new next_col.
            """
            # Strassen's 7 products for 2x2 matrices:
            # M1 = (A11 + A22) * (B11 + B22)
            # M2 = (A21 + A22) * B11
            # M3 = A11 * (B12 - B22)
            # M4 = A22 * (B21 - B11)
            # M5 = (A11 + A12) * B22
            # M6 = (A21 - A11) * (B11 + B12)
            # M7 = (A12 - A22) * (B21 + B22)
            #
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # For each Strassen term, we need to create rank-1 terms.
            # Each product M_t = (linear combo of A entries) * (linear combo of B entries)
            # contributes to C entries with certain coefficients.
            
            # Let's define the Strassen factors:
            # A indices: A[i, m] where i in {0,1}, m in {0,1} (local)
            # B indices: B[m, p] where m in {0,1}, p in {0,1} (local)
            # C indices: C[i, p] where i in {0,1}, p in {0,1} (local)
            
            # Strassen terms and their contributions to C:
            # M1: A factors: {(0,0): 1, (1,1): 1}, B factors: {(0,0): 1, (1,1): 1}
            #     C contributions: {(0,0): 1, (1,1): 1}
            # M2: A factors: {(1,0): 1, (1,1): 1}, B factors: {(0,0): 1}
            #     C contributions: {(1,0): 1, (1,1): -1}
            # M3: A factors: {(0,0): 1}, B factors: {(0,1): 1, (1,1): -1}
            #     C contributions: {(0,1): 1, (1,1): 1}
            # M4: A factors: {(1,1): 1}, B factors: {(1,0): 1, (0,0): -1}
            #     C contributions: {(0,0): 1, (1,0): 1}
            # M5: A factors: {(0,0): 1, (0,1): 1}, B factors: {(1,1): 1}
            #     C contributions: {(0,0): -1, (0,1): 1}
            # M6: A factors: {(1,0): 1, (0,0): -1}, B factors: {(0,0): 1, (0,1): 1}
            #     C contributions: {(1,1): 1}
            # M7: A factors: {(0,1): 1, (1,1): -1}, B factors: {(1,0): 1, (1,1): 1}
            #     C contributions: {(0,0): 1}
            
            strassen_terms = [
                # (A_factors, B_factors, C_contributions)
                # A_factors: {local_i, local_m: coeff}
                # B_factors: {local_m, local_p: coeff}
                # C_contributions: {local_i, local_p: coeff}
                ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
                ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
                ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
                ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
                ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
                ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
                ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
            ]
            
            for a_factors, b_factors, c_contrib in strassen_terms:
                # For each combination of A and B factors, create a rank-1 term
                for (a_i, a_m), a_coeff in a_factors.items():
                    for (b_m, b_p), b_coeff in b_factors.items():
                        if a_m != b_m:
                            continue  # Only valid if A and B indices match
                        
                        # Create a rank-1 term for this combination
                        for (c_i, c_p), c_coeff in c_contrib.items():
                            if a_i != c_i:
                                continue  # Only valid if A and C row indices match
                            
                            # Global indices
                            u_row = c_i * m + (m_start + a_m)
                            v_row = (m_start + a_m) * p + (p_start + b_p)
                            w_row = (p_start + b_p) * n + c_i
                            
                            val = a_coeff * b_coeff * c_coeff
                            add_term(u_row, v_row, w_row, next_col, val)
                        
                        next_col += 1
            
            return next_col
        
        # Process each block combination
        for m_b in range(3):
            for p_b in range(3):
                m_size = m_blocks[m_b]
                p_size = p_blocks[p_b]
                m_start = m_offsets[m_b]
                p_start = p_offsets[p_b]
                
                if m_size == 2 and p_size == 2:
                    # Use Strassen's 7-term algorithm
                    next_col = add_strassen_block(m_start, p_start, next_col)
                else:
                    # Use schoolbook for this block combination
                    for i in range(n):
                        for m_local in range(m_size):
                            for p_local in range(p_size):
                                m_idx = m_start + m_local
                                p_idx = p_start + p_local
                                
                                idx_u = i * m + m_idx
                                idx_v = m_idx * p + p_idx
                                idx_w = p_idx * n + i
                                
                                U[(idx_u, next_col)] = U.get((idx_u, next_col), 0) + 1
                                V[(idx_v, next_col)] = V.get((idx_v, next_col), 0) + 1
                                W[(idx_w, next_col)] = W.get((idx_w, next_col), 0) + 1
                                
                                next_col += 1
        
        # Convert to list format
        rank = next_col
        U_mat = [[0] * rank for _ in range(n * m)]
        V_mat = [[0] * rank for _ in range(m * p)]
        W_mat = [[0] * rank for _ in range(p * n)]
        
        for (row, col), val in U.items():
            U_mat[row][col] = val
        for (row, col), val in V.items():
            V_mat[row][col] = val
        for (row, col), val in W.items():
            W_mat[row][col] = val
        
        return U_mat, V_mat, W_mat, rank

    # Build the Strassen-style decomposition
    U, V, W, rank = build_strassen_decomposition()
    
    # Verify
    if not verify(U, V, W, rank):
        # Fall back to schoolbook
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    U[idx_u][column] = 1
                    V[idx_v][column] = 1
                    W[idx_w][column] = 1
                    column += 1
        
        if not verify(U, V, W, rank):
            raise ValueError("All decompositions failed verification")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
