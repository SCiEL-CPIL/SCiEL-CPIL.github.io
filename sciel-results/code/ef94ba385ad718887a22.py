"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    """Block decomposition with Strassen on 2x2x2 sub-blocks for 3x5x7."""
    start_time = time.time()
    deadline = start_time + 100  # safety margin
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Split dimensions to maximize 2x2x2 blocks
    row_splits = [2, 1]
    mid_splits = [2, 2, 1]
    col_splits = [2, 2, 1, 2]
    
    row_starts = [0]
    for s in row_splits:
        row_starts.append(row_starts[-1] + s)
    mid_starts = [0]
    for s in mid_splits:
        mid_starts.append(mid_starts[-1] + s)
    col_starts = [0]
    for s in col_splits:
        col_starts.append(col_starts[-1] + s)
    
    # Collect all rank-1 terms: (U_row, V_row, W_row, coeff)
    terms = []
    
    def add_strassen_2x2x2(r_start, m_start, p_start):
        """Add 7 Strassen terms for a 2x2x2 block."""
        i0, i1 = r_start, r_start + 1
        j0, j1 = m_start, m_start + 1
        k0, k1 = p_start, p_start + 1
        
        # U indices: (i, j) -> i*m + j
        # V indices: (j, k) -> j*p + k
        # W indices: (k, i) -> k*n + i
        
        # Term 1 (M1): A-part (1,0,0,1), B-part (1,0,0,1), W-part (1,0,0,1)
        # U: (i0,j0), (i1,j1)
        # V: (j0,k0), (j1,k1)
        # W: (k0,i0), (k1,i1)
        terms.append((i0 * m + j0, j0 * p + k0, k0 * n + i0, 1))
        terms.append((i1 * m + j1, j1 * p + k1, k1 * n + i1, 1))
        # Wait, this adds 2 terms, not 1.
        # The rank-1 term is a single column.
        # We need to add ONE column for M1.
        # U_col has 1s at (i0,j0) and (i1,j1).
        # V_col has 1s at (j0,k0) and (j1,k1).
        # W_col has 1s at (k0,i0) and (k1,i1).
        # This is not how the terms list works. The terms list adds (u_row, v_row, w_row, coeff).
        # This implies we are building the matrix column by column.
        # If I add (u1, v1, w1, 1) and (u2, v2, w2, 1), it creates two separate columns.
        # I need a way to add a single column with multiple non-zero entries.
        
        # Let's change the structure. Instead of terms, I'll collect columns.
        # Each column is a dict of U, V, W entries.
        pass
    
    # Re-implement using a list of columns
    columns = []
    
    def add_strassen_2x2x2(r_start, m_start, p_start):
        i0, i1 = r_start, r_start + 1
        j0, j1 = m_start, m_start + 1
        k0, k1 = p_start, p_start + 1
        
        # M1
        u = {i0*m+j0: 1, i1*m+j1: 1}
        v = {j0*p+k0: 1, j1*p+k1: 1}
        w = {k0*n+i0: 1, k1*n+i1: 1}
        columns.append((u, v, w))
        
        # M2
        u = {i1*m+j0: 1, i1*m+j1: 1}
        v = {j0*p+k0: 1}
        w = {k0*n+i1: 1, k1*n+i1: -1}
        columns.append((u, v, w))
        
        # M3
        u = {i0*m+j0: 1}
        v = {j1*p+k0: 1, j1*p+k1: -1}
        w = {k0*n+i0: 0, k1*n+i0: 1, k0*n+i1: 0, k1*n+i1: 1} # Wait, W indices are (k,i).
        # W part for M3: C01 (+1), C11 (+1).
        # C01 is i=0, k=1. W[1,0] += 1.
        # C11 is i=1, k=1. W[1,1] += 1.
        # So W = {k1*n+i0: 1, k1*n+i1: 1}
        w = {k1*n+i0: 1, k1*n+i1: 1}
        columns.append((u, v, w))
        
        # M4
        u = {i1*m+j1: 1}
        v = {j0*p+k1: 1, j0*p+k0: -1}
        w = {k0*n+i0: 1, k0*n+i1: 1} # C00 (+1), C10 (+1).
        # C00: i=0, k=0. W[0,0] += 1.
        # C10: i=1, k=0. W[0,1] += 1.
        columns.append((u, v, w))
        
        # M5
        u = {i0*m+j0: 1, i0*m+j1: 1}
        v = {j1*p+k1: 1}
        w = {k0*n+i0: -1, k1*n+i0: 1} # C00 (-1), C01 (+1).
        columns.append((u, v, w))
        
        # M6
        u = {i1*m+j0: 1, i0*m+j0: -1}
        v = {j0*p+k0: 1, j0*p+k1: 1}
        w = {k1*n+i1: 1} # C11 (+1).
        columns.append((u, v, w))
        
        # M7
        u = {i0*m+j1: 1, i1*m+j1: -1}
        v = {j1*p+k0: 1, j1*p+k1: 1}
        w = {k0*n+i0: 1} # C00 (+1).
        columns.append((u, v, w))
        
    def add_schoolbook(r_start, m_start, p_start, r_size, m_size, p_size):
        for i in range(r_start, r_start + r_size):
            for j in range(m_start, m_start + m_size):
                for k in range(p_start, p_start + p_size):
                    u = {i*m+j: 1}
                    v = {j*p+k: 1}
                    w = {k*n+i: 1}
                    columns.append((u, v, w))
    
    for ri, (r_start, r_size) in enumerate(zip(row_starts[:-1], row_splits)):
        for mi, (m_start, m_size) in enumerate(zip(mid_starts[:-1], mid_splits)):
            for pi, (p_start, p_size) in enumerate(zip(col_starts[:-1], col_splits)):
                if r_size == 2 and m_size == 2 and p_size == 2:
                    add_strassen_2x2x2(r_start, m_start, p_start)
                else:
                    add_schoolbook(r_start, m_start, p_start, r_size, m_size, p_size)
    
    # Combine columns: group by (U_row, V_row, W_row)
    from collections import defaultdict
    combined = defaultdict(int)
    for u_dict, v_dict, w_dict in columns:
        # This approach is flawed because a single column has multiple entries in U, V, W.
        # We cannot just sum coefficients for (u_row, v_row, w_row) if they come from different columns.
        # We need to keep the column structure.
        # However, if two columns are identical, we can merge them.
        # But Strassen columns are distinct.
        
        # Let's just create the matrices directly from the columns list.
        pass
    
    # Create U, V, W
    rank = len(columns)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col_idx, (u_dict, v_dict, w_dict) in enumerate(columns):
        for row, val in u_dict.items():
            U[row][col_idx] = val
        for row, val in v_dict.items():
            V[row][col_idx] = val
        for row, val in w_dict.items():
            W[row][col_idx] = val
    
    # Verify
    if time.time() < deadline:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err > 1e-9:
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
