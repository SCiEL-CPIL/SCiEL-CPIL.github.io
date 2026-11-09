import time

def construct():
    start_time = time.time()
    
    # Helper to generate Strassen terms for n x n
    def strassen_terms(n):
        if n == 1:
            return [([1], [1], [1])]
        
        h = n // 2
        terms_h = strassen_terms(h)
        new_terms = []
        
        # Define the 7 Strassen steps
        # X_def: list of (block_row, block_col, coeff)
        # Y_def: list of (block_row, block_col, coeff)
        # C_contrib: list of (block_row, block_col, coeff)
        
        steps = [
            # M1 = (A00+A11)(B00+B11)
            ([((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))], [(0,0,1), (1,1,1)]),
            # M2 = (A10+A11)B00
            ([((1,0,1), (1,1,1))], [((0,0,1))], [(1,0,1)]),
            # M3 = A00(B01-B11)
            ([((0,0,1))], [((0,1,1), (1,1,-1))], [(0,1,1)]),
            # M4 = A11(B10-B00)
            ([((1,1,1))], [((1,0,1), (0,0,-1))], [(1,0,1)]),
            # M5 = (A00+A01)B11
            ([((0,0,1), (0,1,1))], [((1,1,1))], [(0,0,-1), (0,1,1)]),
            # M6 = (A10-A00)(B00+B01)
            ([((1,0,1), (0,0,-1))], [((0,0,1), (0,1,1))], [(1,1,1)]),
            # M7 = (A01-A11)(B10+B11)
            ([((0,1,1), (1,1,-1))], [((1,0,1), (1,1,1))], [(0,0,1)]),
        ]
        
        for X_def, Y_def, C_contrib in steps:
            for u_h, v_h, w_h in terms_h:
                u = [0] * (n*n)
                v = [0] * (n*n)
                w = [0] * (n*n)
                
                # Fill u
                for br, bc, coeff in X_def:
                    for i in range(h):
                        for j in range(h):
                            idx_h = i * h + j
                            idx_global = (br * h + i) * n + (bc * h + j)
                            u[idx_global] += coeff * u_h[idx_h]
                
                # Fill v
                for br, bc, coeff in Y_def:
                    for i in range(h):
                        for j in range(h):
                            idx_h = i * h + j
                            idx_global = (br * h + i) * n + (bc * h + j)
                            v[idx_global] += coeff * v_h[idx_h]
                
                # Fill w
                for br, bc, coeff in C_contrib:
                    for i in range(h):
                        for j in range(h):
                            idx_h = i * h + j
                            idx_global = (br * h + i) * n + (bc * h + j)
                            w[idx_global] += coeff * w_h[idx_h]
                
                # Check if term is non-zero
                if any(x != 0 for x in u) and any(x != 0 for x in v) and any(x != 0 for x in w):
                    new_terms.append((u, v, w))
        
        return new_terms

    # Get 49 terms for 4x4x4
    strassen_49 = strassen_terms(4)
    
    # Dimensions
    n, m, p = 4, 4, 5
    rank_strassen = len(strassen_49) # 49
    rank_scalar = n * (p - 4) # 4 * 1 = 16
    total_rank = rank_strassen + rank_scalar # 65
    
    # Initialize matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill Strassen terms (first 49)
    for r in range(rank_strassen):
        u, v, w = strassen_49[r]
        # u, v, w are length 16 (4x4)
        # Map to global indices
        for idx in range(16):
            U[idx][r] = u[idx]
            V[idx][r] = v[idx] # v corresponds to B[0:4, 0:4]
            W[idx][r] = w[idx] # w corresponds to C[0:4, 0:4]
            
        # Note: V and W have extra columns/rows for the 5th part
        # V has size 4x5=20. Indices 16-19 correspond to B[:, 4]
        # W has size 5x4=20. Indices 16-19 correspond to C[:, 4]
        # These are already 0, which is correct for the Strassen part.

    # Fill Scalar terms (next 16)
    # Multiplies A (4x4) with B_right (4x1) to get C_right (4x1)
    # Term for A[i,j] * B[j,4] -> C[i,4]
    col_offset = 4 # 5th column index is 4
    for i in range(n):
        for j in range(m):
            r = rank_strassen + i * m + j
            # U index: i*m + j
            U[i * m + j][r] = 1
            # V index: j*p + 4
            V[j * p + col_offset][r] = 1
            # W index: 4*n + i
            W[col_offset * n + i][r] = 1

    return {"U": U, "V": V, "W": W}
