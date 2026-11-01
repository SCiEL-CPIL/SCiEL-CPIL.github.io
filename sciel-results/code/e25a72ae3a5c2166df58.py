import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS

    def schoolbook():
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1.0
                    V[j*p + k, idx] = 1.0
                    W[k*n + i, idx] = 1.0
                    idx += 1
        return U, V, W

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        if abs(s - (1.0 if j == jp else 0.0)) > tol:
                            return False
        return True

    start_time = time.time()
    deadline = start_time + 140.0

    # Block decomposition: 6 Strassen 2x2x2 + 2 schoolbook 2x2x1 = rank 50
    rank = 50
    U = np.zeros((n*m, rank), dtype=np.int8)
    V = np.zeros((m*p, rank), dtype=np.int8)
    W = np.zeros((p*n, rank), dtype=np.int8)

    col = 0

    def add_strassen_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)

        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1

        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1

        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1

        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1

        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)

        # Use 3 multiplications instead of 4 for the 2x2x1 block
        # M1 = (a+b)*(e+g) -> gives a*e + a*g + b*e + b*g
        # M2 = (c+d)*(e+g) -> gives c*e + c*g + d*e + d*g  
        # M3 = (a-c)*(e-g) -> gives a*e - a*g - c*e + c*g
        # Then: c00 = M1 - M3/2 (extracts a*e + b*g... wait, this doesn't work cleanly)
        
        # Actually for 2x2x1: we need C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # and C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # This is rank 4 in general (can't do better than schoolbook for 2x2x1)
        
        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1; col += 1
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1; col += 1

    # Try different block partitioning: use Strassen for all 7 columns
    # Split the 4x7 product into: 4x4 (using 2 Strassen blocks) + 4x3 (using 1.5 blocks)
    # Or: try 2x4x7 with different groupings
    
    # Option: Use 3 Strassen 2x2x2 blocks covering columns 0-5, then handle column 6
    # 3 Strassen = 21, + 4 for last column = 25 per 2x4x7? No...
    
    # Let's try: partition the 4 middle dimension differently
    # Use Strassen on (rows 0-1, cols 0-1), (rows 0-1, cols 2-3), (rows 0-1, cols 4-5)
    # Then schoolbook for col 6, and repeat for rows 2-3
    # That's what we have: 6*7 + 2*4 = 42 + 8 = 50
    
    # Alternative: Can we use a 2x4x4 block that's more efficient than 2 Strassen?
    # The 2x4x4 tensor rank is at most 2*7 = 14, but maybe we can do better with structure
    
    # Try: For each pair of rows, decompose 2x4x7 as:
    # - 2x4x4 part: use 2 Strassen (14 mults)  
    # - 2x4x3 part: use schoolbook (12 mults) or find better
    # Total per row-pair: 26, times 2 row-pairs = 52? That's worse.
    
    # Current: 6 Strassen (42) + 2 schoolbook-221 (8) = 50
    # Keep this structure but verify it's correct
    
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)

    assert col == rank, f"Expected rank {rank}, got {col}"

    if verify(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback: try a different decomposition
        # Use 7 Strassen-like blocks with different partitioning
        rank2 = 49
        U2 = np.zeros((n*m, rank2), dtype=np.int8)
        V2 = np.zeros((m*p, rank2), dtype=np.int8)
        W2 = np.zeros((p*n, rank2), dtype=np.int8)
        
        col2 = 0
        # Use Strassen for all 3x2 = 6 pairs of (j,k) blocks
        # Plus handle the 7th column with 1 multiplication per row
        for j_start in [0, 2]:
            for k_start in [0, 2, 4]:
                # Call add_strassen_block but on U2, V2, W2
                i0, j0, k0 = 0, j_start, k_start
                a_idx = i0 * m + j0
                b_idx = i0 * m + (j0 + 1)
                c_idx = (i0 + 1) * m + j0
                d_idx = (i0 + 1) * m + (j0 + 1)
                e_idx = j0 * p + k0
                f_idx = j0 * p + (k0 + 1)
                g_idx = (j0 + 1) * p + k0
                h_idx = (j0 + 1) * p + (k0 + 1)
                c00_idx = k0 * n + i0
                c01_idx = (k0 + 1) * n + i0
                c10_idx = k0 * n + (i0 + 1)
                c11_idx = (k0 + 1) * n + (i0 + 1)
                
                U2[a_idx, col2] += 1; U2[d_idx, col2] += 1
                V2[e_idx, col2] += 1; V2[h_idx, col2] += 1
                W2[c00_idx, col2] += 1; W2[c11_idx, col2] += 1
                col2 += 1

                U2[c_idx, col2] += 1; U2[d_idx, col2] += 1
                V2[e_idx, col2] += 1
                W2[c10_idx, col2] += 1; W2[c11_idx, col2] -= 1
                col2 += 1

                U2[a_idx, col2] += 1
                V2[f_idx, col2] += 1; V2[h_idx, col2] -= 1
                W2[c01_idx, col2] += 1; W2[c11_idx, col2] += 1
                col2 += 1

                U2[d_idx, col2] += 1
                V2[g_idx, col2] += 1; V2[e_idx, col2] -= 1
                W2[c00_idx, col2] += 1; W2[c10_idx, col2] += 1
                col2 += 1

                U2[a_idx, col2] += 1; U2[b_idx, col2] += 1
                V2[h_idx, col2] += 1
                W2[c00_idx, col2] -= 1; W2[c01_idx, col2] += 1
                col2 += 1

                U2[c_idx, col2] += 1; U2[a_idx, col2] -= 1
                V2[e_idx, col2] += 1; V2[f_idx, col2] += 1
                W2[c11_idx, col2] += 1
                col2 += 1

                U2[b_idx, col2] += 1; U2[d_idx, col2] -= 1
                V2[g_idx, col2] += 1; V2[h_idx, col2] += 1
                W2[c00_idx, col2] += 1
                col2 += 1
            
            # Handle column 6 with 2 multiplications (one per row)
            for i in range(n):
                U2[i*m + j_start, col2] += 1
                V2[j_start*p + 6, col2] += 1
                W2[6*n + i, col2] += 1
                col2 += 1
        
        if col2 == rank2 and verify(U2, V2, W2):
            return {
                "U": U2.tolist(),
                "V": V2.tolist(),
                "W": W2.tolist()
            }
        
        U, V, W = schoolbook()
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
