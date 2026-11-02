import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Define Strassen terms for a 2x2x2 block
    # Local indices: i' in {0,1}, j' in {0,1}, k' in {0,1}
    # U entries: (i', j', coeff)
    # V entries: (j', k', coeff)
    # W entries: (k', i', coeff)
    
    strassen_terms = [
        # M1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),
        # M3
        ([(0,0,1)], [(1,0,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
        # M4
        ([(1,1,1)], [(0,1,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
        # M5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)]),
        # M6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)], [(1,1,1)]),
        # M7
        ([(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)], [(0,0,1),(1,1,-1)]),
    ]
    
    # Define 4 disjoint 2x2x2 blocks
    # i in {0,1}, j in {0,1} or {2,3}, k in {0,1} or {2,3}
    blocks = [
        (0, 0, 0),  # i0=0, j0=0, k0=0
        (0, 2, 0),  # i0=0, j0=2, k0=0
        (0, 0, 2),  # i0=0, j0=0, k0=2
        (0, 2, 2),  # i0=0, j0=2, k0=2
    ]
    
    # Total rank: 4 * 7 (Strassen) + (75 - 4 * 8) (Schoolbook) = 28 + 43 = 71
    rank = 71
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    col = 0
    used_indices = set()
    
    # Add Strassen terms for each block
    for i0, j0, k0 in blocks:
        for u_entries, v_entries, w_entries in strassen_terms:
            for (i_loc, j_loc, c) in u_entries:
                i_global = i0 + i_loc
                j_global = j0 + j_loc
                U[i_global * m + j_global, col] += c
            for (j_loc, k_loc, c) in v_entries:
                j_global = j0 + j_loc
                k_global = k0 + k_loc
                V[j_global * p + k_global, col] += c
            for (k_loc, i_loc, c) in w_entries:
                k_global = k0 + k_loc
                i_global = i0 + i_loc
                W[k_global * n + i_global, col] += c
            col += 1
            used_indices.add((i0, i0+1, j0, j0+1, k0, k0+1)) # Mark as used? No, just track (i,j,k)
            
    # Actually, easier to track used (i,j,k) tuples
    used_ijk = set()
    for i0, j0, k0 in blocks:
        for i_loc in range(2):
            for j_loc in range(2):
                for k_loc in range(2):
                    used_ijk.add((i0+i_loc, j0+j_loc, k0+k_loc))
                    
    # Reset col to 0 and redo properly
    col = 0
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    for i0, j0, k0 in blocks:
        for u_entries, v_entries, w_entries in strassen_terms:
            for (i_loc, j_loc, c) in u_entries:
                i_global = i0 + i_loc
                j_global = j0 + j_loc
                U[i_global * m + j_global, col] += c
            for (j_loc, k_loc, c) in v_entries:
                j_global = j0 + j_loc
                k_global = k0 + k_loc
                V[j_global * p + k_global, col] += c
            for (k_loc, i_loc, c) in w_entries:
                k_global = k0 + k_loc
                i_global = i0 + i_loc
                W[k_global * n + i_global, col] += c
            col += 1
    
    # Add schoolbook terms for remaining (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in used_ijk:
                    continue
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Verify
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if s != 1:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_seed = n * m * p
        U_seed = np.zeros((n * m, rank_seed), dtype=object)
        V_seed = np.zeros((m * p, rank_seed), dtype=object)
        W_seed = np.zeros((p * n, rank_seed), dtype=object)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_seed[i * m + j, col] = 1
                    V_seed[j * p + k, col] = 1
                    W_seed[k * n + i, col] = 1
                    col += 1
        return {"U": U_seed.tolist(), "V": V_seed.tolist(), "W": W_seed.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
