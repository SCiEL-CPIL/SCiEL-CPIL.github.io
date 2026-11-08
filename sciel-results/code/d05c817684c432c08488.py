"""Matrix multiplication 3x5 by 5x5 rank-minimal decomposition."""
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    # Partition indices into blocks
    # i-blocks: I0={0,1}, I1={2}
    # j-blocks: J0={0,1}, J1={2,3}, J2={4}
    # k-blocks: K0={0,1}, K1={2,3}, K2={4}
    
    i_blocks = [(0, 2), (2, 3)]  # (start, end) for rows of A
    j_blocks = [(0, 2), (2, 4), (4, 5)]  # (start, end) for cols of A / rows of B
    k_blocks = [(0, 2), (2, 4), (4, 5)]  # (start, end) for cols of B
    
    # Collect all (i_block, j_block, k_block) triples
    all_blocks = []
    for ib in i_blocks:
        for jb in j_blocks:
            for kb in k_blocks:
                all_blocks.append((ib, jb, kb))
    
    # For each block, determine if it's a 2x2x2 Strassen block or schoolbook
    # Build U, V, W lists
    U = []  # each element is a dict: key=(i,j) -> value
    V = []  # each element is a dict: key=(j,k) -> value
    W = []  # each element is a dict: key=(k,i) -> value
    
    for (i_start, i_end), (j_start, j_end), (k_start, k_end) in all_blocks:
        i_size = i_end - i_start
        j_size = j_end - j_start
        k_size = k_end - k_start
        
        if i_size == 2 and j_size == 2 and k_size == 2:
            # Strassen 2x2x2
            # Local indices: i_l in {0,1}, j_l in {0,1}, k_l in {0,1}
            # Global: i = i_start + i_l, j = j_start + j_l, k = k_start + k_l
            # Output: W[k*n + i]
            
            # Strassen multipliers:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A10)B11
            # M6 = (A01-A00)(B00+B10)
            # M7 = (A10-A11)(B01+B11)
            
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # For each multiplier, define U, V, W contributions
            # U entry: (i_global, j_global) coefficient
            # V entry: (j_global, k_global) coefficient
            # W entry: (k_global, i_global) coefficient (since W[k*n+i])
            
            strassen_mults = [
                # (U_coeffs, V_coeffs, W_coeffs)
                # M1 = (A00+A11)(B00+B11)
                # U: A00 at (i0,j0), A11 at (i1,j1)
                # V: B00 at (j0,k0), B11 at (j1,k1)
                # W: C00 += M1, C11 += M1
                # C00 is at (i0, k0), C11 is at (i1, k1)
                # W[k*n+i]: for C00: k=k0, i=i0 -> W[k0*n+i0]
                # for C11: k=k1, i=i1 -> W[k1*n+i1]
                {
                    'U': {((i_start+0, j_start+0), 1), ((i_start+1, j_start+1), 1)},
                    'V': {((j_start+0, k_start+0), 1), ((j_start+1, k_start+1), 1)},
                    'W': {((k_start+0, i_start+0), 1), ((k_start+1, i_start+1), 1)}
                },
                # M2 = (A10+A11)B00
                # U: A10 at (i1,j0), A11 at (i1,j1)
                # V: B00 at (j0,k0)
                # W: C10 += M2 -> (i1, k0) -> W[k0*n+i1]
                {
                    'U': {((i_start+1, j_start+0), 1), ((i_start+1, j_start+1), 1)},
                    'V': {((j_start+0, k_start+0), 1)},
                    'W': {((k_start+0, i_start+1), 1)}
                },
                # M3 = A00(B01-B11)
                # U: A00 at (i0,j0)
                # V: B01 at (j0,k1), B11 at (j1,k1) with -1
                # W: C01 += M3 -> (i0, k1) -> W[k1*n+i0]
                {
                    'U': {((i_start+0, j_start+0), 1)},
                    'V': {((j_start+0, k_start+1), 1), ((j_start+1, k_start+1), -1)},
                    'W': {((k_start+1, i_start+0), 1)}
                },
                # M4 = A11(B10-B00)
                # U: A11 at (i1,j1)
                # V: B10 at (j1,k0), B00 at (j0,k0) with -1
                # W: C00 += M4 -> (i0, k0) -> W[k0*n+i0], C10 += M4 -> (i1, k0) -> W[k0*n+i1]
                {
                    'U': {((i_start+1, j_start+1), 1)},
                    'V': {((j_start+1, k_start+0), 1), ((j_start+0, k_start+0), -1)},
                    'W': {((k_start+0, i_start+0), 1), ((k_start+0, i_start+1), 1)}
                },
                # M5 = (A00+A10)B11
                # U: A00 at (i0,j0), A10 at (i1,j0)
                # V: B11 at (j1,k1)
                # W: C00 -= M5 -> (i0, k0) -> W[k0*n+i0] with -1, C01 += M5 -> (i0, k1) -> W[k1*n+i0]
                {
                    'U': {((i_start+0, j_start+0), 1), ((i_start+1, j_start+0), 1)},
                    'V': {((j_start+1, k_start+1), 1)},
                    'W': {((k_start+0, i_start+0), -1), ((k_start+1, i_start+0), 1)}
                },
                # M6 = (A01-A00)(B00+B10)
                # U: A01 at (i0,j1) with 1, A00 at (i0,j0) with -1
                # V: B00 at (j0,k0), B10 at (j1,k0)
                # W: C11 += M6 -> (i1, k1) -> W[k1*n+i1]
                {
                    'U': {((i_start+0, j_start+1), 1), ((i_start+0, j_start+0), -1)},
                    'V': {((j_start+0, k_start+0), 1), ((j_start+1, k_start+0), 1)},
                    'W': {((k_start+1, i_start+1), 1)}
                },
                # M7 = (A10-A11)(B01+B11)
                # U: A10 at (i1,j0) with 1, A11 at (i1,j1) with -1
                # V: B01 at (j0,k1), B11 at (j1,k1)
                # W: C00 += M7 -> (i0, k0) -> W[k0*n+i0]
                {
                    'U': {((i_start+1, j_start+0), 1), ((i_start+1, j_start+1), -1)},
                    'V': {((j_start+0, k_start+1), 1), ((j_start+1, k_start+1), 1)},
                    'W': {((k_start+0, i_start+0), 1)}
                },
            ]
            
            for mult in strassen_mults:
                U.append(mult['U'])
                V.append(mult['V'])
                W.append(mult['W'])
        else:
            # Schoolbook for this block
            for i_l in range(i_size):
                for j_l in range(j_size):
                    for k_l in range(k_size):
                        i_g = i_start + i_l
                        j_g = j_start + j_l
                        k_g = k_start + k_l
                        U.append({(i_g, j_g): 1})
                        V.append({(j_g, k_g): 1})
                        W.append({(k_g, i_g): 1})
    
    rank = len(U)
    
    # Convert to matrix format
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for (i, j), val in U[c].items():
            U_mat[i * m + j][c] = val
        for (j, k), val in V[c].items():
            V_mat[j * p + k][c] = val
        for (k, i), val in W[c].items():
            W_mat[k * n + i][c] = val
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
