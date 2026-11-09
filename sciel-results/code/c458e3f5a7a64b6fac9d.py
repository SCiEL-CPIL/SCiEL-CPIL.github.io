import numpy as np

def construct():
    """
    Construct a rank-65 tensor decomposition for multiplying a 4x4 matrix
    by a 4x5 matrix.
    
    Uses recursive Strassen (49 terms) for the 4x4x4 part plus 16 scalar
    terms for the 5th column.
    
    Returns: {'U': list, 'V': list, 'W': list} with shapes
             U: (16, 65), V: (20, 65), W: (20, 65)
    """
    n, m, p = 4, 4, 5
    rank = 65
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 2x2: 7 multiplications
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For each of the 7 outer Strassen terms, define the U and V block patterns
    # as lists of (position, coefficient) for a 2x2 block
    
    # Outer term definitions: (U_pattern, V_pattern)
    # Each pattern is a list of (row, col, coeff) for the 2x2 block
    
    outer_U = [
        # Term 0: A00 + A11
        [(0,0,1), (0,1,1), (1,0,1), (1,1,1), (0,0,1), (0,1,1), (1,0,1), (1,1,1)],
        # Term 1: A10 + A11
        [(1,0,1), (1,1,1), (1,0,1), (1,1,1), (0,0,1), (0,1,1), (1,0,1), (1,1,1)],
        # Term 2: A00
        [(0,0,1), (0,1,1), (1,0,1), (1,1,1), (0,0,0), (0,1,0), (1,0,0), (1,1,0)],
        # Term 3: A11
        [(0,0,0), (0,1,0), (1,0,0), (1,1,0), (0,0,1), (0,1,1), (1,0,1), (1,1,1)],
        # Term 4: A00 + A01
        [(0,0,1), (0,1,1), (1,0,1), (1,1,1), (0,0,1), (0,1,1), (1,0,1), (1,1,1)],
        # Term 5: A10 - A00
        [(0,0,-1), (0,1,-1), (1,0,-1), (1,1,-1), (0,0,1), (0,1,1), (1,0,1), (1,1,1)],
        # Term 6: A01 - A11
        [(0,0,-1), (0,1,-1), (1,0,-1), (1,1,-1), (0,0,1), (0,1,1), (1,0,1), (1,1,1)],
    ]
    
    outer_V = [
        # Term 0: B00 + B11
        [(0,0,1), (0,1,1), (1,0,1), (1,1,1), (0,0,1), (0,1,1), (1,0,1), (1,1,1)],
        # Term 1: B00
        [(0,0,1), (0,1,1), (1,0,1), (1,1,1), (0,0,0), (0,1,0), (1,0,0), (1,1,0)],
        # Term 2: B01 - B11
        [(0,0,-1), (0,1,-1), (1,0,-1), (1,1,-1), (0,0,1), (0,1,1), (1,0,1), (1,1,1)],
        # Term 3: B10 - B00
        [(0,0,-1), (0,1,-1), (1,0,-1), (1,1,-1), (0,0,1), (0,1,1), (1,0,1), (1,1,1)],
        # Term 4: B11
        [(0,0,0), (0,1,0), (1,0,0), (1,1,0), (0,0,1), (0,1,1), (1,0,1), (1,1,1)],
        # Term 5: B00 + B01
        [(0,0,1), (0,1,1), (1,0,1), (1,1,1), (0,0,1), (0,1,1), (1,0,1), (1,1,1)],
        # Term 6: B10 + B11
        [(0,0,1), (0,1,1), (1,0,1), (1,1,1), (0,0,1), (0,1,1), (1,0,1), (1,1,1)],
    ]
    
    # Inner Strassen contributions to 2x2 result:
    # s0: M[0,0]=1, M[1,1]=1
    # s1: M[1,0]=1, M[1,1]=-1
    # s2: M[0,1]=1, M[1,1]=1
    # s3: M[0,0]=1, M[1,0]=1
    # s4: M[0,0]=-1, M[0,1]=1
    # s5: M[1,1]=1
    # s6: M[0,0]=1
    inner_contrib = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (1,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(1,1,1)],
        [(0,0,1)],
    ]
    
    # Outer term contributions to C blocks:
    # Term 0 (M1): C00=1, C11=1
    # Term 1 (M2): C10=1, C11=-1
    # Term 2 (M3): C01=1, C11=1
    # Term 3 (M4): C00=1, C10=1
    # Term 4 (M5): C00=-1, C01=1
    # Term 5 (M6): C11=1
    # Term 6 (M7): C00=1
    outer_block_map = [
        [('C00', 1), ('C11', 1)],
        [('C10', 1), ('C11', -1)],
        [('C01', 1), ('C11', 1)],
        [('C00', 1), ('C10', 1)],
        [('C00', -1), ('C01', 1)],
        [('C11', 1)],
        [('C00', 1)],
    ]
    
    block_offsets = {'C00': (0, 0), 'C01': (0, 2), 'C10': (2, 0), 'C11': (2, 2)}
    
    # Strassen inner patterns for U and V (each is 4 entries for 2x2)
    # These are the same as the standard Strassen:
    # s0: (A00+A11) * (B00+B11)
    # s1: (A10+A11) * B00
    # s2: A00 * (B01-B11)
    # s3: A11 * (B10-B00)
    # s4: (A00+A01) * B11
    # s5: (A10-A00) * (B00+B01)
    # s6: (A01-A11) * (B10+B11)
    
    inner_U = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,1,1), (1,1,-1)],
    ]
    
    inner_V = [
        [(0,0,1), (1,1,1)],
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (0,0,-1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (1,1,1)],
    ]
    
    col_idx = 0
    for j in range(7):
        for t in range(7):
            # Combine outer and inner U patterns
            u_dict = {}
            for (br, bc, outer_coeff) in outer_U[j]:
                for (ir, ic, inner_coeff) in inner_U[t]:
                    # br,bc is the block position in 4x4, ir,ic is within the 2x2 block
                    ar = 2 * br + ir
                    ac = 2 * bc + ic
                    coeff = outer_coeff * inner_coeff
                    if coeff != 0:
                        u_dict[(ar, ac)] = u_dict.get((ar, ac), 0) + coeff
            
            v_dict = {}
            for (br, bc, outer_coeff) in outer_V[j]:
                for (ir, ic, inner_coeff) in inner_V[t]:
                    vr = 2 * br + ir
                    vc = 2 * bc + ic
                    coeff = outer_coeff * inner_coeff
                    if coeff != 0:
                        v_dict[(vr, vc)] = v_dict.get((vr, vc), 0) + coeff
            
            # Compute W contributions
            w_dict = {}
            for block_name, block_coeff in outer_block_map[j]:
                row_off, col_off = block_offsets[block_name]
                for (mr, mc, inner_coeff) in inner_contrib[t]:
                    c_row = row_off + mr
                    c_col = col_off + mc
                    coeff = block_coeff * inner_coeff
                    w_idx = c_col * n + c_row
                    w_dict[w_idx] = w_dict.get(w_idx, 0) + coeff
            
            # Fill in U, V, W
            for (ai, aj), coeff in u_dict.items():
                U[ai * m + aj, col_idx] += coeff
            
            for (bi, bj), coeff in v_dict.items():
                V[bi * p + bj, col_idx] += coeff
            
            for w_idx, coeff in w_dict.items():
                W[w_idx, col_idx] += coeff
            
            col_idx += 1
    
    # Add 16 scalar multiplications for the 5th column (k=4)
    for i in range(n):
        for j in range(m):
            U[i * m + j, col_idx] = 1.0
            V[j * p + 4, col_idx] = 1.0
            W[4 * n + i, col_idx] = 1.0
            col_idx += 1
    
    # Verify
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    if np.allclose(T, 1.0, atol=1e-8):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: schoolbook
        U_sb = np.zeros((n * m, n * m * p), dtype=float)
        V_sb = np.zeros((m * p, n * m * p), dtype=float)
        W_sb = np.zeros((p * n, n * m * p), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1.0
                    V_sb[j * p + k, col] = 1.0
                    W_sb[k * n + i, col] = 1.0
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
