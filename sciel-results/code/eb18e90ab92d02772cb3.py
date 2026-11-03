import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 3, 5, 7
    
    # Laderman 3x3x3 core
    # The 7 tensor terms for multiplying 3x3 by 3x3:
    # M1 = (a11+a22+a33)(b11+b22+b33)
    # M2 = (a12+a23+a31)(b12+b23+b31)
    # M3 = (a13+a21+a32)(b13+b21+b32)
    # M4 = a11(b22-b33)
    # M5 = a22(b33-b11)
    # M6 = a33(b11-b22)
    # M7 = (a11+a22-a33)(b11+b22-b33) - (M1+M4+M5+M6)
    # Actually, the standard Laderman decomposition uses 7 multiplications.
    # Let me use the explicit coefficients for the 7 terms.
    
    # The Laderman factors for 3x3 matrix multiplication:
    # U matrices (3x3 each, 7 total):
    # U1 = [[1,1,1],[1,1,1],[1,1,1]]
    # U2 = [[0,1,1],[1,0,1],[1,1,0]]
    # U3 = [[1,0,1],[1,1,0],[0,1,1]]
    # U4 = [[1,0,0],[0,0,0],[0,0,0]]
    # U5 = [[0,0,0],[0,1,0],[0,0,0]]
    # U6 = [[0,0,0],[0,0,0],[0,0,1]]
    # U7 = [[1,1,-1],[1,1,-1],[1,1,-1]]  -- wait, this isn't right.
    
    # Let me use the correct Laderman decomposition:
    # M1 = (a11+a22+a33)(b11+b22+b33)
    # M2 = (a12+a23+a31)(b12+b23+b31)  
    # M3 = (a13+a21+a32)(b13+b21+b32)
    # M4 = a11(b22-b33)
    # M5 = a22(b33-b11)
    # M6 = a33(b11-b22)
    # c11 = M1 - M5 - M6 + M7 where M7 = (a11+a22-a33)(b11+b22-b33)
    
    # Actually let me just use the known 7-term Laderman:
    # U factors (for A matrix, 3x3):
    U_lad = [
        [[1,1,1],[1,1,1],[1,1,1]],  # M1
        [[0,1,1],[1,0,1],[1,1,0]],  # M2
        [[1,0,1],[1,1,0],[0,1,1]],  # M3
        [[1,0,0],[0,0,0],[0,0,0]],  # M4
        [[0,0,0],[0,1,0],[0,0,0]],  # M5
        [[0,0,0],[0,0,0],[0,0,1]],  # M6
        [[1,1,-1],[1,1,-1],[1,1,-1]],  # M7
    ]
    V_lad = [
        [[1,1,1],[1,1,1],[1,1,1]],  # M1
        [[0,1,1],[1,0,1],[1,1,0]],  # M2
        [[1,0,1],[1,1,0],[0,1,1]],  # M3
        [[0,1,-1],[0,0,0],[0,0,0]],  # M4
        [[0,0,0],[0,0,1],[0,-1,0]],  # M5
        [[0,-1,0],[0,0,0],[0,1,0]],  # M6
        [[1,1,-1],[1,1,-1],[1,1,-1]],  # M7
    ]
    W_lad = [
        [[1,1,1],[1,1,1],[1,1,1]],  # M1
        [[0,1,1],[1,0,1],[1,1,0]],  # M2
        [[1,0,1],[1,1,0],[0,1,1]],  # M3
        [[1,0,0],[0,0,0],[0,0,0]],  # M4
        [[0,0,0],[0,1,0],[0,0,0]],  # M5
        [[0,0,0],[0,0,0],[0,0,1]],  # M6
        [[1,1,-1],[1,1,-1],[1,1,-1]],  # M7
    ]
    
    # This doesn't look right either. Let me use a verified Laderman decomposition.
    # Standard Laderman 3x3x3:
    # The 7 products are:
    # M1 = (a11+a22+a33)(b11+b22+b33)
    # M2 = (a12+a23+a31)(b12+b23+b31)
    # M3 = (a13+a21+a32)(b13+b21+b32)
    # M4 = a11(b22-b33)
    # M5 = a22(b33-b11)  
    # M6 = a33(b11-b22)
    # M7 = (a11+a22-a33)(b11+b22-b33)
    
    # Then:
    # c11 = M1 - M5 - M6 + M7
    # c12 = M2 - (M4 - M5) ... no this is getting complicated.
    
    # Let me just use the schoolbook approach but with Strassen 2x2x2 blocks where possible.
    # For 3x5x7: we can fit at most 1 disjoint 2x2x2 block (as analyzed).
    # So: 1 Strassen block (7 terms) + 105 - 8 = 97 schoolbook terms = 104 total.
    # Or just pure schoolbook: 105.
    # Or try to find better combinations.
    
    # Actually, let's try a different approach: use multiple 2x2x2 blocks that overlap
    # in some dimensions but not others, and handle the overlaps carefully.
    
    # Simplest correct approach: pure schoolbook with rank 105.
    # But we can do better with at least one 2x2x2 Strassen block.
    
    # Use Strassen for the first 2x2x2 subblock (i=0,1; j=0,1; k=0,1)
    # Then schoolbook for all other entries.
    
    U_cols = []
    V_cols = []
    W_cols = []
    covered = set()
    
    # Strassen 2x2x2 block for i in {0,1}, j in {0,1}, k in {0,1}
    i0, i1 = 0, 1
    j0, j1 = 0, 1
    k0, k1 = 0, 1
    
    for i in (i0, i1):
        for j in (j0, j1):
            for k in (k0, k1):
                covered.add((i, j, k))
    
    a00 = i0 * m + j0
    a01 = i0 * m + j1
    a10 = i1 * m + j0
    a11 = i1 * m + j1
    
    b00 = j0 * p + k0
    b01 = j0 * p + k1
    b10 = j1 * p + k0
    b11 = j1 * p + k1
    
    c00 = k0 * n + i0
    c01 = k1 * n + i0
    c10 = k0 * n + i1
    c11 = k1 * n + i1
    
    # Strassen 7 terms
    U_cols.append({a00: 1, a11: 1})
    V_cols.append({b00: 1, b11: 1})
    W_cols.append({c00: 1, c11: 1})
    
    U_cols.append({a10: 1, a11: 1})
    V_cols.append({b00: 1})
    W_cols.append({c10: 1, c11: -1})
    
    U_cols.append({a00: 1})
    V_cols.append({b01: 1, b11: -1})
    W_cols.append({c01: 1, c11: 1})
    
    U_cols.append({a11: 1})
    V_cols.append({b10: 1, b00: -1})
    W_cols.append({c00: 1, c10: 1})
    
    U_cols.append({a00: 1, a01: 1})
    V_cols.append({b11: 1})
    W_cols.append({c00: -1, c01: 1})
    
    U_cols.append({a10: 1, a00: -1})
    V_cols.append({b00: 1, b01: 1})
    W_cols.append({c11: 1})
    
    U_cols.append({a01: 1, a11: -1})
    V_cols.append({b10: 1, b11: 1})
    W_cols.append({c00: 1})
    
    # Schoolbook for all other entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                U_cols.append({i * m + j: 1})
                V_cols.append({j * p + k: 1})
                W_cols.append({k * n + i: 1})
    
    rank = len(U_cols)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for c in range(rank):
        for idx, val in U_cols[c].items():
            U[idx, c] = val
        for idx, val in V_cols[c].items():
            V[idx, c] = val
        for idx, val in W_cols[c].items():
            W[idx, c] = val
    
    # Verify
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.sum((T_comp - T) ** 2)
    
    if error > 1e-9:
        # Fallback to pure schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
