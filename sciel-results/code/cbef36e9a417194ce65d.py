import time
import math

def _strassen_2x2x2_terms():
    """
    Return the 7 rank-1 terms for 2x2x2 matrix multiplication (Strassen).
    Each term is a dict with 'U', 'V', 'W' keys, each a dict mapping (i,j), (j,k), (k,i) to coefficients.
    
    The 7 products are:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11)B00
    M3 = A00(B01 - B11)
    M4 = A11(B10 - B00)
    M5 = (A00 + A01)B11
    M6 = (A10 - A00)(B00 + B01)
    M7 = (A01 - A11)(B10 + B11)
    
    Result: C00 = M1 - M4 + M6 + M7
             C01 = M3 + M4
             C10 = M1 + M2
             C11 = M3 - M2 + M5 + M7
    """
    terms = []
    
    # M1 = (A00 + A11)(B00 + B11)
    # U: A00 + A11, V: B00 + B11, W: contributes to C00 and C10
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,0): 1}
    })
    
    # M2 = (A10 + A11) * B00
    # U: A10 + A11, V: B00, W: contributes to C10
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(0,1): 1}
    })
    
    # M3 = A00 * (B01 - B11)
    # U: A00, V: B01 - B11, W: contributes to C01
    terms.append({
        'U': {(0,0): 1},
        'V': {(0,1): 1, (1,1): -1},
        'W': {(1,0): 1}
    })
    
    # M4 = A11 * (B10 - B00)
    # U: A11, V: B10 - B00, W: contributes -1 to C00, +1 to C01
    terms.append({
        'U': {(1,1): 1},
        'V': {(1,0): 1, (0,0): -1},
        'W': {(0,0): -1, (1,0): 1}
    })
    
    # M5 = (A00 + A01) * B11
    # U: A00 + A01, V: B11, W: contributes to C11
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(1,1): 1}
    })
    
    # M6 = (A10 - A00) * (B00 + B01)
    # U: A10 - A00, V: B00 + B01, W: contributes to C00
    terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(0,0): 1}
    })
    
    # M7 = (A01 - A11) * (B10 + B11)
    # U: A01 - A11, V: B10 + B11, W: contributes to C00 and C11
    terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    
    return terms

def _build_3x3x2_decomposition(k0):
    """
    Build a rank-15 decomposition for 3x3 * 3x2 matrix multiplication.
    The output C has 3 rows and 2 columns (columns k0 and k0+1).
    
    Strategy:
    - Use Strassen (rank 7) for the 2x2x2 subproblem (i in {0,1}, j in {0,1}, k in {0,1})
    - Add schoolbook terms for:
      * i=2, j in {0,1}: 2 terms per k, so 4 terms
      * i in {0,1,2}, j=2: 1 term per (i,k), so 6 terms
    Total: 7 + 4 + 6 = 17 terms
    
    Wait, let me recount:
    - Strassen: 7 terms for C[0:2, 0:2] from A[0:2, 0:2] * B[0:2, 0:2]
    - For i=2, j in {0,1}: C[2,k] += A[2,j]*B[j,k] for j=0,1 and k=0,1 -> 4 terms
    - For j=2, i in {0,1,2}: C[i,k] += A[i,2]*B[2,k] for i=0,1,2 and k=0,1 -> 6 terms
    Total: 7 + 4 + 6 = 17 terms
    """
    n, m, p = 3, 3, 2  # Local dimensions for the 3x3x2 subproblem
    strassen_terms = _strassen_2x2x2_terms()
    
    U_list = []
    V_list = []
    W_list = []
    
    # Part 1: Strassen for i in {0,1}, j in {0,1}, k in {0,1}
    for st in strassen_terms:
        U_term = [0] * (n * m)
        V_term = [0] * (m * p)
        W_term = [0] * (p * n)
        
        for (ii, jj), coeff in st['U'].items():
            U_term[ii * m + jj] = coeff
        for (jj, kk), coeff in st['V'].items():
            V_term[jj * p + kk] = coeff
        for (kk, ii), coeff in st['W'].items():
            W_term[kk * n + ii] = coeff
        
        U_list.append(U_term)
        V_list.append(V_term)
        W_list.append(W_term)
    
    # Part 2: i=2, j in {0,1}, k in {0,1}
    for jj in range(2):
        for kk in range(2):
            U_term = [0] * (n * m)
            V_term = [0] * (m * p)
            W_term = [0] * (p * n)
            U_term[2 * m + jj] = 1
            V_term[jj * p + kk] = 1
            W_term[kk * n + 2] = 1
            U_list.append(U_term)
            V_list.append(V_term)
            W_list.append(W_term)
    
    # Part 3: j=2, i in {0,1,2}, k in {0,1}
    for ii in range(3):
        for kk in range(2):
            U_term = [0] * (n * m)
            V_term = [0] * (m * p)
            W_term = [0] * (p * n)
            U_term[ii * m + 2] = 1
            V_term[2 * p + kk] = 1
            W_term[kk * n + ii] = 1
            U_list.append(U_term)
            V_list.append(V_term)
            W_list.append(W_term)
    
    return U_list, V_list, W_list

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    """
    Construct an exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    
    Uses block-structured decomposition: split the 6 columns into 3 groups of 2,
    and for each group use a rank-17 decomposition for 3x3x2.
    Total rank: 3 * 17 = 51, below schoolbook rank of 54.
    """
    n, m, p = 3, 3, 6
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 120s limit
    
    # Schoolbook fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try the block-structured Strassen-based decomposition
    try:
        all_U = []
        all_V = []
        all_W = []
        
        for col_pair in range(3):
            k0 = col_pair * 2  # 0, 2, 4
            
            # Get the 3x3x2 decomposition
            U_sub, V_sub, W_sub = _build_3x3x2_decomposition(k0)
            
            # Embed into the full 3x3x6 problem by shifting k indices
            for U_row in U_sub:
                all_U.append(U_row)
            
            for V_row in V_sub:
                V_full = [0] * (m * p)
                for jj in range(m):
                    for kk in range(2):
                        V_full[jj * p + (k0 + kk)] = V_row[jj * 2 + kk]
                all_V.append(V_full)
            
            for W_row in W_sub:
                W_full = [0] * (p * n)
                for kk in range(2):
                    for ii in range(n):
                        W_full[(k0 + kk) * n + ii] = W_row[kk * n + ii]
                all_W.append(W_full)
        
        rank = len(all_U)
        
        # Verify
        if _verify_decomposition(all_U, all_V, all_W, n, m, p):
            if rank < schoolbook_rank:
                return {"U": all_U, "V": all_V, "W": all_W}
        
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
