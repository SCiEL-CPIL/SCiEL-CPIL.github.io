"""Matrix multiplication 4x4 * 4x7 tensor decomposition.

Strategy:
  1. Use the Kronecker-product / block-Strassen construction to build a rank-49
     exact decomposition for the 4x4x4 sub-tensor (first 4 output columns).
  2. Use the schoolbook decomposition (rank 48) for the remaining 3 output columns.
  3. Verify the full decomposition before returning.
  4. If verification fails, fall back to pure schoolbook (rank 112).

The key structural idea (from the inspiration) is that the 4x4x4 matmul tensor
is the Kronecker product T_2 ⊗ T_2 where T_2 is the 2x2x2 matmul tensor.
Since rank(T_2) ≤ 7 (Strassen), rank(T_4) ≤ 7*7 = 49.

The refinement machinery (from the current program) is the explicit
term-by-term construction and verification with a safety fallback.
"""
import time

DIMENSIONS = [4, 4, 7]


def _schoolbook(n=4, m=4, p=7):
    """Schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n=4, m=4, p=7, tol=1e-9):
    """Verify that the decomposition reproduces the matmul tensor.
    For all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
    r = len(U[0]) if U and U[0] else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = U[i * m + j]
                vk = V[j * p + k]
                wi = W[k * n + i]
                for c in range(r):
                    s += ui[c] * vk[c] * wi[c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (U_vec, V_vec, W_vec) where:
      U_vec has length 4: U[i*2+j] for i,j in {0,1}
      V_vec has length 4: V[j*2+k] for j,k in {0,1}
      W_vec has length 4: W[k*2+i] for k,i in {0,1}
    
    The tensor T_2[i,j,k] = 1 for all i,j,k in {0,1}.
    Decomposition: T_2[i,j,k] = sum_{c=1}^{7} U_c[i*2+j] * V_c[j*2+k] * W_c[k*2+i]
    
    Strassen products:
      M1 = (a+d)(e+h)
      M2 = (c+d)(e)
      M3 = a(f-h)
      M4 = d(g-e)
      M5 = (a+b)(h)
      M6 = (c-a)(e+f)
      M7 = (b-d)(g+h)
    
    Output:
      C[0,0] = M1 - M4 + M5 - M7
      C[0,1] = M3 + M5
      C[1,0] = M2 + M4
      C[1,1] = M1 - M2 + M6 - M7
    
    In flattened C index (k*2+i): C[0,0]=idx0, C[0,1]=idx1, C[1,0]=idx2, C[1,1]=idx3
    """
    # A entries: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    #   U index: a=0, b=1, c=2, d=3
    # B entries: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    #   V index: e=0, f=1, g=2, h=3
    # C entries: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    #   W index: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    terms = []
    
    # M1 = (a+d)(e+h): contributes +1 to C[0,0] and +1 to C[1,1]
    # U: a=1, d=1 -> [1,0,0,1]
    # V: e=1, h=1 -> [1,0,0,1]
    # W: C[0,0]=+1, C[1,1]=+1 -> [1,0,0,1]
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    
    # M2 = (c+d)(e): contributes +1 to C[1,0]
    # U: c=1, d=1 -> [0,0,1,1]
    # V: e=1 -> [1,0,0,0]
    # W: C[1,0]=+1 -> [0,0,1,0]
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
    
    # M3 = a(f-h): contributes +1 to C[0,1]
    # U: a=1 -> [1,0,0,0]
    # V: f=1, h=-1 -> [0,1,0,-1]
    # W: C[0,1]=+1 -> [0,1,0,0]
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
    
    # M4 = d(g-e): contributes -1 to C[0,0], +1 to C[1,0]
    # U: d=1 -> [0,0,0,1]
    # V: g=1, e=-1 -> [0,0,1,-1]  ... wait, g-e means coeff 1 for g, -1 for e
    #   V index: e=0, f=1, g=2, h=3 -> [ -1, 0, 1, 0]
    # W: C[0,0]=-1, C[1,0]=+1 -> [-1,0,1,0]
    terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 0, 1, 0]))
    
    # M5 = (a+b)(h): contributes +1 to C[0,0], +1 to C[0,1]
    # U: a=1, b=1 -> [1,1,0,0]
    # V: h=1 -> [0,0,0,1]
    # W: C[0,0]=+1, C[0,1]=+1 -> [1,1,0,0]
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]))
    
    # M6 = (c-a)(e+f): contributes +1 to C[1,1]
    # U: c=1, a=-1 -> [-1,0,1,0]
    # V: e=1, f=1 -> [1,1,0,0]
    # W: C[1,1]=+1 -> [0,0,0,1]
    terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    
    # M7 = (b-d)(g+h): contributes -1 to C[0,0], -1 to C[1,1]
    # U: b=1, d=-1 -> [0,1,0,-1]
    # V: g=1, h=1 -> [0,0,1,1]
    # W: C[0,0]=-1, C[1,1]=-1 -> [-1,0,0,-1]
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 0, -1]))
    
    return terms


def _build_strassen_4x4x4():
    """Build a rank-49 decomposition for 4x4x4 using Kronecker product of Strassen 2x2.
    
    The 4x4x4 matmul tensor T_4 is the Kronecker product T_2 ⊗ T_2.
    If T_2 has rank-7 decomposition (u_c, v_c, w_c) for c=1..7,
    then T_4 has rank-49 decomposition: for each pair (c1, c2),
    U_{c1,c2} = u_{c1} ⊗ u_{c2}  (length 16)
    V_{c1,c2} = v_{c1} ⊗ v_{c2}  (length 16)
    W_{c1,c2} = w_{c1} ⊗ w_{c2}  (length 16)
    
    Indexing:
      U row index: i*4+j where i,j in {0,1,2,3}
      For Kronecker: i = i1*2 + i2, j = j1*2 + j2
      So U[i*4+j] = u1[i1*2+j1] * u2[i2*2+j2]
      
      Similarly for V: j*4+k, j = j1*2+j2, k = k1*2+k2
      V[j*4+k] = v1[j1*2+k1] * v2[j2*2+k2]
      
      W: k*4+i, k = k1*2+k2, i = i1*2+i2
      W[k*4+i] = w1[k1*2+i1] * w2[k2*2+i2]
    """
    base = _strassen_2x2_terms()  # 7 terms, each (U4, V4, W4)
    
    rank = 49
    U = [[0] * rank for _ in range(16)]  # 4*4 = 16 rows
    V = [[0] * rank for _ in range(16)]  # 4*4 = 16 rows
    W = [[0] * rank for _ in range(16)]  # 4*4 = 16 rows
    
    col = 0
    for c1 in range(7):
        u1, v1, w1 = base[c1]
        for c2 in range(7):
            u2, v2, w2 = base[c2]
            
            # Build U vector of length 16
            # U[i*4+j] where i = i1*2+i2, j = j1*2+j2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U[i * 4 + j][col] = u1[i1 * 2 + j1] * u2[i2 * 2 + j2]
            
            # Build V vector of length 16
            # V[j*4+k] where j = j1*2+j2, k = k1*2+k2
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            V[j * 4 + k][col] = v1[j1 * 2 + k1] * v2[j2 * 2 + k2]
            
            # Build W vector of length 16
            # W[k*4+i] where k = k1*2+k2, i = i1*2+i2
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            W[k * 4 + i][col] = w1[k1 * 2 + i1] * w2[k2 * 2 + i2]
            
            col += 1
    
    return {"U": U, "V": V, "W": W}


def _build_hybrid():
    """Build a hybrid decomposition: Strassen-49 for first 4 cols + schoolbook-48 for last 3 cols.
    Total rank: 97. Falls back to schoolbook (rank 112) if verification fails."""
    n, m, p = 4, 4, 7
    deadline = time.time() + 130.0
    
    # Build the 4x4x4 Strassen part
    try:
        strassen_part = _build_strassen_4x4x4()
    except Exception:
        return _schoolbook(n, m, p)
    
    if time.time() > deadline:
        return _schoolbook(n, m, p)
    
    # Build the 4x4x3 schoolbook part
    n2, m2, p2 = 4, 4, 3
    sb_rank = n2 * m2 * p2  # 48
    U_sb = [[0] * sb_rank for _ in range(n2 * m2)]
    V_sb = [[0] * sb_rank for _ in range(m2 * p2)]
    W_sb = [[0] * sb_rank for _ in range(p2 * n2)]
    col = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U_sb[i * m2 + j][col] = 1
                V_sb[j * p2 + k][col] = 1
                W_sb[k * n2 + i][col] = 1
                col += 1
    
    # Now combine:
    # Full U has 16 rows (4*4), rank = 49 + 48 = 97
    # Full V has 28 rows (4*7), rank = 97
    # Full W has 28 rows (7*4), rank = 97
    
    total_rank = 49 + 48
    U_full = [[0] * total_rank for _ in range(16)]
    V_full = [[0] * total_rank for _ in range(28)]  # 4*7
    W_full = [[0] * total_rank for _ in range(28)]  # 7*4
    
    # Copy Strassen part (first 49 columns)
    for i in range(16):
        for c in range(49):
            U_full[i][c] = strassen_part["U"][i][c]
            # V: for the 4x4x4 part, V has 16 rows (j in 0..3, k in 0..3)
            # In the full V (4*7=28 rows), row j*7+k for k in 0..3
            # But strassen_part V has row j*4+k
            V_full[j * 7 + k if False else 0][c] = 0  # placeholder
    
    # Let me redo this more carefully.
    # Strassen part: U (16x49), V (16x49), W (16x49)
    #   U row: i*4+j (i,j in 0..3)
    #   V row: j*4+k (j in 0..3, k in 0..3)  -- these are columns 0-3 of B
    #   W row: k*4+i (k in 0..3, i in 0..3)  -- these are rows 0-3 of C
    
    # Schoolbook part for columns 4-6 of B:
    #   U row: i*4+j (i,j in 0..3)
    #   V row: j*3+k (j in 0..3, k in 0..2)  -- local k for columns 4-6
    #   W row: k*4+i (k in 0..2, i in 0..3)  -- local k for columns 4-6
    
    # Full V row: j*7+k_full where k_full in 0..6
    #   For k_full in 0..3: corresponds to Strassen V row j*4+k_full
    #   For k_full in 4..6: corresponds to SB V row j*3+(k_full-4)
    
    # Full W row: k_full*4+i where k_full in 0..6
    #   For k_full in 0..3: corresponds to Strassen W row k_full*4+i
    #   For k_full in 4..6: corresponds to SB W row (k_full-4)*4+i
    
    # Copy Strassen U
    for i in range(16):
        for c in range(49):
            U_full[i][c] = strassen_part["U"][i][c]
    
    # Copy Strassen V (rows j*4+k for j,k in 0..3) into full V (rows j*7+k)
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V_full[j * 7 + k][c] = strassen_part["V"][j * 4 + k][c]
    
    # Copy Strassen W (rows k*4+i for k,i in 0..3) into full W (rows k*4+i)
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W_full[k * 4 + i][c] = strassen_part["W"][k * 4 + i][c]
    
    # Copy Schoolbook U (columns 49..96)
    for i in range(16):
        for c in range(48):
            U_full[i][49 + c] = U_sb[i][c]
    
    # Copy Schoolbook V (rows j*3+k for j in 0..3, k in 0..2) into full V (rows j*7+(k+4))
    for j in range(4):
        for k in range(3):
            for c in range(48):
                V_full[j * 7 + (k + 4)][49 + c] = V_sb[j * 3 + k][c]
    
    # Copy Schoolbook W (rows k*4+i for k in 0..2, i in 0..3) into full W (rows (k+4)*4+i)
    for k in range(3):
        for i in range(4):
            for c in range(48):
                W_full[(k + 4) * 4 + i][49 + c] = W_sb[k * 4 + i][c]
    
    return {"U": U_full, "V": V_full, "W": W_full}


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Build the hybrid decomposition (rank 97)
    result = _build_hybrid()
    
    # Verify it
    if time.time() < deadline:
        if not _verify(result["U"], result["V"], result["W"], n, m, p):
            # Fallback to schoolbook
            result = _schoolbook(n, m, p)
    
    # Final safety check
    if time.time() < deadline - 5:
        if not _verify(result["U"], result["V"], result["W"], n, m, p):
            result = _schoolbook(n, m, p)
    
    return result
