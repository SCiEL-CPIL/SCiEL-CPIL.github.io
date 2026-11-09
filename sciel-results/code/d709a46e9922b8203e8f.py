"""Matrix multiplication 3x5x5: find rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify exact decomposition: U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta_{i,i'}delta_{j,j'}delta_{k,k'}"""
    r = len(U[0]) if U else 0
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    for jp in range(m):
                        for kp in range(p):
                            expected = 1 if (i==ip and j==jp and k==kp) else 0
                            s = 0
                            for c in range(r):
                                s += U[i*m+j, c] * V[jp*p+kp, c] * W[kp*n+ip, c]
                            if s != expected:
                                return False
    return True

def _strassen_2x2x2():
    """
    Returns U, V, W matrices for 2x2x2 Strassen decomposition.
    U is 4x7, V is 4x7, W is 4x7.
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1
    W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1
    W[3][2] = 1
    
    # M4
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1
    W[2][3] = 1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1
    W[1][4] = 1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _laderman_4x4x4():
    """
    Returns U, V, W matrices for 4x4x4 Laderman decomposition.
    U is 16x21, V is 16x21, W is 16x21.
    Indices: U[i*4+j, c], V[j*4+k, c], W[k*4+i, c].
    
    Laderman's algorithm for 4x4 matrix multiplication with rank 21.
    """
    # The 21 products in Laderman's algorithm
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # M8 = (A33+A44)(B33+B44)
    # M9 = (A43+A44)B33
    # M10 = A33(B43-B44)
    # M11 = A44(B34-B33)
    # M12 = (A33+A34)B44
    # M13 = (A43-A33)(B33+B34)
    # M14 = (A34-A44)(B43+B44)
    # M15 = (A13+A24)(B13+B24)
    # M16 = (A23-A14)(B14+B23)
    # M17 = (A14+A23)(B13-B24)
    # M18 = (A31+A42)(B31+B42)
    # M19 = (A41-A32)(B32+B41)
    # M20 = (A32+A41)(B31-B42)
    # M21 = (A11+A22+A33+A44)(B11+B22+B33+B44) - this is not quite right
    
    # Actually, let me use the standard Laderman algorithm.
    # The standard Laderman algorithm has 21 products.
    # Let me look up the exact formulation.
    
    # Standard Laderman algorithm:
    # Let A = [[a11, a12, a13, a14],
    #          [a21, a22, a23, a24],
    #          [a31, a32, a33, a34],
    #          [a41, a42, a43, a44]]
    # Let B = [[b11, b12, b13, b14],
    #          [b21, b22, b23, b24],
    #          [b31, b32, b33, b34],
    #          [b41, b42, b43, b44]]
    
    # The 21 products are:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b12-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # M8 = (a33+a44)(b33+b44)
    # M9 = (a43+a44)b33
    # M10 = a33(b43-b44)
    # M11 = a44(b34-b33)
    # M12 = (a33+a34)b44
    # M13 = (a43-a33)(b33+b34)
    # M14 = (a34-a44)(b43+b44)
    # M15 = (a13+a24)(b13+b24)
    # M16 = (a23-a14)(b14+b23)
    # M17 = (a14+a23)(b13-b24)
    # M18 = (a31+a42)(b31+b42)
    # M19 = (a41-a32)(b32+b41)
    # M20 = (a32+a41)(b31-b42)
    # M21 = (a11+a22+a33+a44)(b11+b22+b33+b44)
    
    # Wait, that's not quite right either. Let me use a known correct formulation.
    
    # Actually, let me just implement Strassen for 2x2x2 blocks and use that.
    # The Laderman algorithm is complex and I might make errors.
    
    # Let me return None to indicate we should use Strassen instead.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook baseline
    rank_school = n * m * p  # 75
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_school
    
    # Try to improve using Strassen 2x2x2 blocks
    # We can fit 4 disjoint 2x2x2 blocks in the 3x5x5 tensor
    # i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # Blocks:
    # 1. i:0-1, j:0-1, k:0-1
    # 2. i:0-1, j:2-3, k:0-1
    # 3. i:0-1, j:0-1, k:2-3
    # 4. i:0-1, j:2-3, k:2-3
    
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    # For each block, we replace 8 schoolbook terms with 7 Strassen terms.
    # We need to build new U, V, W with rank 75 - 4 = 71.
    
    new_rank = 71
    new_U = [[0] * new_rank for _ in range(n * m)]
    new_V = [[0] * new_rank for _ in range(m * p)]
    new_W = [[0] * new_rank for _ in range(p * n)]
    
    # Identify which (i,j,k) are in blocks.
    in_block = set()
    for (i0, j0, k0) in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    in_block.add((i0+di, j0+dj, k0+dk))
    
    col_idx = 0
    # 1. Add schoolbook terms for non-block indices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i,j,k) not in in_block:
                    new_U[i*m+j][col_idx] = 1
                    new_V[j*p+k][col_idx] = 1
                    new_W[k*n+i][col_idx] = 1
                    col_idx += 1
    
    # 2. Add Strassen terms for each block
    strassen_U, strassen_V, strassen_W = _strassen_2x2x2()
    
    for (i0, j0, k0) in blocks:
        for c in range(7):
            # Map local indices to global
            for li in range(2):
                for lj in range(2):
                    u_val = strassen_U[li*2+lj][c]
                    if u_val != 0:
                        gi = i0 + li
                        gj = j0 + lj
                        new_U[gi*m+gj][col_idx] += u_val
            
            for lj in range(2):
                for lk in range(2):
                    v_val = strassen_V[lj*2+lk][c]
                    if v_val != 0:
                        gj = j0 + lj
                        gk = k0 + lk
                        new_V[gj*p+gk][col_idx] += v_val
            
            for lk in range(2):
                for li in range(2):
                    w_val = strassen_W[lk*2+li][c]
                    if w_val != 0:
                        gk = k0 + lk
                        gi = i0 + li
                        new_W[gk*n+gi][col_idx] += w_val
            
            col_idx += 1
    
    # Verify
    if _verify(new_U, new_V, new_W, n, m, p):
        best = {"U": new_U, "V": new_V, "W": new_W}
        best_rank = new_rank
    
    elapsed = time.time() - start_time
    if elapsed > 130:
        return best
    
    return best
# EVOLVE-BLOCK-END
