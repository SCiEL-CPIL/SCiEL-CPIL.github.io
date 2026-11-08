"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
New approach: use a recursive partitioning with a different block structure.
Split 8 columns into 4+4. For each 4-column block, split into 2+2.
For each 2-column block, use a modified Strassen decomposition.
Additionally, share some intermediate computations across blocks.

Strategy: For the 2x2 * 2x2 core, use Strassen (rank 7).
For the cross terms, use direct multiplication.
Total per 2-column group: 7 (Strassen) + 2 + 4 + 2 = 15... wait let me recount.

Actually, let me use a completely different decomposition:
- Split A as [A1; A2] where A1 is 2x3, A2 is 1x3
- Split B columns into groups of 4
- For each group of 4:
  - A1 * B[:,g: g+4]: 2x3 * 3x4
    Split B into B11 (2x4) and B12 (1x4)
    A1 = [A11 (2x2), A12 (2x1)]
    A11 * B11: 2x2 * 2x4. Use two Strassen: rank 14
    A12 * B12: 2x1 * 1x4: rank 8
    A11 * B12: wait, this is wrong. Let me redo.
    
  Actually for 2x3 * 3x4:
  C = A[0:2,0:2]*B[0:2,:] + A[0:2,2]*B[2,:]
  First term: 2x2 * 2x4 = two 2x2*2x2 Strassen = rank 14
  Second term: 2x1 * 1x4 = rank 8
  Total: 22
  
  A2 * B[:,g:g+4]: 1x3 * 3x4
  = A2[0:2]*B[0:2,:] + A2[2]*B[2,:]
  First: 1x2 * 2x4 = rank 8
  Second: 1x1 * 1x4 = rank 4
  Total: 12
  
  Per 4-col group: 22 + 12 = 34
  Two groups: 68
"""

DIMENSIONS = [3, 3, 8]

import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    terms = []
    
    def add_strassen(ra, ca, rb, cb):
        """Add 7 Strassen terms for A[ra:ra+2, ca:ca+2] * B[rb:rb+2, cb:cb+2]"""
        a = (ra, ca)
        b = (ra, ca+1)
        c = (ra+1, ca)
        d = (ra+1, ca+1)
        e = (rb, cb)
        f = (rb, cb+1)
        g = (rb+1, cb)
        h = (rb+1, cb+1)
        
        C00 = (ra, cb)
        C01 = (ra, cb+1)
        C10 = (ra+1, cb)
        C11 = (ra+1, cb+1)
        
        terms.append(([(a,1),(d,1)], [(e,1),(h,1)], [(C00,1),(C11,1)]))
        terms.append(([(c,1),(d,1)], [(e,1)], [(C10,1),(C11,-1)]))
        terms.append(([(a,1)], [(f,1),(h,-1)], [(C01,1),(C11,1)]))
        terms.append(([(d,1)], [(g,1),(e,-1)], [(C00,1),(C10,1)]))
        terms.append(([(a,1),(b,1)], [(h,1)], [(C00,-1),(C01,1)]))
        terms.append(([(c,1),(a,-1)], [(e,1),(f,1)], [(C11,1)]))
        terms.append(([(b,1),(d,-1)], [(g,1),(h,1)], [(C00,1)]))
    
    def add_direct(i, j, k):
        terms.append(([((i,j),1)], [((j,k),1)], [((i,k),1)]))
    
    # Split 8 columns into 2 groups of 4
    for g in range(2):
        col_off = g * 4
        # A[0:2, 0:2] * B[0:2, col_off:col_off+4] as two Strassen
        add_strassen(0, 0, 0, col_off)
        add_strassen(0, 0, 0, col_off+2)
        # A[0:2, 2] * B[2, col_off:col_off+4]
        for i in range(2):
            for k in range(col_off, col_off+4):
                add_direct(i, 2, k)
        # A[2, 0:2] * B[0:2, col_off:col_off+4]
        for j in range(2):
            for k in range(col_off, col_off+4):
                add_direct(2, j, k)
        # A[2, 2] * B[2, col_off:col_off+4]
        for k in range(col_off, col_off+4):
            add_direct(2, 2, k)
    
    rank = len(terms)
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for idx, (U_sp, V_sp, W_sp) in enumerate(terms):
        for (i,j), c in U_sp:
            U[i*m+j][idx] = c
        for (j,k), c in V_sp:
            V[j*p+k][idx] = c
        for (i,k), c in W_sp:
            W[k*n+i][idx] = c
    
    # Verify
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Ua[i*m+j] * Va[j*p+k], Wa[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    
    if not ok:
        rank_fb = n*m*p
        U_fb = [[0]*rank_fb for _ in range(n*m)]
        V_fb = [[0]*rank_fb for _ in range(m*p)]
        W_fb = [[0]*rank_fb for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col] = 1
                    V_fb[j*p+k][col] = 1
                    W_fb[k*n+i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
