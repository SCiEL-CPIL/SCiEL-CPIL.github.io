import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

def construct():
    start = time.time()
    deadline = start + 140.0
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # 4 non-overlapping 2x2x2 Strassen blocks
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2),
    ]
    
    total_rank = len(blocks) * 7 + (n * m * p - len(blocks) * 8)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        # M1 = (A00+A11)(B00+B11)
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M2 = (A10+A11)B00
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        # M3 = A00(B01-B11)
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M4 = A11(B10-B00)
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        # M5 = (A00+A01)B11
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        # M6 = (A10-A00)(B00+B01)
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M7 = (A01-A11)(B10+B11)
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
    
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    # Mark covered triples
    covered = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered[i0+di][j0+dj][k0+dk] = True
    
    # Try to find a better decomposition of the residual
    # The residual has 43 terms. Let's try to group them.
    # Residual terms: all (i,j,k) not covered
    # i=2: 5*5=25 terms (j=0..4, k=0..4)
    # i=0,1 with j=4: 2*5=10 terms (k=0..4)
    # i=0,1 with k=4 and j=0..3: 2*4=8 terms
    # Total: 25+10+8 = 43
    
    # Strategy: Try to decompose the residual with rank < 43
    # Use a greedy approach: try to find rank-1 tensors in the residual
    # that can be combined with the Strassen structure
    
    # For now, use schoolbook for the residual (rank 43)
    # But try a small optimization: see if any pairs of residual terms
    # can be combined (they can't directly, but let's verify)
    
    residual_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i][j][k]:
                    residual_terms.append((i, j, k))
    
    # Try to find if we can reduce the rank by using a different block placement
    # or by finding a rank-deficient substructure in the residual
    
    # Alternative: try 3 blocks of size 2x2x2 + handle the rest differently
    # Or try to find a 3x3x3 sub-block that can be decomposed more efficiently
    
    # Let's try: can we use a Strassen-like trick on the i=2 row?
    # The i=2 row has 25 terms. If we could decompose a 1x5x5 as rank < 25...
    # But a 1x5x5 tensor is just a 5x5 matrix, which has rank 5 (if full rank).
    # However, our tensor is not a matrix multiplication in the usual sense.
    
    # Actually, let's try a different approach: use the fact that the problem
    # is 3x5x5, not 4x4x4. Maybe we can find a better tiling.
    
    # Try: 2 blocks of 2x2x2 + 1 block of 1x2x2 or similar
    # Actually, let's just stick with the known good decomposition and verify
    
    for (i, j, k) in residual_terms:
        U[i*m+j][col] += 1
        V[j*p+k][col] += 1
        W[k*n+i][col] += 1
        col += 1
    
    # Verify correctness
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        target = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1
        T_decomp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        if not np.allclose(T_decomp, target, atol=1e-10):
            raise ValueError("Verification failed")
    except Exception:
        # Fallback to schoolbook
        rank = n * m * p
        U_f = [[0]*rank for _ in range(n*m)]
        V_f = [[0]*rank for _ in range(m*p)]
        W_f = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i*m+j][c] = 1
                    V_f[j*p+k][c] = 1
                    W_f[k*n+i][c] = 1
                    c += 1
        return {"U": U_f, "V": V_f, "W": W_f}
    
    if time.time() > deadline:
        rank = n * m * p
        U_f = [[0]*rank for _ in range(n*m)]
        V_f = [[0]*rank for _ in range(m*p)]
        W_f = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i*m+j][c] = 1
                    V_f[j*p+k][c] = 1
                    W_f[k*n+i][c] = 1
                    c += 1
        return {"U": U_f, "V": V_f, "W": W_f}
    
    return {"U": U, "V": V, "W": W}
