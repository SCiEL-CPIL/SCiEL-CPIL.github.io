import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

def construct():
    start = time.time()
    deadline = start + 135  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 3, 5, 5

    def verify(U, V, W):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        target = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1
        T_decomp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        return np.allclose(T_decomp, target, atol=1e-10)

    def build_from_blocks(blocks):
        """Build decomposition from given 2x2x2 Strassen block positions."""
        total_rank = len(blocks) * 7 + (n * m * p - len(blocks) * 8)
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        col = 0
        for i0, j0, k0 in blocks:
            # Strassen 7 terms for 2x2x2 block
            # Term 0: (A00+A11)(B00+B11) -> contributes to C00, C11
            c = col + 0
            U[i0*m + j0][c] += 1
            U[(i0+1)*m + (j0+1)][c] += 1
            V[j0*p + k0][c] += 1
            V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += 1
            W[(k0+1)*n + (i0+1)][c] += 1
            
            # Term 1: (A10+A11)B00 -> contributes to C10, C11
            c = col + 1
            U[(i0+1)*m + j0][c] += 1
            U[(i0+1)*m + (j0+1)][c] += 1
            V[j0*p + k0][c] += 1
            W[k0*n + (i0+1)][c] += 1
            W[(k0+1)*n + (i0+1)][c] += -1
            
            # Term 2: A00(B10-B00) -> contributes to C01, C11
            c = col + 2
            U[i0*m + j0][c] += 1
            V[j0*p + (k0+1)][c] += 1
            V[(j0+1)*p + (k0+1)][c] += -1
            W[(k0+1)*n + i0][c] += 1
            W[(k0+1)*n + (i0+1)][c] += 1
            
            # Term 3: A11(B01-B10) -> contributes to C00, C01
            c = col + 3
            U[(i0+1)*m + (j0+1)][c] += 1
            V[(j0+1)*p + k0][c] += 1
            V[j0*p + k0][c] += -1
            W[k0*n + i0][c] += 1
            W[k0*n + (i0+1)][c] += 1
            
            # Term 4: (A00+A01)B11 -> contributes to C00, C01
            c = col + 4
            U[i0*m + j0][c] += 1
            U[i0*m + (j0+1)][c] += 1
            V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += -1
            W[(k0+1)*n + i0][c] += 1
            
            # Term 5: (A10-A00)(B00+B10) -> contributes to C11
            c = col + 5
            U[(i0+1)*m + j0][c] += 1
            U[i0*m + j0][c] += -1
            V[j0*p + k0][c] += 1
            V[j0*p + (k0+1)][c] += 1
            W[(k0+1)*n + (i0+1)][c] += 1
            
            # Term 6: (A01-A11)(B10+B11) -> contributes to C00
            c = col + 6
            U[i0*m + (j0+1)][c] += 1
            U[(i0+1)*m + (j0+1)][c] += -1
            V[(j0+1)*p + k0][c] += 1
            V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += 1
            
            col += 7
        
        covered = [[[False]*p for _ in range(m)] for _ in range(n)]
        for i0, j0, k0 in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        if (i0+di) < n and (j0+dj) < m and (k0+dk) < p:
                            covered[i0+di][j0+dj][k0+dk] = True
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i][j][k]:
                        U[i*m+j][col] += 1
                        V[j*p+k][col] += 1
                        W[k*n+i][col] += 1
                        col += 1
        
        return U, V, W

    # Default: current best known placement
    default_blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
    
    # Try default first
    U, V, W = build_from_blocks(default_blocks)
    best_rank = len(U[0])
    best_result = (U, V, W)
    
    if verify(U, V, W):
        best_rank = len(U[0])
        best_result = (U, V, W)
    else:
        # Fallback to trivial
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

    # Search for better block placements
    # Valid 2x2x2 block positions: i0 in {0,1}, j0 in {0,1,2,3}, k0 in {0,1,2,3}
    valid_positions = [(i0, j0, k0) for i0 in range(1) for j0 in range(3) for k0 in range(3)]
    
    # Try all combinations of 3 and 4 blocks
    from itertools import combinations
    for num_blocks in [4, 3]:
        if time.time() > deadline:
            break
        for combo in combinations(valid_positions, num_blocks):
            if time.time() > deadline:
                break
            # Check no overlapping blocks
            valid = True
            for a in range(len(combo)):
                for b in range(a+1, len(combo)):
                    i0a, j0a, k0a = combo[a]
                    i0b, j0b, k0b = combo[b]
                    # Blocks overlap if they share any cell
                    if (abs(i0a-i0b) < 2 and abs(j0a-j0b) < 2 and abs(k0a-k0b) < 2):
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            U_t, V_t, W_t = build_from_blocks(list(combo))
            r = len(U_t[0])
            if r < best_rank:
                if verify(U_t, V_t, W_t):
                    best_rank = r
                    best_result = (U_t, V_t, W_t)
    
    if time.time() > deadline:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
