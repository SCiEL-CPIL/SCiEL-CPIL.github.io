import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for 150s limit
    
    n, m, p = DIMENSIONS
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_222(i0, j0, k0):
        nonlocal col
        if col >= rank:
            return False
            
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)
        
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h) -> C00 += 1, C11 += 1
        U[a][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        V[h][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M2 = (c+d)e -> C10 += 1, C11 -= 1
        U[c][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        W[c10][col] += 1
        W[c11][col] -= 1
        col += 1
        
        # M3 = a(f-h) -> C01 += 1, C11 += 1
        U[a][col] += 1
        V[f][col] += 1
        V[h][col] -= 1
        W[c01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M4 = d(g-e) -> C00 += 1, C10 += 1
        U[d][col] += 1
        V[g][col] += 1
        V[e][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M5 = (a+b)h -> C00 -= 1, C01 += 1
        U[a][col] += 1
        U[b][col] += 1
        V[h][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f) -> C11 += 1
        U[c][col] += 1
        U[a][col] -= 1
        V[e][col] += 1
        V[f][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h) -> C00 += 1
        U[b][col] += 1
        U[d][col] -= 1
        V[g][col] += 1
        V[h][col] += 1
        W[c00][col] += 1
        col += 1
        
        return True

    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        if col >= rank:
            return False
            
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)
        
        # P1 = a*e
        U[a][col] += 1
        V[e][col] += 1
        W[c00][col] += 1
        col += 1
        
        # P2 = b*g
        U[b][col] += 1
        V[g][col] += 1
        W[c00][col] += 1
        col += 1
        
        # P3 = c*e
        U[c][col] += 1
        V[e][col] += 1
        W[c10][col] += 1
        col += 1
        
        # P4 = d*g
        U[d][col] += 1
        V[g][col] += 1
        W[c10][col] += 1
        col += 1
        
        return True

    # Structure:
    # A is 2x4. Split into A_left (cols 0,1) and A_right (cols 2,3).
    # B is 4x7. Split into B_top (rows 0,1) and B_bot (rows 2,3).
    # C = A_left @ B_top + A_right @ B_bot
    
    # For A_left @ B_top:
    # B_top is 2x7. Split B_top into cols 0-1, 2-3, 4-5, 6.
    # Blocks: (0,0), (0,1), (0,2) are 2x2x2 Strassen. (0,3) is 2x2x1 Schoolbook.
    
    # For A_right @ B_bot:
    # B_bot is 2x7. Split B_bot into cols 0-1, 2-3, 4-5, 6.
    # Blocks: (1,0), (1,1), (1,2) are 2x2x2 Strassen. (1,3) is 2x2x1 Schoolbook.
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # Generate blocks for A_left @ B_top
    for k0 in [0, 2, 4]:
        if not add_strassen_222(0, 0, k0):
            break
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # Last column of B_top is k=6 (single col)
    # This is a 2x2x1 block. A rows 0,1 cols 0,1. B rows 0,1 col 6.
    # i0=0, j0=0, k0=6.
    if not add_schoolbook_221(0, 0, 6):
        pass
        
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # Generate blocks for A_right @ B_bot
    # A_right: rows 0,1, cols 2,3. i0=0, j0=2.
    # B_bot: rows 2,3, cols k. j0=2.
    for k0 in [0, 2, 4]:
        if not add_strassen_222(0, 2, k0):
            break
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # Last column of B_bot is k=6
    # i0=0, j0=2, k0=6.
    if not add_schoolbook_221(0, 2, 6):
        pass

    # Verification: Check that the decomposition is exact
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_error = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                reconstructed = 0
                for r in range(rank):
                    reconstructed += U_arr[a_idx, r] * V_arr[b_idx, r] * W_arr[c_idx, r]
                if abs(reconstructed - 1) > 1e-9:
                    max_error = max(max_error, abs(reconstructed - 1))
    
    if max_error > 1e-9:
        # Fallback: return empty (invalid) if verification fails
        return {"U": [[0]*rank for _ in range(n*m)], 
                "V": [[0]*rank for _ in range(m*p)], 
                "W": [[0]*rank for _ in range(p*n)]}
    
    return {"U": U, "V": V, "W": W}
