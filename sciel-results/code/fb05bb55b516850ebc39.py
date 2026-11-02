import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 4, 7
    
    # Rank 50 solution
    def get_rank_50():
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        def add_strassen_222(i0, j0, k0):
            nonlocal col
            if col >= rank: return False
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
            
            U[a][col] += 1; U[d][col] += 1; V[e][col] += 1; V[h][col] += 1; W[c00][col] += 1; W[c11][col] += 1; col += 1
            U[c][col] += 1; U[d][col] += 1; V[e][col] += 1; W[c10][col] += 1; W[c11][col] -= 1; col += 1
            U[a][col] += 1; V[f][col] += 1; V[h][col] -= 1; W[c01][col] += 1; W[c11][col] += 1; col += 1
            U[d][col] += 1; V[g][col] += 1; V[e][col] -= 1; W[c00][col] += 1; W[c10][col] += 1; col += 1
            U[a][col] += 1; U[b][col] += 1; V[h][col] += 1; W[c00][col] -= 1; W[c01][col] += 1; col += 1
            U[c][col] += 1; U[a][col] -= 1; V[e][col] += 1; V[f][col] += 1; W[c11][col] += 1; col += 1
            U[b][col] += 1; U[d][col] -= 1; V[g][col] += 1; V[h][col] += 1; W[c00][col] += 1; col += 1
            return True

        def add_schoolbook_221(i0, j0, k0):
            nonlocal col
            if col >= rank: return False
            a = i0 * m + j0
            b = i0 * m + (j0 + 1)
            c = (i0 + 1) * m + j0
            d = (i0 + 1) * m + (j0 + 1)
            e = j0 * p + k0
            g = (j0 + 1) * p + k0
            c00 = k0 * n + i0
            c10 = k0 * n + (i0 + 1)
            
            U[a][col] += 1; V[e][col] += 1; W[c00][col] += 1; col += 1
            U[b][col] += 1; V[g][col] += 1; W[c00][col] += 1; col += 1
            U[c][col] += 1; V[e][col] += 1; W[c10][col] += 1; col += 1
            U[d][col] += 1; V[g][col] += 1; W[c10][col] += 1; col += 1
            return True

        for k0 in [0, 2, 4]:
            add_strassen_222(0, 0, k0)
        add_schoolbook_221(0, 0, 6)
        for k0 in [0, 2, 4]:
            add_strassen_222(0, 2, k0)
        add_schoolbook_221(0, 2, 6)
        
        return {"U": U, "V": V, "W": W}

    best_solution = get_rank_50()
    best_rank = 50
    
    # Try to find a better solution for 2x2x7 subproblem
    # This is a complex search, so we might skip it if time is tight
    # For now, we return the known 50-rank solution
    # A structural change would be to implement the search, but given the constraints,
    # ensuring the 50-rank solution is robust and fast is priority.
    
    # Verification
    U_arr = np.array(best_solution["U"], dtype=float)
    V_arr = np.array(best_solution["V"], dtype=float)
    W_arr = np.array(best_solution["W"], dtype=float)
    
    # Check a few entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                reconstructed = np.sum(U_arr[a_idx] * V_arr[b_idx] * W_arr[c_idx])
                if abs(reconstructed - 1) > 1e-9:
                    # Fallback if verification fails
                    return {"U": [[0]*50 for _ in range(n*m)], "V": [[0]*50 for _ in range(m*p)], "W": [[0]*50 for _ in range(p*n)]}
    
    return best_solution
