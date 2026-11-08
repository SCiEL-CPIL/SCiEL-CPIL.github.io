"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time

def _verify(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = U[i*m+j]
                vj = V[j*p+k]
                wk = W[k*n+i]
                for c in range(rank):
                    s += ui[c] * vj[c] * wk[c]
                if s != 1.0:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Greedy column removal
    while time.time() < deadline:
        found = False
        current_rank = len(U[0])
        for c in range(current_rank):
            if time.time() > deadline:
                break
            U_new = [row[:c] + row[c+1:] for row in U]
            V_new = [row[:c] + row[c+1:] for row in V]
            W_new = [row[:c] + row[c+1:] for row in W]
            if _verify(U_new, V_new, W_new, n, m, p):
                U, V, W = U_new, V_new, W_new
                found = True
                if len(U[0]) < best_rank:
                    best_rank = len(U[0])
                    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
                break
        if not found:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
