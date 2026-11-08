"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    nm = n * m  # 6
    mp = m * p  # 15
    pn = p * n  # 10
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(nm)]
    V_sb = [[0] * rank_sb for _ in range(mp)]
    W_sb = [[0] * rank_sb for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    def is_half_int(x, tol=1e-8):
        return abs(2*x - round(2*x)) < tol
    
    def check_all_half_int(U, V, W, rank, tol=1e-8):
        for i in range(nm):
            for c in range(rank):
                if not is_half_int(U[i][c], tol):
                    return False
        for i in range(mp):
            for c in range(rank):
                if not is_half_int(V[i][c], tol):
                    return False
        for i in range(pn):
            for c in range(rank):
                if not is_half_int(W[i][c], tol):
                    return False
        return True
    
    def verify(U, V, W, rank, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try ranks from low to high
    for rank in range(3, best_rank):
        if time.time() > deadline:
            break
        
        found = False
        base_seed = 42 + rank * 13
        attempts = 0
        
        while time.time() < deadline - 2:
            np.random.seed(base_seed + attempts)
            vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            U = np.random.choice(vals, size=(nm, rank))
            V = np.random.choice(vals, size=(mp, rank))
            attempts += 1
            
            W = np.zeros((pn, rank))
            ok = True
            for k in range(p):
                for i in range(n):
                    A_sys = np.zeros((m, rank))
                    for j in range(m):
                        A_sys[j] = U[i*m+j] * V[j*p+k]
                    b = np.ones(m)
                    try:
                        w = np.linalg.lstsq(A_sys, b, rcond=1e-12)[0]
                        if np.max(np.abs(A_sys @ w - b)) > 1e-9:
                            ok = False
                            break
                        W[k*n+i] = w
                    except:
                        ok = False
                        break
                if not ok:
                    break
            
            if not ok:
                continue
            
            if not check_all_half_int(U.tolist(), V.tolist(), W.tolist(), rank):
                continue
            
            if verify(U.tolist(), V.tolist(), W.tolist(), rank):
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = rank
                found = True
                break
        
        if found:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
