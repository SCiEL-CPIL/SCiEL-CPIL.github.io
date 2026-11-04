import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    def build_schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def check_solution(U, V, W, n, m, p, tol=1e-9):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    if abs(val - 1.0) > tol:
                        return False
        # Check integer/half-integer
        if not np.all(np.abs(2 * U - np.round(2 * U)) < tol): return False
        if not np.all(np.abs(2 * V - np.round(2 * V)) < tol): return False
        if not np.all(np.abs(2 * W - np.round(2 * W)) < tol): return False
        return True

    def find_decomp(n, m, p, rank, time_limit):
        # Try to find a rank-`rank` decomposition for M_{n,m,p}
        # Variables: U (nm*rank), V (mp*rank), W (pn*rank)
        # Total vars: (nm + mp + pn) * rank
        nm, mp, pn = n * m, m * p, p * n
        total_vars = (nm + mp + pn) * rank
        
        def residuals(x):
            U = x[:nm * rank].reshape(nm, rank)
            V = x[nm * rank : nm * rank + mp * rank].reshape(mp, rank)
            W = x[nm * rank + mp * rank :].reshape(pn, rank)
            
            res = np.zeros(n * m * p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                        res[idx] = val - 1.0
                        idx += 1
            return res

        best_sol = None
        best_cost = float('inf')
        
        # Try a few random restarts
        max_restarts = 10
        for restart in range(max_restarts):
            if time.time() > time_limit:
                break
            
            x0 = np.random.randn(total_vars) * 0.5
            # Add some structure: maybe start with schoolbook if rank >= nmp?
            # Here rank < nmp usually, so random is fine.
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=1000)
                cost = result.cost
                if cost < best_cost:
                    best_cost = cost
                    best_sol = result.x
            except Exception:
                pass

        if best_sol is None:
            return None

        U = best_sol[:nm * rank].reshape(nm, rank)
        V = best_sol[nm * rank : nm * rank + mp * rank].reshape(mp, rank)
        W = best_sol[nm * rank + mp * rank :].reshape(pn, rank)
        
        # Snap to grid
        U_snap = np.round(2 * U) / 2.0
        V_snap = np.round(2 * V) / 2.0
        W_snap = np.round(2 * W) / 2.0
        
        if check_solution(U_snap, V_snap, W_snap, n, m, p):
            return U_snap, V_snap, W_snap
        
        # If snap fails, maybe the continuous solution was close but not exact?
        # Try a local polish on the snapped values?
        # Or just return None.
        return None

    # Strategy:
    # 1. Try to find rank 7 for 3x3x3
    # 2. Use schoolbook for 3x3x2 (rank 9)
    # 3. Concatenate
    
    # Time allocation
    time_for_333 = time.time() + 30
    time_for_332 = time.time() + 10 # Schoolbook is instant, so just in case
    
    # Find 3x3x3 rank 7
    U333, V333, W333 = find_decomp(3, 3, 3, 7, time_for_333)
    
    if U333 is None:
        # Fallback to schoolbook for 3x3x3
        U333, V333, W333 = build_schoolbook(3, 3, 3)
        rank_333 = 9
    else:
        rank_333 = 7

    # Find 3x3x2 rank 9 (Schoolbook)
    U332, V332, W332 = build_schoolbook(3, 3, 2)
    rank_332 = 6 # Wait, schoolbook rank for 3x3x2 is 3*3*2 = 18?
    # No, schoolbook rank is n*m*p.
    # For 3x3x2, rank is 18.
    # But we established lower bound is 9.
    # Is rank 9 achievable?
    # Let's try to find rank 9 for 3x3x2.
    
    # Actually, let's just use schoolbook for 3x3x2 to be safe, rank 18.
    # Total rank would be 7+7+18 = 32.
    # Or 9+9+18 = 36.
    # Still much better than 72.
    
    # Let's try to find rank 9 for 3x3x2.
    U332_opt, V332_opt, W332_opt = find_decomp(3, 3, 2, 9, time_for_332)
    if U332_opt is not None:
        U332, V332, W332 = U332_opt, V332_opt, W332_opt
        rank_332 = 9
    else:
        # Fallback to schoolbook
        U332, V332, W332 = build_schoolbook(3, 3, 2)
        rank_332 = 18

    # Construct final matrices
    total_rank = rank_333 + rank_333 + rank_332
    U_final = np.zeros((3 * 3, total_rank))
    V_final = np.zeros((3 * 8, total_rank))
    W_final = np.zeros((8 * 3, total_rank))
    
    # Block 1: k=0,1,2
    offset = 0
    U_final[:, offset:offset+rank_333] = U333
    # V333 is (3*3, rank). V_final is (3*8, rank).
    # V333[j*3+k, r] -> V_final[j*8+k, offset+r]
    for j in range(3):
        for k in range(3):
            V_final[j * 8 + k, offset:offset+rank_333] = V333[j * 3 + k, :]
    # W333 is (3*3, rank). W_final is (8*3, rank).
    # W333[k*3+i, r] -> W_final[k*3+i, offset+r]
    for k in range(3):
        for i in range(3):
            W_final[k * 3 + i, offset:offset+rank_333] = W333[k * 3 + i, :]
            
    # Block 2: k=3,4,5
    offset += rank_333
    U_final[:, offset:offset+rank_333] = U333
    for j in range(3):
        for k in range(3):
            V_final[j * 8 + (k + 3), offset:offset+rank_333] = V333[j * 3 + k, :]
    for k in range(3):
        for i in range(3):
            W_final[(k + 3) * 3 + i, offset:offset+rank_333] = W333[k * 3 + i, :]

    # Block 3: k=6,7
    offset += rank_333
    U_final[:, offset:offset+rank_332] = U332
    for j in range(3):
        for k in range(2):
            V_final[j * 8 + (k + 6), offset:offset+rank_332] = V332[j * 2 + k, :]
    for k in range(2):
        for i in range(3):
            W_final[(k + 6) * 3 + i, offset:offset+rank_332] = W332[k * 3 + i, :]

    # Verify final solution
    if check_solution(U_final, V_final, W_final, 3, 3, 8):
        U_list = [[int(x) if x == int(x) else x for x in row] for row in U_final]
        V_list = [[int(x) if x == int(x) else x for x in row] for row in V_final]
        W_list = [[int(x) if x == int(x) else x for x in row] for row in W_final]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to full schoolbook
        U_school, V_school, W_school = build_schoolbook(3, 3, 8)
        U_list = [[int(x) if x == int(x) else x for x in row] for row in U_school]
        V_list = [[int(x) if x == int(x) else x for x in row] for row in V_school]
        W_list = [[int(x) if x == int(x) else x for x in row] for row in W_school]
        return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
