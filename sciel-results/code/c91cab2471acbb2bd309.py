import time
import numpy as np

DIMENSIONS = [3, 4, 4]
_START = time.time()
_TIME_LIMIT = 140.0

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    r0 = n * m * p

    # Schoolbook decomposition
    U0 = np.zeros((nm, r0))
    V0 = np.zeros((mp, r0))
    W0 = np.zeros((pn, r0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1

    best_rank, best_U, best_V, best_W = r0, U0, V0, W0

    # Strategy: Decompose M_3,4,4 into M_3,3,2 + M_3,3,2 + M_3,1,4
    # A (3x4) -> A1 (3x3), A2 (3x1)
    # B (4x4) -> B1 (3x4), B2 (1x4)
    # B1 (3x4) -> B1a (3x2), B1b (3x2)
    # C = A1 B1a + A1 B1b + A2 B2
    
    # Find best rank for M_3,3,2
    best_332 = _find_best_rank_332()
    
    # Rank for M_3,1,4 is 12 (Schoolbook)
    r_314 = 12
    
    total_rank = 2 * best_332 + r_314
    
    if total_rank < best_rank:
        # Assemble the decomposition
        U_new, V_new, W_new = _assemble_decomposition(best_332, r_314)
        if U_new is not None:
            best_rank = total_rank
            best_U, best_V, best_W = U_new, V_new, W_new

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _find_best_rank_332():
    # Find minimal rank for 3x3x2
    # LB is 15. Try 15, 16, 17, 18
    for r in range(15, 19):
        if time.time() - _START > _TIME_LIMIT - 10:
            break
        res = _try_rank_332(r)
        if res is not None:
            return r
    return 18 # Fallback to schoolbook

def _try_rank_332(r, n_trials=10):
    n, m, p = 3, 3, 2
    nm, mp, pn = n * m, m * p, p * n
    idx = [(i * m + j, j * p + k, k * n + i)
           for i in range(n) for j in range(m) for k in range(p)]
           
    for t in range(n_trials):
        if time.time() - _START > _TIME_LIMIT - 10:
            return None
        
        rng = np.random.RandomState(t * 101 + r)
        U = rng.randn(nm, r)
        V = rng.randn(mp, r)
        W = rng.randn(pn, r)
        
        # Normalize
        for c in range(r):
            for mat in [U, V, W]:
                norm = np.linalg.norm(mat[:, c])
                if norm > 1e-10:
                    mat[:, c] /= norm

        # Gradient Descent
        for it in range(500):
            if time.time() - _START > _TIME_LIMIT - 10:
                return None
            
            Ug = np.zeros_like(U)
            Vg = np.zeros_like(V)
            Wg = np.zeros_like(W)
            loss = 0.0
            
            for ui, vi, wi in idx:
                vw = V[vi] * W[wi]
                uv = U[ui] * V[vi]
                uw = U[ui] * W[wi]
                s = np.dot(U[ui], vw)
                e = s - 1.0
                loss += e * e
                Ug[ui] += e * vw
                Vg[vi] += e * uw
                Wg[wi] += e * uv
                
            if loss < 1e-10:
                # Check for half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if _verify_332(U_r, V_r, W_r):
                    return (U_r, V_r, W_r)
                
                # If not half-integer, try to see if it's valid integer/rational
                # For now, just return the float solution if valid, 
                # but the task requires integer/half-integer.
                # So we must round.
                # If rounding fails, we might need to try harder or reject.
                # But often GD converges to a solution that is close to integers.
                
            lr = 0.01 / (1.0 + it * 0.01)
            U -= lr * Ug
            V -= lr * Vg
            W -= lr * Wg
            
    return None

def _verify_332(U, V, W):
    n, m, p = 3, 3, 2
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vi = j * p + k
                wi = k * n + i
                val = np.dot(U[ui], V[vi] * W[wi])
                if abs(val - 1.0) > 1e-5:
                    return False
    return True

def _assemble_decomposition(r_332, r_314):
    # Get the actual matrices for 3x3x2
    # We need to re-run the solver to get the matrices, or store them.
    # To save time, we assume the solver found a valid one.
    # We need to retrieve U, V, W for 3x3x2.
    # Let's modify _try_rank_332 to return matrices.
    
    # Re-implementing slightly to return matrices
    n, m, p = 3, 3, 2
    nm, mp, pn = n * m, m * p, p * n
    idx = [(i * m + j, j * p + k, k * n + i)
           for i in range(n) for j in range(m) for k in range(p)]
           
    U_332 = None
    V_332 = None
    W_332 = None
    
    for t in range(10):
        if time.time() - _START > _TIME_LIMIT - 5:
            break
        rng = np.random.RandomState(t * 101 + r_332)
        U = rng.randn(nm, r_332)
        V = rng.randn(mp, r_332)
        W = rng.randn(pn, r_332)
        
        for c in range(r_332):
            for mat in [U, V, W]:
                norm = np.linalg.norm(mat[:, c])
                if norm > 1e-10:
                    mat[:, c] /= norm

        for it in range(500):
            if time.time() - _START > _TIME_LIMIT - 5:
                break
            Ug = np.zeros_like(U)
            Vg = np.zeros_like(V)
            Wg = np.zeros_like(W)
            loss = 0.0
            for ui, vi, wi in idx:
                vw = V[vi] * W[wi]
                uv = U[ui] * V[vi]
                uw = U[ui] * W[wi]
                s = np.dot(U[ui], vw)
                e = s - 1.0
                loss += e * e
                Ug[ui] += e * vw
                Vg[vi] += e * uw
                Wg[wi] += e * uv
            if loss < 1e-10:
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if _verify_332(U_r, V_r, W_r):
                    U_332, V_332, W_332 = U_r, V_r, W_r
                    break
            lr = 0.01 / (1.0 + it * 0.01)
            U -= lr * Ug
            V -= lr * Vg
            W -= lr * Wg
        if U_332 is not None:
            break
            
    if U_332 is None:
        return None, None, None

    # Assemble 3x4x4
    # U (12 x R_total), V (16 x R_total), W (12 x R_total)
    R_total = 2 * r_332 + r_314
    U_full = np.zeros((12, R_total))
    V_full = np.zeros((16, R_total))
    W_full = np.zeros((12, R_total))
    
    # Part 1: A1 (rows 0-2) * B1a (cols 0-1 of B, rows 0-2)
    # Indices in 3x3x2:
    # U index: i*3+j (i=0..2, j=0..2) -> 0..8
    # V index: j*2+k (j=0..2, k=0..1) -> 0..5
    # W index: k*3+i (k=0..1, i=0..2) -> 0..5
    
    # Mapping to 3x4x4:
    # A is 3x4. A1 is cols 0-2.
    # B is 4x4. B1a is rows 0-2, cols 0-1.
    # C is 3x4. C1a is cols 0-1.
    
    # U_full rows 0-8 (i=0..2, j=0..2) correspond to U_332
    U_full[0:9, 0:r_332] = U_332
    
    # V_full rows for B1a: j=0..2, k=0..1
    # In 3x4x4, V index is j*4+k.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 4
    # j=1, k=1 -> 5
    # j=2, k=0 -> 8
    # j=2, k=1 -> 9
    v_indices_1 = [0, 1, 4, 5, 8, 9]
    V_full[v_indices_1, 0:r_332] = V_332
    
    # W_full cols for C1a: k=0..1, i=0..2
    # In 3x4x4, W index is k*3+i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=0, i=2 -> 2
    # k=1, i=0 -> 3
    # k=1, i=1 -> 4
    # k=1, i=2 -> 5
    w_indices_1 = [0, 1, 2, 3, 4, 5]
    W_full[w_indices_1, 0:r_332] = W_332
    
    # Part 2: A1 (rows 0-2) * B1b (cols 2-3 of B, rows 0-2)
    # U_full rows 0-8 same U_332
    U_full[0:9, r_332:2*r_332] = U_332
    
    # V_full rows for B1b: j=0..2, k=2..3
    # j=0, k=2 -> 2
    # j=0, k=3 -> 3
    # j=1, k=2 -> 6
    # j=1, k=3 -> 7
    # j=2, k=2 -> 10
    # j=2, k=3 -> 11
    v_indices_2 = [2, 3, 6, 7, 10, 11]
    V_full[v_indices_2, r_332:2*r_332] = V_332
    
    # W_full cols for C1b: k=2..3, i=0..2
    # k=2, i=0 -> 6
    # k=2, i=1 -> 7
    # k=2, i=2 -> 8
    # k=3, i=0 -> 9
    # k=3, i=1 -> 10
    # k=3, i=2 -> 11
    w_indices_2 = [6, 7, 8, 9, 10, 11]
    W_full[w_indices_2, r_332:2*r_332] = W_332
    
    # Part 3: A2 (col 3 of A) * B2 (row 3 of B)
    # A2 is 3x1. B2 is 1x4.
    # U index: i*4+3 (i=0..2) -> 3, 7, 11
    # V index: 3*4+k (k=0..3) -> 12, 13, 14, 15
    # W index: k*3+i (k=0..3, i=0..2) -> 0..11
    
    # This is M_3,1,4. Rank 12.
    # U_314: 3x12. U_314[i, c] = 1 if c corresponds to i.
    # V_314: 4x12. V_314[k, c] = 1 if c corresponds to k.
    # W_314: 12x12. W_314[wi, c] = 1 if wi corresponds to c.
    
    u_indices_3 = [3, 7, 11]
    v_indices_3 = [12, 13, 14, 15]
    
    for c in range(12):
        i = c // 4
        k = c % 4
        U_full[u_indices_3[i], 2*r_332 + c] = 1
        V_full[v_indices_3[k], 2*r_332 + c] = 1
        W_full[k*3 + i, 2*r_332 + c] = 1
        
    return U_full, V_full, W_full
