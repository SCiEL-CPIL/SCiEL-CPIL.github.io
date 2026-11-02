"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time as _time

def _build_strassen_2x2x2(p, t, U, V, W, base, n=2):
    """Add Strassen rank-7 terms for 2x2x2 block t into arrays."""
    v_b00 = 0 * p + 2 * t
    v_b01 = 0 * p + 2 * t + 1
    v_b10 = 1 * p + 2 * t
    v_b11 = 1 * p + 2 * t + 1
    
    w_c00 = (2 * t) * n + 0
    w_c01 = (2 * t + 1) * n + 0
    w_c10 = (2 * t) * n + 1
    w_c11 = (2 * t + 1) * n + 1
    
    for s in range(7):
        c = base + s
        if s == 0:
            U[0][c] += 1; U[3][c] += 1
            V[v_b00][c] += 1; V[v_b11][c] += 1
            W[w_c00][c] += 1; W[w_c11][c] += 1
        elif s == 1:
            U[2][c] += 1; U[3][c] += 1
            V[v_b00][c] += 1
            W[w_c10][c] += 1; W[w_c11][c] += -1
        elif s == 2:
            U[0][c] += 1
            V[v_b01][c] += 1; V[v_b11][c] += -1
            W[w_c01][c] += 1; W[w_c11][c] += 1
        elif s == 3:
            U[3][c] += 1
            V[v_b10][c] += 1; V[v_b00][c] += -1
            W[w_c00][c] += 1; W[w_c10][c] += 1
        elif s == 4:
            U[0][c] += 1; U[1][c] += 1
            V[v_b11][c] += 1
            W[w_c00][c] += -1; W[w_c01][c] += 1
        elif s == 5:
            U[2][c] += 1; U[0][c] += -1
            V[v_b00][c] += 1; V[v_b01][c] += 1
            W[w_c11][c] += 1
        elif s == 6:
            U[1][c] += 1; U[3][c] += -1
            V[v_b10][c] += 1; V[v_b11][c] += 1
            W[w_c00][c] += 1

def _tensor_residual(U, V, W, n, m, p):
    """Compute max absolute residual of the tensor decomposition."""
    max_res = 0.0
    # Check all (a,b,c) where T should be 1
    for i in range(n):
        for j in range(p):
            for k in range(m):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                pred = np.dot(U[a], V[b] * W[c])
                max_res = max(max_res, abs(pred - 1.0))
    # Check zeros
    for i in range(n):
        for k in range(m):
            a = i * m + k
            for kp in range(m):
                for j in range(p):
                    b = kp * p + j
                    for ip in range(n):
                        for jp in range(p):
                            c = jp * n + ip
                            if not (i == ip and k == kp and j == jp):
                                pred = np.dot(U[a], V[b] * W[c])
                                max_res = max(max_res, abs(pred))
    return max_res

def _try_find_2x2x4_rank13(deadline):
    """Try to find a rank-13 decomposition for 2x2 x 2x4 matmul."""
    n, m, p = 2, 2, 4
    rank = 13
    
    for seed in range(80):
        if _time.time() > deadline:
            break
        rng = np.random.default_rng(seed * 7919 + 1)
        U = rng.standard_normal((n*m, rank)) * 0.3
        V = rng.standard_normal((m*p, rank)) * 0.3
        W = rng.standard_normal((p*n, rank)) * 0.3
        
        best_res = float('inf')
        for it in range(150):
            if _time.time() > deadline:
                break
            
            # Update U
            for a in range(n*m):
                i = a // m; k = a % m
                M = np.zeros((m*p*p*n, rank))
                rhs = np.zeros(m*p*p*n)
                idx = 0
                for kp in range(m):
                    for j in range(p):
                        b = kp * p + j
                        for jp in range(p):
                            for ip in range(n):
                                c = jp * n + ip
                                M[idx] = V[b] * W[c]
                                if i == ip and k == kp and j == jp:
                                    rhs[idx] = 1.0
                                idx += 1
                sol = np.linalg.lstsq(M, rhs, rcond=None)[0]
                U[a] = sol
            
            # Update V
            for b in range(m*p):
                kp = b // p; j = b % p
                M = np.zeros((n*m*p*n, rank))
                rhs = np.zeros(n*m*p*n)
                idx = 0
                for i in range(n):
                    for k in range(m):
                        a = i * m + k
                        for jp in range(p):
                            for ip in range(n):
                                c = jp * n + ip
                                M[idx] = U[a] * W[c]
                                if i == ip and k == kp and j == jp:
                                    rhs[idx] = 1.0
                                idx += 1
                sol = np.linalg.lstsq(M, rhs, rcond=None)[0]
                V[b] = sol
            
            # Update W
            for c in range(p*n):
                jp = c // n; ip = c % n
                M = np.zeros((n*m*m*p, rank))
                rhs = np.zeros(n*m*m*p)
                idx = 0
                for i in range(n):
                    for k in range(m):
                        a = i * m + k
                        for kp in range(m):
                            for j in range(p):
                                b = kp * p + j
                                M[idx] = U[a] * V[b]
                                if i == ip and k == kp and j == jp:
                                    rhs[idx] = 1.0
                                idx += 1
                sol = np.linalg.lstsq(M, rhs, rcond=None)[0]
                W[c] = sol
            
            res = _tensor_residual(U, V, W, n, m, p)
            if res < best_res:
                best_res = res
                if res < 1e-10:
                    return U.copy(), V.copy(), W.copy()
    
    return None

def construct():
    """
    Rank-39 exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    Tries rank-13 per 2x2x4 block; falls back to Strassen rank-42.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    deadline = _time.time() + 105
    
    # Try rank-13 for 2x2x4
    result = _try_find_2x2x4_rank13(deadline)
    
    if result is not None:
        U_24, V_24, W_24 = result
        num_blocks = p // 4  # 3
        rank = num_blocks * 13  # 39
        
        U = [[0.0] * rank for _ in range(n*m)]
        V = [[0.0] * rank for _ in range(m*p)]
        W = [[0.0] * rank for _ in range(p*n)]
        
        for t in range(num_blocks):
            base = t * 13
            for r in range(13):
                c = base + r
                for a in range(n*m):
                    U[a][c] = float(U_24[a, r])
                for k in range(m):
                    for j in range(4):
                        b_local = k * 4 + j
                        b_global = k * p + 4*t + j
                        V[b_global][c] = float(V_24[b_local, r])
                for j in range(4):
                    for i in range(n):
                        c_local = j * n + i
                        c_global = (4*t + j) * n + i
                        W[c_global][c] = float(W_24[c_local, r])
        
        return {"U": U, "V": V, "W": W}
    
    # Fallback: Strassen rank-42
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(num_blocks):
        _build_strassen_2x2x2(p, t, U, V, W, t * 7, n)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
