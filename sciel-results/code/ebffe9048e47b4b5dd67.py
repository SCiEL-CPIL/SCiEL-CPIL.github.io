import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    for i in range(n):
        for k in range(p):
            for ip in range(n):
                for jp in range(m):
                    for jp2 in range(m):
                        for kp in range(p):
                            s = 0.0
                            for c in range(rank):
                                s += W[k*n+i, c] * U[ip*m+jp, c] * V[jp*p+kp, c]
                            target = 1.0 if (ip == i and kp == k and jp == jp2) else 0.0
                            if abs(s - target) > 1e-9:
                                return False
    return True

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook Solution
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Search for R = 89
    R = rank_schoolbook - 1
    if R > 29:
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            
            U_rand = np.random.choice([-0.5, 0.0, 0.5], size=(n*m, R))
            V_rand = np.random.choice([-0.5, 0.0, 0.5], size=(m*p, R))
            
            num_eqs = n * p * n * m * m * p
            num_vars = R * p * n
            
            A = np.zeros((num_eqs, num_vars))
            b = np.zeros(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for k in range(p):
                    for ip in range(n):
                        for jp in range(m):
                            for jp2 in range(m):
                                for kp in range(p):
                                    if (ip == i and kp == k and jp == jp2):
                                        b[eq_idx] = 1.0
                                    for c in range(R):
                                        val = U_rand[ip*m+jp, c] * V_rand[jp*p+kp, c]
                                        if val != 0:
                                            var_idx = c * (p*n) + k*n + i
                                            A[eq_idx, var_idx] = val
                                    eq_idx += 1
                                
            try:
                w_sol, residuals, rank_a, s = np.linalg.lstsq(A, b, rcond=None)
                if len(residuals) > 0 and residuals[0] < 1e-8:
                    W_sol = w_sol.reshape(R, p, n).T.reshape(p*n, R)
                    U_round = np.round(U_rand * 2) / 2
                    V_round = np.round(V_rand * 2) / 2
                    W_round = np.round(W_sol * 2) / 2
                    
                    if verify(U_round.tolist(), V_round.tolist(), W_round.tolist()):
                        return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
            except Exception:
                pass

    return {"U": best_U, "V": best_V, "W": best_W}
