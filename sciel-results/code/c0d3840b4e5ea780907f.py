import numpy as np
import time

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    def calc_error(U, V, W, rank, n, m, p):
        S = np.einsum('sij,sjk,sik->ijk', U, V, W)
        return np.sum((S - 1.0)**2)

    def get_rank_30_solution():
        rank = 30
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}

    def try_rank(rank, max_restarts=20):
        for attempt in range(max_restarts):
            if time.time() > deadline - 10:
                return None
                
            np.random.seed(attempt * 100 + rank)
            vals = np.arange(-2, 2.5, 0.5)
            
            U = np.random.choice(vals, size=(rank, n, m))
            V = np.random.choice(vals, size=(rank, m, p))
            
            W_real = np.zeros((rank, n, p))
            residual = 0.0
            possible = True
            
            for i in range(n):
                for k in range(p):
                    M_ik = np.zeros((m, rank))
                    for j in range(m):
                        for s in range(rank):
                            M_ik[j, s] = U[s, i, j] * V[s, j, k]
                    
                    try:
                        W_ik, residuals, rank_m, sv = np.linalg.lstsq(M_ik, np.ones(m), rcond=None)
                        res_vec = M_ik @ W_ik - np.ones(m)
                        res_val = np.sum(res_vec**2)
                        residual += res_val
                        W_real[:, i, k] = W_ik
                    except:
                        possible = False
                        break
                if not possible:
                    break
            
            if not possible:
                continue
                
            if residual > 1e-4:
                continue

            W = np.round(W_real * 2) / 2
            error = calc_error(U, V, W, rank, n, m, p)
            
            max_iters = 5000
            iter_count = 0
            S = np.einsum('sij,sjk,sik->ijk', U, V, W)
            
            while error > 1e-9 and iter_count < max_iters:
                if time.time() > deadline - 5:
                    break
                iter_count += 1
                tensor_id = np.random.choice(3)
                
                if tensor_id == 0:
                    s, i, j = np.random.randint(rank), np.random.randint(n), np.random.randint(m)
                    old_val = U[s, i, j]
                    new_val = old_val + np.random.choice([-0.5, 0.5])
                    if new_val < -5 or new_val > 5: continue
                    delta = new_val - old_val
                    contrib = V[s, j, :] * W[s, i, :]
                    old_S = S[i, j, :]
                    new_S = old_S + delta * contrib
                    old_err = np.sum((old_S - 1.0)**2)
                    new_err = np.sum((new_S - 1.0)**2)
                    if new_err < old_err - 1e-12:
                        U[s, i, j] = new_val
                        S[i, j, :] = new_S
                        error += (new_err - old_err)
                elif tensor_id == 1:
                    s, j, k = np.random.randint(rank), np.random.randint(m), np.random.randint(p)
                    old_val = V[s, j, k]
                    new_val = old_val + np.random.choice([-0.5, 0.5])
                    if new_val < -5 or new_val > 5: continue
                    delta = new_val - old_val
                    contrib = np.array([U[s, i, j] * W[s, i, k] for i in range(n)])
                    old_S = S[:, j, k]
                    new_S = old_S + delta * contrib
                    old_err = np.sum((old_S - 1.0)**2)
                    new_err = np.sum((new_S - 1.0)**2)
                    if new_err < old_err - 1e-12:
                        V[s, j, k] = new_val
                        S[:, j, k] = new_S
                        error += (new_err - old_err)
                else:
                    s, i, k = np.random.randint(rank), np.random.randint(n), np.random.randint(p)
                    old_val = W[s, i, k]
                    new_val = old_val + np.random.choice([-0.5, 0.5])
                    if new_val < -5 or new_val > 5: continue
                    delta = new_val - old_val
                    contrib = np.array([U[s, i, j] * V[s, j, k] for j in range(m)])
                    old_S = S[i, :, k]
                    new_S = old_S + delta * contrib
                    old_err = np.sum((old_S - 1.0)**2)
                    new_err = np.sum((new_S - 1.0)**2)
                    if new_err < old_err - 1e-12:
                        W[s, i, k] = new_val
                        S[i, :, k] = new_S
                        error += (new_err - old_err)

            if error < 1e-9:
                U_list = [[U[s, i, j] for s in range(rank)] for i in range(n) for j in range(m)]
                V_list = [[V[s, j, k] for s in range(rank)] for j in range(m) for k in range(p)]
                W_list = [[W[s, i, k] for s in range(rank)] for k in range(p) for i in range(n)]
                return {"U": U_list, "V": V_list, "W": W_list}
                
        return None

    for rank in [22, 21, 20, 19, 18]:
        if time.time() > deadline - 20: break
        result = try_rank(rank, max_restarts=10)
        if result: return result
        
    return get_rank_30_solution()
