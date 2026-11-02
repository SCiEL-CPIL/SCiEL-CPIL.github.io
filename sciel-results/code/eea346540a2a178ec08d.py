import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 130.0  # Safety margin

    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    # Dimensions
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n

    # Tensor T as a dictionary of non-zeros: {(a, b, c): 1}
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[(a, b, c)] = 1

    # 3D array for fast computation
    T_arr = np.zeros((dim_u, dim_v, dim_w))
    for (a, b, c), val in T.items():
        T_arr[a, b, c] = val

    np.random.seed(42)

    # Search for lower rank
    # Lower bound is max(dim_u, dim_v, dim_w) = 20
    # We try ranks from 20 up to best_rank - 1
    # Focus on 20-40 first as known ranks are low
    for r in range(20, min(40, best_rank)):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Number of trials per rank
        # More trials for lower ranks
        num_trials = 50
        if r < 25:
            num_trials = 100
        elif r < 30:
            num_trials = 80
        else:
            num_trials = 50

        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            # Initialize with small random integers
            # Use sparse initialization sometimes
            if trial % 2 == 0:
                # Dense
                U = np.random.randint(-2, 3, size=(dim_u, r)).astype(float)
                V = np.random.randint(-2, 3, size=(dim_v, r)).astype(float)
                W = np.random.randint(-2, 3, size=(dim_w, r)).astype(float)
            else:
                # Sparse
                U = np.zeros((dim_u, r))
                V = np.zeros((dim_v, r))
                W = np.zeros((dim_w, r))
                for _ in range(int(dim_u * r * 0.3)):
                    U[np.random.randint(0, dim_u), np.random.randint(0, r)] = np.random.choice([-1, 1, 2])
                for _ in range(int(dim_v * r * 0.3)):
                    V[np.random.randint(0, dim_v), np.random.randint(0, r)] = np.random.choice([-1, 1, 2])
                for _ in range(int(dim_w * r * 0.3)):
                    W[np.random.randint(0, dim_w), np.random.randint(0, r)] = np.random.choice([-1, 1, 2])
            
            max_iters = 50
            prev_err = float('inf')
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                M_VW = np.zeros((dim_v * dim_w, r))
                for b in range(dim_v):
                    for c in range(dim_w):
                        M_VW[b * dim_w + c, :] = V[b, :] * W[c, :]
                
                y_U = T_arr.reshape(dim_u, dim_v * dim_w)
                
                try:
                    M_VW_T = M_VW.T
                    y_U_T = y_U.T
                    U_sol = np.linalg.lstsq(M_VW_T, y_U_T, rcond=None)[0]
                    U = U_sol.T
                except Exception:
                    break

                # Update V
                M_UW = np.zeros((dim_u * dim_w, r))
                for a in range(dim_u):
                    for c in range(dim_w):
                        M_UW[a * dim_w + c, :] = U[a, :] * W[c, :]
                
                y_V = T_arr.transpose(1, 0, 2).reshape(dim_v, dim_u * dim_w)
                
                try:
                    M_UW_T = M_UW.T
                    y_V_T = y_V.T
                    V_sol = np.linalg.lstsq(M_UW_T, y_V_T, rcond=None)[0]
                    V = V_sol.T
                except Exception:
                    break

                # Update W
                M_UV = np.zeros((dim_u * dim_v, r))
                for a in range(dim_u):
                    for b in range(dim_v):
                        M_UV[a * dim_v + b, :] = U[a, :] * V[b, :]
                
                y_W = T_arr.transpose(2, 0, 1).reshape(dim_w, dim_u * dim_v)
                
                try:
                    M_UV_T = M_UV.T
                    y_W_T = y_W.T
                    W_sol = np.linalg.lstsq(M_UV_T, y_W_T, rcond=None)[0]
                    W = W_sol.T
                except Exception:
                    break

                # Calculate error periodically
                if it % 10 == 0 or it == max_iters - 1:
                    T_est = np.einsum('at,bt,ct->abc', U, V, W)
                    err = np.linalg.norm(T_est - T_arr)
                    
                    if err < 1e-6:
                        # Try rational reconstruction
                        valid = False
                        for denom in [1, 2, 4, 8, 16, 32]:
                            U_rat = [[Fraction(U[a, t]).limit_denominator(denom) for t in range(r)] for a in range(dim_u)]
                            V_rat = [[Fraction(V[b, t]).limit_denominator(denom) for t in range(r)] for b in range(dim_v)]
                            W_rat = [[Fraction(W[c, t]).limit_denominator(denom) for t in range(r)] for c in range(dim_w)]
                            
                            # Verify exactly
                            ok = True
                            for (a, b, c), val in T.items():
                                s = sum(U_rat[a][t] * V_rat[b][t] * W_rat[c][t] for t in range(r))
                                if s != val:
                                    ok = False
                                    break
                            
                            if ok:
                                for a in range(dim_u):
                                    for b in range(dim_v):
                                        for c in range(dim_w):
                                            if (a, b, c) not in T:
                                                s = sum(U_rat[a][t] * V_rat[b][t] * W_rat[c][t] for t in range(r))
                                                if s != 0:
                                                    ok = False
                                                    break
                                        if not ok: break
                                    if not ok: break
                            
                            if ok:
                                best = {"U": U_rat, "V": V_rat, "W": W_rat}
                                best_rank = r
                                valid = True
                                break
                        
                        if valid:
                            found = True
                            break

            if found:
                break
        
        if found:
            break

    return best
