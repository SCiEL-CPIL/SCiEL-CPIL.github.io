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

    # Precompute indices for faster access
    # T_flat is not needed if we use the dict or a 3D array.
    # A 3D array is faster for ALS.
    T_arr = np.zeros((dim_u, dim_v, dim_w))
    for (a, b, c), val in T.items():
        T_arr[a, b, c] = val

    np.random.seed(42)

    # Search for lower rank
    # Lower bound is max(dim_u, dim_v, dim_w) = 20
    # We try ranks from 20 up to best_rank - 1
    for r in range(20, best_rank):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Number of trials per rank
        # We can afford more trials for lower ranks if they fail quickly
        num_trials = 20
        if r < 30:
            num_trials = 50
        elif r < 40:
            num_trials = 30
        else:
            num_trials = 10

        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            # Initialize with small random integers or floats
            # Using small integers helps with rational reconstruction later
            U = np.random.randint(-2, 3, size=(dim_u, r)).astype(float)
            V = np.random.randint(-2, 3, size=(dim_v, r)).astype(float)
            W = np.random.randint(-2, 3, size=(dim_w, r)).astype(float)
            
            # Scale to avoid immediate overflow/underflow issues, though values are small
            # U, V, W = U * 0.5, V * 0.5, W * 0.5
            
            max_iters = 100
            prev_err = float('inf')
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                # For each a, solve M_a u_a = y_a
                # M_a is (dim_v * dim_w) x r
                # y_a is (dim_v * dim_w)
                # We can construct the full system for all a at once?
                # No, the M matrix depends on a only through the target y_a.
                # The matrix M_a is the same structure but entries depend on V and W.
                # Actually, M_a[b*C+c, t] = V[b,t] * W[c,t]. This is independent of a!
                # So we can build the matrix M once per iteration for U update.
                
                # Build M_VW: shape (dim_v * dim_w, r)
                M_VW = np.zeros((dim_v * dim_w, r))
                for b in range(dim_v):
                    for c in range(dim_w):
                        M_VW[b * dim_w + c, :] = V[b, :] * W[c, :]
                
                # y_U: shape (dim_u, dim_v * dim_w)
                y_U = T_arr.reshape(dim_u, dim_v * dim_w)
                
                # Solve U: U[a, :] = argmin || M_VW.T @ U[a, :] - y_U[a, :] || ?
                # Wait, the equation is sum_t U[a,t] M_VW[bc, t] = T[a,b,c].
                # This is U[a, :] @ M_VW[bc, :] = T[a,b,c].
                # Or M_VW.T @ U[a, :] = y_U[a, :].
                # We solve M_VW.T @ U[a, :] = y_U[a, :] for U[a, :].
                # This is a least squares problem: U[a, :] = (M_VW.T)^+ y_U[a, :]
                # We can use np.linalg.lstsq on the whole matrix at once.
                # M_VW.T is (r, dim_v * dim_w)
                # y_U is (dim_u, dim_v * dim_w)
                # We want U (dim_u, r) such that U @ M_VW = y_U ?
                # No.
                # Let X = U.T (r, dim_u).
                # We want X.T @ M_VW = y_U.
                # M_VW.T @ X = y_U.T.
                # X = np.linalg.lstsq(M_VW.T, y_U.T, rcond=None)[0]
                # U = X.T
                
                try:
                    M_VW_T = M_VW.T # (r, dim_v * dim_w)
                    y_U_T = y_U.T # (dim_v * dim_w, dim_u)
                    U_sol = np.linalg.lstsq(M_VW_T, y_U_T, rcond=None)[0] # (r, dim_u)
                    U = U_sol.T # (dim_u, r)
                except Exception:
                    break

                # Update V
                # M_UW: (dim_u * dim_w, r)
                # M_UW[ac, t] = U[a,t] * W[c,t]
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
                # M_UV: (dim_u * dim_v, r)
                # M_UV[ab, t] = U[a,t] * V[b,t]
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

                # Calculate error
                # T_est[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
                # We can compute this efficiently?
                # T_est = np.einsum('at,bt,ct->abc', U, V, W)
                # err = np.linalg.norm(T_est - T_arr)
                
                # To save time, we can compute error only every few iterations or at the end.
                # But we need to check convergence.
                if it % 5 == 0 or it == max_iters - 1:
                    T_est = np.einsum('at,bt,ct->abc', U, V, W)
                    err = np.linalg.norm(T_est - T_arr)
                    if err < 1e-8:
                        # Check if we can rationalize
                        U_rat = [[Fraction(U[a, t]).limit_denominator(100) for t in range(r)] for a in range(dim_u)]
                        V_rat = [[Fraction(V[b, t]).limit_denominator(100) for t in range(r)] for b in range(dim_v)]
                        W_rat = [[Fraction(W[c, t]).limit_denominator(100) for t in range(r)] for c in range(dim_w)]
                        
                        # Verify exactly
                        valid = True
                        for (a, b, c), val in T.items():
                            s = sum(U_rat[a][t] * V_rat[b][t] * W_rat[c][t] for t in range(r))
                            if s != val:
                                valid = False
                                break
                        
                        if valid:
                            # Check zero entries
                            for a in range(dim_u):
                                for b in range(dim_v):
                                    for c in range(dim_w):
                                        if (a, b, c) not in T:
                                            s = sum(U_rat[a][t] * V_rat[b][t] * W_rat[c][t] for t in range(r))
                                            if s != 0:
                                                valid = False
                                                break
                                    if not valid: break
                                if not valid: break
                            
                            if valid:
                                best = {"U": U_rat, "V": V_rat, "W": W_rat}
                                best_rank = r
                                found = True
                                break
                    if err < 1e-10:
                        # Very small error, but rationalization failed?
                        # Try higher denominator
                        U_rat = [[Fraction(U[a, t]).limit_denominator(1000) for t in range(r)] for a in range(dim_u)]
                        V_rat = [[Fraction(V[b, t]).limit_denominator(1000) for t in range(r)] for b in range(dim_v)]
                        W_rat = [[Fraction(W[c, t]).limit_denominator(1000) for t in range(r)] for c in range(dim_w)]
                        
                        valid = True
                        for (a, b, c), val in T.items():
                            s = sum(U_rat[a][t] * V_rat[b][t] * W_rat[c][t] for t in range(r))
                            if s != val:
                                valid = False
                                break
                        
                        if valid:
                            for a in range(dim_u):
                                for b in range(dim_v):
                                    for c in range(dim_w):
                                        if (a, b, c) not in T:
                                            s = sum(U_rat[a][t] * V_rat[b][t] * W_rat[c][t] for t in range(r))
                                            if s != 0:
                                                valid = False
                                                break
                                    if not valid: break
                                if not valid: break
                            
                            if valid:
                                best = {"U": U_rat, "V": V_rat, "W": W_rat}
                                best_rank = r
                                found = True
                                break

            if found:
                break
        
        if found:
            break

    return best
