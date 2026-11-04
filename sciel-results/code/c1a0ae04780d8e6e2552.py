import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
                
    def compute_error_full(U, V, W, r, n, m, p):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Tensor T: (n*m, m*p, p*n)
        # We want T[a,b,c] = 1 if consistent, 0 otherwise.
        # Consistent if a=i*m+j, b=j*p+k, c=k*n+i.
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Build target tensor
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_target[a, b, c] = 1.0
        
        err = np.sum((T_recon - T_target) ** 2)
        return err

    def verify_solution(U, V, W, r, n, m, p, tol=1e-8):
        err = compute_error_full(U, V, W, r, n, m, p)
        if err < tol:
            if check_integer_half(U, V, W, r):
                return True
        return False

    def check_integer_half(U, V, W, r, tol=1e-9):
        if not np.all(np.abs(2*U - np.round(2*U)) < tol): return False
        if not np.all(np.abs(2*V - np.round(2*V)) < tol): return False
        if not np.all(np.abs(2*W - np.round(2*W)) < tol): return False
        return True

    def snap_to_grid(U, V, W):
        U_snap = np.round(2 * U) / 2.0
        V_snap = np.round(2 * V) / 2.0
        W_snap = np.round(2 * W) / 2.0
        return U_snap, V_snap, W_snap

    def try_rank(r, n, m, p, time_limit, max_restarts=50):
        best_err = float('inf')
        best_sol = None
        
        for restart in range(max_restarts):
            if time.time() > time_limit:
                break
            
            # Initialize with random half-integer values
            def rand_half():
                return np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
            
            U = np.array([[rand_half() for _ in range(r)] for _ in range(n * m)])
            V = np.array([[rand_half() for _ in range(r)] for _ in range(m * p)])
            W = np.array([[rand_half() for _ in range(r)] for _ in range(p * n)])
            
            curr_err = compute_error_full(U, V, W, r, n, m, p)
            
            if curr_err < best_err:
                best_err = curr_err
                best_sol = (U.copy(), V.copy(), W.copy())
            
            if curr_err < 1e-10:
                U_snap, V_snap, W_snap = snap_to_grid(U, V, W)
                if verify_solution(U_snap, V_snap, W_snap, r, n, m, p):
                    return U_snap, V_snap, W_snap
            
            # Local search
            no_improve = 0
            max_iters = 2000
            step_size = 0.5
            
            for _ in range(max_iters):
                if time.time() > time_limit:
                    break
                
                var_type = np.random.randint(3)
                if var_type == 0:
                    a = np.random.randint(n * m)
                    t = np.random.randint(r)
                    old_val = U[a, t]
                    U[a, t] += np.random.choice([-1, 1]) * step_size
                elif var_type == 1:
                    b = np.random.randint(m * p)
                    t = np.random.randint(r)
                    old_val = V[b, t]
                    V[b, t] += np.random.choice([-1, 1]) * step_size
                else:
                    c = np.random.randint(p * n)
                    t = np.random.randint(r)
                    old_val = W[c, t]
                    W[c, t] += np.random.choice([-1, 1]) * step_size
                
                new_err = compute_error_full(U, V, W, r, n, m, p)
                
                if new_err < curr_err:
                    curr_err = new_err
                    no_improve = 0
                    if np.random.rand() < 0.05:
                        step_size = min(step_size * 1.1, 2.0)
                    if curr_err < 1e-10:
                        U_snap, V_snap, W_snap = snap_to_grid(U, V, W)
                        if verify_solution(U_snap, V_snap, W_snap, r, n, m, p):
                            return U_snap, V_snap, W_snap
                else:
                    if var_type == 0:
                        U[a, t] = old_val
                    elif var_type == 1:
                        V[b, t] = old_val
                    else:
                        W[c, t] = old_val
                    no_improve += 1
                    if no_improve % 10 == 0:
                        step_size *= 0.95
                    if no_improve > 50:
                        break
                        
            if curr_err < 0.5:
                U_snap, V_snap, W_snap = snap_to_grid(U, V, W)
                snap_err = compute_error_full(U_snap, V_snap, W_snap, r, n, m, p)
                if snap_err < 1e-8:
                    if verify_solution(U_snap, V_snap, W_snap, r, n, m, p):
                        return U_snap, V_snap, W_snap
        
        return None

    # Try to find a rank 21 decomposition
    sol = try_rank(21, n, m, p, deadline, max_restarts=100)
    
    if sol:
        U, V, W = sol
        U_list = [[int(x) if x == int(x) else x for x in row] for row in U]
        V_list = [[int(x) if x == int(x) else x for x in row] for row in V]
        W_list = [[int(x) if x == int(x) else x for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
