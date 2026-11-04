import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Target tensor for 3x3x5
    T_full = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_full[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook fallback
    rank_school = n * m * p
    U_sch = np.zeros((n*m, rank_school))
    V_sch = np.zeros((m*p, rank_school))
    W_sch = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i*m+j, col] = 1
                V_sch[j*p+k, col] = 1
                W_sch[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_sch.tolist(), "V": V_sch.tolist(), "W": W_sch.tolist()}
    
    def check_exact(U, V, W, T_target):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_target, T_approx, atol=1e-9)

    # Strategy: Find rank 23 decomposition for 3x3x3, combine with schoolbook 3x3x2 (rank 18)
    # Total rank 41.
    
    n3, m3, p3 = 3, 3, 3
    T_333 = np.zeros((n3*m3, m3*p3, p3*n3))
    for i in range(n3):
        for j in range(m3):
            for k in range(p3):
                T_333[i*m3+j, j*p3+k, k*n3+i] = 1.0
                
    n3b, m3b, p3b = 3, 3, 2
    T_332 = np.zeros((n3b*m3b, m3b*p3b, p3b*n3b))
    for i in range(n3b):
        for j in range(m3b):
            for k in range(p3b):
                T_332[i*m3b+j, j*p3b+k, k*n3b+i] = 1.0

    # Schoolbook for 3x3x2
    rank_332 = 18
    U_332_sch = np.zeros((n3b*m3b, rank_332))
    V_332_sch = np.zeros((m3b*p3b, rank_332))
    W_332_sch = np.zeros((p3b*n3b, rank_332))
    col = 0
    for i in range(n3b):
        for j in range(m3b):
            for k in range(p3b):
                U_332_sch[i*m3b+j, col] = 1
                V_332_sch[j*p3b+k, col] = 1
                W_332_sch[k*n3b+i, col] = 1
                col += 1

    def find_333_decomp(target_rank):
        # Discrete local search
        # Domain: -1, -0.5, 0, 0.5, 1
        vals = [-1, -0.5, 0, 0.5, 1]
        
        # Initialize with schoolbook terms (27 terms)
        # We need to reduce to target_rank.
        # This is hard to do directly with local search on fixed size.
        # Instead, we initialize random discrete vectors of size target_rank.
        
        best_loss = np.inf
        best_U = None
        best_V = None
        best_W = None
        
        max_restarts = 50 if target_rank == 23 else 20
        max_iters = 1000
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            U = np.random.choice(vals, size=(n3*m3, target_rank))
            V = np.random.choice(vals, size=(m3*p3, target_rank))
            W = np.random.choice(vals, size=(p3*n3, target_rank))
            
            # Local search: flip one random element in U, V, or W
            current_loss = np.sum((T_333 - np.einsum('ar,br,cr->abc', U, V, W))**2)
            
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    break
                    
                if current_loss < 1e-9:
                    return U, V, W
                    
                # Pick a random matrix and a random element
                mat_idx = np.random.randint(3)
                if mat_idx == 0:
                    r, c = np.random.randint(n3*m3), np.random.randint(target_rank)
                    old_val = U[r, c]
                    new_val = np.random.choice(vals)
                    if new_val == old_val: continue
                    U[r, c] = new_val
                    new_loss = np.sum((T_333 - np.einsum('ar,br,cr->abc', U, V, W))**2)
                    if new_loss < current_loss:
                        current_loss = new_loss
                    else:
                        U[r, c] = old_val
                elif mat_idx == 1:
                    r, c = np.random.randint(m3*p3), np.random.randint(target_rank)
                    old_val = V[r, c]
                    new_val = np.random.choice(vals)
                    if new_val == old_val: continue
                    V[r, c] = new_val
                    new_loss = np.sum((T_333 - np.einsum('ar,br,cr->abc', U, V, W))**2)
                    if new_loss < current_loss:
                        current_loss = new_loss
                    else:
                        V[r, c] = old_val
                else:
                    r, c = np.random.randint(p3*n3), np.random.randint(target_rank)
                    old_val = W[r, c]
                    new_val = np.random.choice(vals)
                    if new_val == old_val: continue
                    W[r, c] = new_val
                    new_loss = np.sum((T_333 - np.einsum('ar,br,cr->abc', U, V, W))**2)
                    if new_loss < current_loss:
                        current_loss = new_loss
                    else:
                        W[r, c] = old_val
                        
            if current_loss < best_loss:
                best_loss = current_loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
        if best_loss < 1e-9:
            return best_U, best_V, best_W
        return None

    # Try to find rank 23 for 3x3x3
    if time.time() < deadline - 10:
        res_333 = find_333_decomp(23)
        if res_333 is not None:
            U_333, V_333, W_333 = res_333
            # Combine with 3x3x2 schoolbook
            U_total = np.hstack([U_333, U_332_sch])
            V_total = np.hstack([V_333, V_332_sch])
            W_total = np.hstack([W_333, W_332_sch])
            
            if check_exact(U_total, V_total, W_total, T_full):
                return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
            
    # Fallback to schoolbook
    return best_result
