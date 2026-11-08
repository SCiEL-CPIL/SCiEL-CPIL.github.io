import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p

    # Build the target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook solution
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    best_rank = schoolbook_rank
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()

    np.random.seed(42)

    def compute_tensor(U_mat, V_mat, W_mat):
        return np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)

    def objective_and_grads(U_mat, V_mat, W_mat):
        T_comp = compute_tensor(U_mat, V_mat, W_mat)
        diff = T_comp - T
        obj = np.sum(diff**2)
        dU = np.einsum('abc,br,cr->ar', diff, V_mat, W_mat)
        dV = np.einsum('abc,ar,cr->br', diff, U_mat, W_mat)
        dW = np.einsum('abc,ar,br->cr', diff, U_mat, V_mat)
        return obj, 2*dU, 2*dV, 2*dW

    def try_rank(rank, max_restarts=5):
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None, None, None, False
            
            # Structured initialization:
            # Split into two 2x2x6 subproblems
            # Initialize each with random values scaled appropriately
            U_mat = np.random.randn(n*m, rank) * 0.5
            V_mat = np.random.randn(m*p, rank) * 0.5
            W_mat = np.random.randn(p*n, rank) * 0.5
            
            lr = 0.1
            obj, _, _, _ = objective_and_grads(U_mat, V_mat, W_mat)
            best_obj = obj
            stall_count = 0
            max_iter = 1000
            
            for it in range(max_iter):
                if time.time() > deadline:
                    return None, None, None, False
                
                obj, dU, dV, dW = objective_and_grads(U_mat, V_mat, W_mat)
                
                step_scale = lr / (1 + np.max(np.abs(dU)) + np.max(np.abs(dV)) + np.max(np.abs(dW)))
                
                U_new = U_mat - step_scale * dU
                V_new = V_mat - step_scale * dV
                W_new = W_mat - step_scale * dW
                
                obj_new, _, _, _ = objective_and_grads(U_new, V_new, W_new)
                
                if obj_new < best_obj - 1e-12:
                    U_mat, V_mat, W_mat = U_new, V_new, W_new
                    best_obj = obj_new
                    stall_count = 0
                    lr *= 1.01
                else:
                    stall_count += 1
                    lr *= 0.7
                    if lr < 1e-10 or stall_count > 100:
                        break
                
                if best_obj < 1e-8:
                    break
            
            if best_obj < 1e-6:
                # Polish
                lr = 1e-3
                for _ in range(500):
                    if time.time() > deadline:
                        break
                    obj, dU, dV, dW = objective_and_grads(U_mat, V_mat, W_mat)
                    step_scale = lr / (1 + np.max(np.abs(dU)) + np.max(np.abs(dV)) + np.max(np.abs(dW)))
                    U_mat -= step_scale * dU
                    V_mat -= step_scale * dV
                    W_mat -= step_scale * dW
                    lr *= 0.995
                    if obj < 1e-10:
                        break
                
                final_obj = objective_and_grads(U_mat, V_mat, W_mat)[0]
                if final_obj < 1e-6:
                    return U_mat, V_mat, W_mat, True
        
        return None, None, None, False

    # Try ranks 28, 32, 36, 40, 44, 48
    # 28 is the theoretical bound based on R(2,2,3)=7
    for rank in [28, 32, 36, 40, 44, 48]:
        if time.time() > deadline:
            break
        if rank >= best_rank:
            continue
            
        U_mat, V_mat, W_mat, found = try_rank(rank, max_restarts=10)
        
        if found:
            best_rank = rank
            best_U = U_mat.copy()
            best_V = V_mat.copy()
            best_W = W_mat.copy()
            break
    
    rank = best_rank
    U_out = [[0.0] * rank for _ in range(n * m)]
    V_out = [[0.0] * rank for _ in range(m * p)]
    W_out = [[0.0] * rank for _ in range(p * n)]
    
    for a in range(n * m):
        for r in range(rank):
            U_out[a][r] = float(best_U[a, r])
    for b in range(m * p):
        for r in range(rank):
            V_out[b][r] = float(best_V[b, r])
    for c in range(p * n):
        for r in range(rank):
            W_out[c][r] = float(best_W[c, r])
    
    return {"U": U_out, "V": V_out, "W": W_out}
