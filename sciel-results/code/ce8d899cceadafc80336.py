import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10

    # Build target tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def validate(U, V, W, atol=1e-8):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)

    def snap_and_validate(U, V, W):
        # Try multiple snap granularities
        for denom in [1, 2, 3, 4, 5, 6, 10, 12]:
            scale = float(denom)
            U_s = np.round(U * scale) / scale
            V_s = np.round(V * scale) / scale
            W_s = np.round(W * scale) / scale
            if validate(U_s, V_s, W_s):
                return True, U_s, V_s, W_s
        return False, None, None

    def objective(x, T, rank):
        U = x[:U_dim * rank].reshape(U_dim, rank)
        V = x[U_dim * rank : U_dim * rank + V_dim * rank].reshape(V_dim, rank)
        W = x[U_dim * rank + V_dim * rank :].reshape(W_dim, rank)
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = prod - T
        return np.sum(diff * diff)

    def get_trivial_solution():
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def try_merge_columns(U, V, W, col_a, col_b, alpha):
        """Try to merge column b into column a with coefficient alpha"""
        U_new = U.copy()
        V_new = V.copy()
        W_new = W.copy()
        
        U_new[:, col_a] = U_new[:, col_a] + alpha * U_new[:, col_b]
        V_new[:, col_a] = V_new[:, col_a] + alpha * V_new[:, col_b]
        W_new[:, col_a] = W_new[:, col_a] + alpha * W_new[:, col_b]
        
        # Remove column b
        U_new = np.delete(U_new, col_b, axis=1)
        V_new = np.delete(V_new, col_b, axis=1)
        W_new = np.delete(W_new, col_b, axis=1)
        
        return U_new, V_new, W_new

    def greedy_merge(U, V, W):
        """Greedily try to merge columns to reduce rank"""
        U_curr, V_curr, W_curr = U.copy(), V.copy(), W.copy()
        rank = U_curr.shape[1]
        
        for _ in range(rank - 1):
            if time.time() > deadline - 5:
                break
            if U_curr.shape[1] <= 1:
                break
            
            best_merge = None
            best_error = float('inf')
            best_alpha = 0
            best_pair = None
            
            for a in range(U_curr.shape[1]):
                for b in range(a + 1, U_curr.shape[1]):
                    # Try several alpha values
                    for alpha in [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]:
                        U_new, V_new, W_new = try_merge_columns(U_curr, V_curr, W_curr, a, b, alpha)
                        prod = np.einsum('ar,br,cr->abc', U_new, V_new, W_new, optimize=True)
                        error = np.sum((prod - T) ** 2)
                        if error < best_error:
                            best_error = error
                            best_alpha = alpha
                            best_pair = (a, b)
            
            if best_pair is not None and best_error < 1e-6:
                a, b = best_pair
                U_curr, V_curr, W_curr = try_merge_columns(U_curr, V_curr, W_curr, a, b, best_alpha)
            else:
                break
        
        return U_curr, V_curr, W_curr

    def structured_init(rank, seed):
        """Create structured initialization based on matrix multiplication structure"""
        np.random.seed(seed)
        
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        
        # Assign each (i,j,k) triple to a rank component
        assigned = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    # Try to find an existing rank component that can cover this
                    found = False
                    for r in range(rank):
                        # Check if this rank component already has entries for i,j or j,k or k,i
                        if U[u_idx, r] != 0 and V[v_idx, r] == 0 and W[w_idx, r] == 0:
                            # Can add V and W entries
                            V[v_idx, r] = 1
                            W[w_idx, r] = 1
                            assigned[(i, j, k)] = r
                            found = True
                            break
                        elif U[u_idx, r] == 0 and V[v_idx, r] != 0 and W[w_idx, r] == 0:
                            U[u_idx, r] = 1
                            W[w_idx, r] = 1
                            assigned[(i, j, k)] = r
                            found = True
                            break
                        elif U[u_idx, r] == 0 and V[v_idx, r] == 0 and W[w_idx, r] != 0:
                            U[u_idx, r] = 1
                            V[v_idx, r] = 1
                            assigned[(i, j, k)] = r
                            found = True
                            break
                    
                    if not found:
                        # Assign to a new rank component if available
                        for r in range(rank):
                            if U[:, r].sum() == 0:
                                U[u_idx, r] = 1
                                V[v_idx, r] = 1
                                W[w_idx, r] = 1
                                assigned[(i, j, k)] = r
                                break
        
        return U, V, W

    U50, V50, W50 = get_trivial_solution()
    
    best_rank = 50
    best_U, best_V, best_W = U50, V50, W50

    # First, try greedy merging from the trivial solution
    U_merged, V_merged, W_merged = greedy_merge(U50, V50, W50)
    if validate(U_merged, V_merged, W_merged):
        valid, U_s, V_s, W_s = snap_and_validate(U_merged, V_merged, W_merged)
        if valid:
            best_rank = U_s.shape[1]
            best_U, best_V, best_W = U_s, V_s, W_s

    # Try structured initializations and optimization for lower ranks
    for target_rank in range(48, 15, -1):
        if time.time() > deadline - 20:
            break
        if target_rank >= best_rank:
            continue
            
        found = False
        for seed in range(20):
            if time.time() > deadline - 10:
                    break
                    
            # Try structured initialization
            U_init, V_init, W_init = structured_init(target_rank, seed)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                # Optimization with tighter tolerances
                res = minimize(
                    objective, 
                    x0, 
                    args=(T, target_rank), 
                    method='L-BFGS-B', 
                    options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-12}
                )
                
                U_opt = res.x[:U_dim * target_rank].reshape(U_dim, target_rank)
                V_opt = res.x[U_dim * target_rank : U_dim * target_rank + V_dim * target_rank].reshape(V_dim, target_rank)
                W_opt = res.x[U_dim * target_rank + V_dim * target_rank :].reshape(W_dim, target_rank)
                
                valid, U_s, V_s, W_s = snap_and_validate(U_opt, V_opt, W_opt)
                if valid:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_s, V_s, W_s
                    found = True
                    break
            except Exception:
                pass

        if found:
            continue

    return {
        "U": [[float(x) for x in row] for row in best_U],
        "V": [[float(x) for x in row] for row in best_V],
        "W": [[float(x) for x in row] for row in best_W]
    }
