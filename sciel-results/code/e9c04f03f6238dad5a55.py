import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
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

    def validate(U, V, W, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)
    
    def snap_to_half(arr):
        return np.round(arr * 2) / 2

    def try_snap_validate(U, V, W):
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        if validate(U_s, V_s, W_s):
            return True, U_s, V_s, W_s
        return False, None, None, None

    def build_rank46():
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        
        def add_strassen(j_off, k_off, col_start):
            c = col_start
            # P1 = (A00 + A11)(B00 + B11)
            U[0 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P2 = (A10 + A11) B00
            U[1 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            W[k_off * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1
            
            # P3 = A00 (B01 - B11)
            U[0 * m + j_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P4 = A11 (B10 - B00)
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[j_off * p + k_off][c] -= 1
            W[k_off * n + 0][c] += 1
            W[k_off * n + 1][c] += 1
            c += 1
            
            # P5 = (A00 + A01) B11
            U[0 * m + j_off][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1
            
            # P6 = (A10 - A00)(B00 + B01)
            U[1 * m + j_off][c] += 1
            U[0 * m + j_off][c] -= 1
            V[j_off * p + k_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P7 = (A01 - A11)(B10 + B11)
            U[0 * m + (j_off + 1)][c] += 1
            U[1 * m + (j_off + 1)][c] -= 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1

        add_strassen(0, 0, 0)
        add_strassen(0, 2, 7)
        add_strassen(2, 0, 14)
        add_strassen(2, 2, 21)
        
        col = 28
        for i in range(2):
            for k in range(4):
                U[i * m + 4][col] = 1
                V[4 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                if time.time() > deadline - 10: return U, V, W
                
        for i in range(2):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
                if time.time() > deadline - 10: return U, V, W
                
        for i in range(2):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            if time.time() > deadline - 10: return U, V, W
            
        return U, V, W

    def loss_and_grad(U, V, W):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = prod - T
        loss = np.sum(diff * diff)
        gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
        gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
        gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
        return loss, (gradU, gradV, gradW)

    def optimize_rank(target_rank, max_iters=200, seed=42):
        if time.time() > deadline - 20:
            return None, None, None
            
        np.random.seed(seed)
        # Initialize with small random noise
        U = np.random.randn(U_dim, target_rank) * 0.1
        V = np.random.randn(V_dim, target_rank) * 0.1
        W = np.random.randn(W_dim, target_rank) * 0.1
        
        # Flatten parameters
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(params):
            if time.time() > deadline - 10:
                return 1e10
            U_p = params[:U_dim*target_rank].reshape(U_dim, target_rank)
            V_p = params[U_dim*target_rank:U_dim*target_rank+V_dim*target_rank].reshape(V_dim, target_rank)
            W_p = params[U_dim*target_rank+V_dim*target_rank:].reshape(W_dim, target_rank)
            loss, _ = loss_and_grad(U_p, V_p, W_p)
            return loss
        
        def gradient(params):
            if time.time() > deadline - 10:
                return np.zeros_like(params)
            U_p = params[:U_dim*target_rank].reshape(U_dim, target_rank)
            V_p = params[U_dim*target_rank:U_dim*target_rank+V_dim*target_rank].reshape(V_dim, target_rank)
            W_p = params[U_dim*target_rank+V_dim*target_rank:].reshape(W_dim, target_rank)
            _, grads = loss_and_grad(U_p, V_p, W_p)
            return np.concatenate([grads[0].flatten(), grads[1].flatten(), grads[2].flatten()])

        # Use L-BFGS-B for better convergence on smooth problems
        # maxiter is limited by time check inside objective/grad
        res = minimize(objective, params, jac=gradient, method='L-BFGS-B', 
                       options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
        
        if time.time() > deadline - 5:
            return None, None, None

        best_params = res.x
        U_opt = best_params[:U_dim*target_rank].reshape(U_dim, target_rank)
        V_opt = best_params[U_dim*target_rank:U_dim*target_rank+V_dim*target_rank].reshape(V_dim, target_rank)
        W_opt = best_params[U_dim*target_rank+V_dim*target_rank:].reshape(W_dim, target_rank)
        
        valid, U_s, V_s, W_s = try_snap_validate(U_opt, V_opt, W_opt)
        if valid:
            return U_s, V_s, W_s
        return None, None, None

    # Build and verify rank-46
    U46, V46, W46 = build_rank46()
    if not validate(U46, V46, W46):
        # Fallback to standard rank 50 if something went wrong
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }

    best_result = None
    
    # Try to find rank 45
    # We use multiple seeds to explore the landscape
    for rank_to_try in [45, 44]:
        if time.time() > deadline - 60:
            break
            
        max_seeds = 10 if rank_to_try == 45 else 5
        for seed in range(max_seeds):
            if time.time() > deadline - 30:
                break
                
            U_opt, V_opt, W_opt = optimize_rank(rank_to_try, max_iters=100, seed=42 + seed)
            if U_opt is not None:
                best_result = (U_opt, V_opt, W_opt)
                break
        if best_result is not None:
            break

    if best_result is not None:
        U_r, V_r, W_r = best_result
        # Verify the result
        if validate(U_r, V_r, W_r):
            return {
                "U": [[float(x) for x in row] for row in U_r],
                "V": [[float(x) for x in row] for row in V_r],
                "W": [[float(x) for x in row] for row in W_r]
            }
    
    return {
        "U": [[float(x) for x in row] for row in U46],
        "V": [[float(x) for x in row] for row in V46],
        "W": [[float(x) for x in row] for row in W46]
    }
# EVOLVE-BLOCK-END
