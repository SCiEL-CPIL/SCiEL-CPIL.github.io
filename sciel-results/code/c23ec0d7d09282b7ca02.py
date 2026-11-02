"""Find a rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 10
    dim_v = m * p  # 25
    dim_w = p * n  # 10
    
    # Build the target tensor
    # T[a, b, c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0

    def get_schoolbook():
        """Return schoolbook decomposition as dict of lists."""
        rank = n * m * p  # 50
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify_decomposition(U, V, W, atol=1e-9):
        """Verify that U, V, W exactly decompose the target tensor."""
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T, target, atol=atol)

    def try_kronecker_rank10():
        """
        Try to construct a rank-10 decomposition using Kronecker product structure.
        M_{2,5,5} = M_{2,1,1} ⊗ M_{1,5,5}, rank ≤ 2*5 = 10.
        """
        # M_{2,1,1}: (2x1)*(1x1) -> (2x1), rank 2
        # U1[i, r1] = delta_{i,r1}, i=0,1; r1=0,1
        # V1[0, r1] = 1
        # W1[0, r1] = 1
        # M_{1,5,5}: (1x5)*(5x5) -> (1x5), rank 5
        # U2[0, r2] = 1, r2=0..4
        # V2[j*5+k, r2] = delta_{j,r2}
        # W2[k, r2] = 1
        
        rank = 10
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        
        for r1 in range(2):
            for r2 in range(5):
                r = r1 * 5 + r2
                # U: (i*5+j, r) where i is from M_{2,1,1} row, j is from M_{1,5,5} col
                # U[i*5+j, r] = U1[i, r1] * U2[j, r2] = delta_{i,r1} * 1
                for i in range(2):
                    for j in range(5):
                        U[i * 5 + j, r] = (1.0 if i == r1 else 0.0) * 1.0
                
                # V: (j*5+k, r) where j is row in B, k is col in B
                # V[j*5+k, r] = V1[0, r1] * V2[j*5+k, r2] = 1 * delta_{j,r2}
                for j in range(5):
                    for k in range(5):
                        V[j * 5 + k, r] = 1.0 * (1.0 if j == r2 else 0.0)
                
                # W: (k*2+i, r) where k is col in C, i is row in C
                # W[k*2+i, r] = W1[0, r1] * W2[k, r2] = 1 * 1
                for k in range(5):
                    for i in range(2):
                        W[k * 2 + i, r] = 1.0 * 1.0
        
        if verify_decomposition(U, V, W):
            return (U, V, W)
        return None

    def try_rank(rank, max_restarts=10, max_iter=30):
        """
        Try to find a decomposition of given rank using L-BFGS-B optimization.
        Uses structured initialization to improve convergence.
        """
        if time.time() > deadline:
            return None
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            np.random.seed(42 + restart * 13 + rank)
            
            # Initialize with a mix of random and structured components
            U = np.random.randn(dim_u, rank) * 0.5
            V = np.random.randn(dim_v, rank) * 0.5
            W = np.random.randn(dim_w, rank) * 0.5
            
            # Add some schoolbook-like structure to help convergence
            if rank >= 10:
                for col in range(min(rank // 5, 10)):
                    i = col % n
                    j = (col // n) % m
                    k = (col // (n * m)) % p
                    U[i * m + j, col] += 1.0
                    V[j * p + k, col] += 1.0
                    W[k * n + i, col] += 1.0
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def fun(params):
                u_flat = params[:dim_u * rank]
                v_flat = params[dim_u * rank : dim_u * rank + dim_v * rank]
                w_flat = params[dim_u * rank + dim_v * rank : ]
                
                U_m = u_flat.reshape(dim_u, rank)
                V_m = v_flat.reshape(dim_v, rank)
                W_m = w_flat.reshape(dim_w, rank)
                
                T = np.einsum('ar,br,cr->abc', U_m, V_m, W_m)
                diff = T - target
                loss = np.sum(diff * diff)
                
                grad_u = 2.0 * np.einsum('abc,br,cr->ar', diff, V_m, W_m)
                grad_v = 2.0 * np.einsum('abc,ar,cr->br', diff, U_m, W_m)
                grad_w = 2.0 * np.einsum('abc,ar,br->cr', diff, U_m, V_m)
                
                grad = np.concatenate([grad_u.flatten(), grad_v.flatten(), grad_w.flatten()])
                return loss, grad
            
            try:
                res = minimize(fun, params, method='L-BFGS-B', jac=True,
                               options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20})
                
                if res.fun < 1e-8:
                    u_final = res.x[:dim_u * rank].reshape(dim_u, rank)
                    v_final = res.x[dim_u * rank : dim_u * rank + dim_v * rank].reshape(dim_v, rank)
                    w_final = res.x[dim_u * rank + dim_v * rank : ].reshape(dim_w, rank)
                    
                    # Snap to nearest half-integer
                    u_snap = np.round(u_final * 2.0) / 2.0
                    v_snap = np.round(v_final * 2.0) / 2.0
                    w_snap = np.round(w_final * 2.0) / 2.0
                    
                    if verify_decomposition(u_snap, v_snap, w_snap):
                        return (u_snap, v_snap, w_snap)
                    
                    # Try cleaning small values
                    u_clean = np.where(np.abs(u_snap) < 1e-6, 0, u_snap)
                    v_clean = np.where(np.abs(v_snap) < 1e-6, 0, v_snap)
                    w_clean = np.where(np.abs(w_snap) < 1e-6, 0, w_snap)
                    if verify_decomposition(u_clean, v_clean, w_clean):
                        return (u_clean, v_clean, w_clean)
                        
                    # Try snapping to integers only
                    u_int = np.round(u_final)
                    v_int = np.round(v_final)
                    w_int = np.round(w_final)
                    if verify_decomposition(u_int, v_int, w_int):
                        return (u_int, v_int, w_int)
                        
            except Exception:
                continue
        
        return None

    # First try the Kronecker product rank-10 construction
    kr_result = try_kronecker_rank10()
    if kr_result is not None:
        U, V, W = kr_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try various ranks from lower to upper bound
    best_result = None
    
    # Try ranks around 10-20 first (in case Kronecker didn't work but something close does)
    for target_rank in [10, 12, 15, 20, 25, 30, 35, 40, 45, 49, 48, 47, 46]:
        if time.time() > deadline:
            break
        
        result = try_rank(target_rank, max_restarts=8, max_iter=40)
        if result is not None:
            U, V, W = result
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
    
    if best_result is not None:
        return best_result
    else:
        return get_schoolbook()
# EVOLVE-BLOCK-END
