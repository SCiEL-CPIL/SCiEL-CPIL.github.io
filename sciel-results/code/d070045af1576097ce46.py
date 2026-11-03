import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 3, 8
    dim_U = n * m  # 6
    dim_V = m * p  # 24
    dim_W = p * n  # 16
    
    t_start = time.time()
    deadline = t_start + 140.0
    
    # Build the known rank-44 decomposition
    def build_rank44():
        rank = 44
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        col = 0
        
        # Part 1: j=0, 16 terms
        for k in range(p):
            U[0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + 0, col] = 1
            col += 1
            U[1 * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + 1, col] = 1
            col += 1
        
        # Part 2: j=1,2 in 4 blocks of 2x2x2, 7 terms each = 28
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            # P1
            U[0 * m + 1, col] = 1
            U[1 * m + 2, col] = 1
            V[1 * p + k0, col] = 1
            V[2 * p + k1, col] = 1
            W[k0 * n + 0, col] = 1
            W[k1 * n + 1, col] = 1
            col += 1
            
            # P2
            U[1 * m + 1, col] = 1
            U[1 * m + 2, col] = 1
            V[1 * p + k0, col] = 1
            W[k0 * n + 1, col] = 1
            W[k1 * n + 1, col] = -1
            col += 1
            
            # P3
            U[0 * m + 1, col] = 1
            V[1 * p + k1, col] = 1
            V[2 * p + k1, col] = -1
            W[k1 * n + 0, col] = 1
            W[k1 * n + 1, col] = 1
            col += 1
            
            # P4
            U[1 * m + 2, col] = 1
            V[2 * p + k0, col] = 1
            V[1 * p + k0, col] = -1
            W[k0 * n + 0, col] = 1
            W[k0 * n + 1, col] = 1
            col += 1
            
            # P5
            U[0 * m + 1, col] = 1
            U[0 * m + 2, col] = 1
            V[2 * p + k1, col] = 1
            W[k0 * n + 0, col] = -1
            W[k1 * n + 0, col] = 1
            col += 1
            
            # P6
            U[1 * m + 1, col] = 1
            U[0 * m + 1, col] = -1
            V[1 * p + k0, col] = 1
            V[1 * p + k1, col] = 1
            W[k1 * n + 1, col] = 1
            col += 1
            
            # P7
            U[0 * m + 2, col] = 1
            U[1 * m + 2, col] = -1
            V[2 * p + k0, col] = 1
            V[2 * p + k1, col] = 1
            W[k0 * n + 0, col] = 1
            col += 1
        
        return U, V, W, rank
    
    U44, V44, W44, rank44 = build_rank44()
    best_U, best_V, best_W = U44.copy(), V44.copy(), W44.copy()
    best_rank = rank44
    
    # Verification function
    def verify(U, V, W, tol=1e-9):
        R = U.shape[1]
        for i0 in range(n):
            for j0 in range(m):
                for k0 in range(p):
                    for i in range(n):
                        for k in range(p):
                            val = 0
                            for r in range(R):
                                val += W[k * n + i, r] * U[i * m + j0, r] * V[j0 * p + k, r]
                            expected = 1.0 if (i == i0 and k == k0) else 0.0
                            if abs(val - expected) > tol:
                                return False
        return True
    
    # Try to find lower-rank solutions using optimization
    def try_optimize(R, U_init=None, V_init=None, W_init=None, max_iter=2000):
        if time.time() > deadline:
            return None
        
        size_U, size_V, size_W = dim_U, dim_V, dim_W
        total_size = R * (size_U + size_V + size_W)
        
        # Initialize
        if U_init is not None:
            params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        else:
            params = np.random.randn(total_size) * 0.1
        
        def residuals(params):
            U = params[0:R*size_U].reshape(R, size_U)
            V = params[R*size_U:R*size_U + R*size_V].reshape(R, size_V)
            W = params[R*size_U + R*size_V:].reshape(R, size_W)
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(W[:, k*n+i] * U[:, i*m+j] * V[:, j*p+k])
                        res.append(val - 1.0)
            return np.array(res)
        
        try:
            result = least_squares(residuals, params, method='lm', max_nfev=max_iter)
        except Exception:
            return None
        
        if result.cost < 1e-12:
            U = result.x[0:R*size_U].reshape(R, size_U)
            V = result.x[R*size_U:R*size_U + R*size_V].reshape(R, size_V)
            W = result.x[R*size_U + R*size_V:].reshape(R, size_W)
            
            # Snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            if verify(U_snap, V_snap, W_snap):
                return U_snap, V_snap, W_snap
        
        return None
    
    # Try ranks from 43 down to 30
    for target_rank in range(43, 29, -1):
        if time.time() > deadline:
            break
        
        found = False
        
        # Try with different initializations
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            if attempt == 0 and target_rank < rank44:
                # Initialize from rank-44 by zeroing some columns
                U_init = U44[:, :target_rank].copy()
                V_init = V44[:, :target_rank].copy()
                W_init = W44[:, :target_rank].copy()
                result = try_optimize(target_rank, U_init, V_init, W_init, max_iter=1000)
            elif attempt == 1:
                # Random initialization
                result = try_optimize(target_rank, max_iter=1500)
            else:
                # Perturbed rank-44
                U_init = U44[:, :target_rank].copy() + np.random.randn(dim_U, target_rank) * 0.01
                V_init = V44[:, :target_rank].copy() + np.random.randn(dim_V, target_rank) * 0.01
                W_init = W44[:, :target_rank].copy() + np.random.randn(dim_W, target_rank) * 0.01
                result = try_optimize(target_rank, U_init, V_init, W_init, max_iter=1500)
            
            if result is not None:
                U_opt, V_opt, W_opt = result
                best_U, best_V, best_W = U_opt, V_opt, W_opt
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    # Convert to list format
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}

def _verify(U, V, W):
    n, m, p = 2, 3, 8
    R = len(U[0]) if U else 0
    
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                for i in range(n):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            if i == i0 and k == k0:
                                val += W[k * n + i][r] * U[i * m + j0][r] * V[j0 * p + k][r]
                        expected_val = 1 if (i == i0 and k == k0) else 0
                        if val != expected_val:
                            return False
    return True

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    if _verify(res["U"], res["V"], res["W"]):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
