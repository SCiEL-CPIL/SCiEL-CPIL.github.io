import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s margin
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U, V, W, best_rank = schoolbook()
    
    def check_solution(U, V, W, tol=1e-6):
        """Check if the tensor decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        for i in range(n):
            for k in range(p):
                for b in range(m):
                    for c in range(m):
                        val = 0.0
                        for r in range(rank):
                            val += W[k*n+i, r] * U[i*m+b, r] * V[c*p+k, r]
                        expected = 1.0 if b == c else 0.0
                        if abs(val - expected) > tol:
                            return False
        return True
    
    def to_half_int(arr):
        return np.round(arr * 2) / 2
    
    def objective_simple(x, rank):
        n, m, p = DIMENSIONS
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
        W = x[n*m*rank+m*p*rank:].reshape(p*n, rank)
        
        U_reshaped = U.reshape(n, m, rank)  # U[i, b, r]
        V_reshaped = V.reshape(m, p, rank)  # V[c, k, r]
        W_reshaped = W.reshape(p, n, rank)  # W[k, i, r]
        
        # T[i, b, c, k] = sum_r W[k,i,r] * U[i,b,r] * V[c,k,r]
        T = np.einsum('kir,ibr,ckr->ibck', W_reshaped, U_reshaped, V_reshaped)
        
        # Residuals: T[i,b,c,k] - delta(b,c)
        # delta(b,c) is independent of i,k
        delta = np.eye(m)
        residuals = T - np.tile(delta, (n, 1, 1, p)).transpose(0, 1, 2, 3)
        # Actually T has shape (n, m, m, p) and delta needs to be broadcast
        # T[i,b,c,k] should equal delta[b,c]
        res = np.zeros_like(T)
        for i in range(n):
            for k in range(p):
                res[i, :, :, k] = T[i, :, :, k] - delta
        return np.sum(res ** 2)
    
    # Build a structured initial guess using the block decomposition
    # 9 columns = 3 blocks of 3
    # For each block, use a 2x3x3 decomposition
    # If we find rank r_block for 2x3x3, total rank = 3 * r_block
    
    def build_block_solution(r_block):
        """Build a solution using block structure with rank r_block per block."""
        n, m, p = DIMENSIONS
        n_blocks = p // m  # 3 blocks
        total_rank = n_blocks * r_block
        
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        for blk in range(n_blocks):
            for rb in range(r_block):
                col = blk * r_block + rb
                # For block blk, columns blk*m to (blk+1)*m-1
                # U[i*m+b, col] = U_block[i, b, rb]
                # V[c*p + (blk*m+k), col] = V_block[c, k, rb] for k in 0..m-1
                # W[(blk*m+k)*n+i, col] = W_block[k, i, rb] for k in 0..m-1
                # We'll fill these later during optimization
                pass
        
        return U, V, W, total_rank
    
    # Try to find a good 2x3x3 decomposition first
    def find_233_decomposition(max_rank=12, max_time=30):
        """Find a low-rank decomposition for 2x3x3 matrix multiplication."""
        n2, m2, p2 = 2, 3, 3
        start = time.time()
        
        def obj_233(x, rank):
            U = x[:n2*m2*rank].reshape(n2*m2, rank)
            V = x[n2*m2*rank:n2*m2*rank+m2*p2*rank].reshape(m2*p2, rank)
            W = x[n2*m2*rank+m2*p2*rank:].reshape(p2*n2, rank)
            
            U_r = U.reshape(n2, m2, rank)
            V_r = V.reshape(m2, p2, rank)
            W_r = W.reshape(p2, n2, rank)
            
            T = np.einsum('kir,ibr,ckr->ibck', W_r, U_r, V_r)
            delta = np.eye(m2)
            res = np.zeros((n2, m2, m2, p2))
            for i in range(n2):
                for k in range(p2):
                    res[i, :, :, k] = T[i, :, :, k] - delta
            return np.sum(res ** 2)
        
        def check_233(U, V, W, tol=1e-6):
            n2, m2, p2 = 2, 3, 3
            rank = U.shape[1]
            for i in range(n2):
                for k in range(p2):
                    for b in range(m2):
                        for c in range(m2):
                            val = 0.0
                            for r in range(rank):
                                val += W[k*n2+i, r] * U[i*m2+b, r] * V[c*p2+k, r]
                            expected = 1.0 if b == c else 0.0
                            if abs(val - expected) > tol:
                                return False
            return True
        
        # Schoolbook for 2x3x3
        U_sb = np.zeros((6, 18))
        V_sb = np.zeros((9, 18))
        W_sb = np.zeros((6, 18))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U_sb[i*3+j, col] = 1
                    V_sb[j*3+k, col] = 1
                    W_sb[k*2+i, col] = 1
                    col += 1
        
        best_U, best_V, best_W, best_r = U_sb, V_sb, W_sb, 18
        
        for target_rank in range(7, 19):
            if time.time() - start > max_time:
                break
            n_params = (6 + 9 + 6) * target_rank
            
            for attempt in range(20):
                if time.time() - start > max_time:
                    break
                
                np.random.seed(attempt * 7919 + target_rank * 31)
                U_init = np.random.randn(6, target_rank) * 0.5
                V_init = np.random.randn(9, target_rank) * 0.5
                W_init = np.random.randn(6, target_rank) * 0.5
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                
                try:
                    from scipy.optimize import minimize
                    result = minimize(obj_233, x0, args=(target_rank,), method='L-BFGS-B',
                                      options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
                    
                    if result.fun < 1e-10:
                        x_opt = result.x
                        U_opt = x_opt[:6*target_rank].reshape(6, target_rank)
                        V_opt = x_opt[6*target_rank:15*target_rank].reshape(9, target_rank)
                        W_opt = x_opt[15*target_rank:].reshape(6, target_rank)
                        
                        U_h = to_half_int(U_opt)
                        V_h = to_half_int(V_opt)
                        W_h = to_half_int(W_opt)
                        
                        if check_233(U_h, V_h, W_h, tol=1e-6):
                            return U_h, V_h, W_h, target_rank
                        elif check_233(U_opt, V_opt, W_opt, tol=1e-6):
                            return U_opt, V_opt, W_opt, target_rank
                except Exception:
                    pass
        
        return best_U, best_V, best_W, best_r
    
    # Try to find a good 2x3x3 decomposition
    U_233, V_233, W_233, r_233 = find_233_decomposition(max_rank=12, max_time=25)
    
    # Lift to 2x3x9 using block structure
    def lift_233_to_239(U_233, V_233, W_233, r_233):
        """Lift a 2x3x3 decomposition to 2x3x9 using block structure."""
        n, m, p = DIMENSIONS
        n_blocks = p // m  # 3
        total_rank = n_blocks * r_233
        
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        for blk in range(n_blocks):
            for rb in range(r_233):
                col = blk * r_233 + rb
                # U[i*m+b, col] = U_233[i*m+b, rb]
                U[:, col] = U_233[:, rb]
                # V[c*p + (blk*m+k), col] = V_233[c*m+k, rb]
                for c in range(m):
                    for k in range(m):
                        V[c*p + blk*m + k, col] = V_233[c*m + k, rb]
                # W[(blk*m+k)*n+i, col] = W_233[k*n+i, rb]
                for k in range(m):
                    for i in range(n):
                        W[(blk*m + k)*n + i, col] = W_233[k*n + i, rb]
        
        return U, V, W, total_rank
    
    U_lifted, V_lifted, W_lifted, r_lifted = lift_233_to_239(U_233, V_233, W_233, r_233)
    
    # Verify the lifted solution
    if check_solution(U_lifted, V_lifted, W_lifted, tol=1e-6):
        if r_lifted < best_rank:
            best_rank = r_lifted
            U, V, W = U_lifted, V_lifted, W_lifted
    
    # Now try to optimize the full 2x3x9 problem from the lifted solution
    if best_rank < 54:
        target_rank = best_rank
        n_params = (n*m + m*p + p*n) * target_rank
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Try to further optimize
        from scipy.optimize import minimize
        
        for attempt in range(10):
            if time.time() > deadline - 15:
                break
            
            if attempt == 0:
                x_init = x0.copy()
            else:
                np.random.seed(attempt * 4441 + target_rank)
                noise = np.random.randn(n_params) * 0.1
                x_init = x0 + noise
            
            try:
                result = minimize(objective_simple, x_init, args=(target_rank,), method='L-BFGS-B',
                                  options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
                
                if result.fun < 1e-10:
                    x_opt = result.x
                    U_opt = x_opt[:n*m*target_rank].reshape(n*m, target_rank)
                    V_opt = x_opt[n*m*target_rank:n*m*target_rank+m*p*target_rank].reshape(m*p, target_rank)
                    W_opt = x_opt[n*m*target_rank+m*p*target_rank:].reshape(p*n, target_rank)
                    
                    U_h = to_half_int(U_opt)
                    V_h = to_half_int(V_opt)
                    W_h = to_half_int(W_opt)
                    
                    if check_solution(U_h, V_h, W_h, tol=1e-6):
                        break
                    elif check_solution(U_opt, V_opt, W_opt, tol=1e-6):
                        U, V, W = U_opt, V_opt, W_opt
                        break
            except Exception:
                pass
    
    # Try lower ranks with direct search
    lower_bound = max(n*p, m*p, n*m)  # max(18, 27, 6) = 27
    
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline - 10:
            break
        
        n_params = (n*m + m*p + p*n) * target_rank
        
        for attempt in range(15):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(attempt * 6151 + target_rank * 17)
            U_init = np.random.randn(n*m, target_rank) * 0.3
            V_init = np.random.randn(m*p, target_rank) * 0.3
            W_init = np.random.randn(p*n, target_rank) * 0.3
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = minimize(objective_simple, x0, args=(target_rank,), method='L-BFGS-B',
                                  options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10})
                
                if result.fun < 1e-10:
                    x_opt = result.x
                    U_opt = x_opt[:n*m*target_rank].reshape(n*m, target_rank)
                    V_opt = x_opt[n*m*target_rank:n*m*target_rank+m*p*target_rank].reshape(m*p, target_rank)
                    W_opt = x_opt[n*m*target_rank+m*p*target_rank:].reshape(p*n, target_rank)
                    
                    U_h = to_half_int(U_opt)
                    V_h = to_half_int(V_opt)
                    W_h = to_half_int(W_opt)
                    
                    if check_solution(U_h, V_h, W_h, tol=1e-6):
                        best_rank = target_rank
                        U, V, W = U_h, V_h, W_h
                        break
                    elif check_solution(U_opt, V_opt, W_opt, tol=1e-6):
                        best_rank = target_rank
                        U, V, W = U_opt, V_opt, W_opt
                        break
            except Exception:
                pass
    
    # Final validation
    if not check_solution(U, V, W, tol=1e-6):
        # Fall back to schoolbook
        U, V, W, best_rank = schoolbook()
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
