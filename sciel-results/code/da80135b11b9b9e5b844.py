import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (16, rank)  
    # W: (p*n, rank) = (8, rank)
    
    def time_up():
        return time.time() > deadline
    
    def verify(U, V, W):
        """Verify that U, V, W decompose the matrix multiplication tensor."""
        n, m, p = 2, 4, 4
        rank = U.shape[1]
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True
    
    def verify_fast(U, V, W):
        """Faster verification using numpy."""
        n, m, p = 2, 4, 4
        rank = U.shape[1]
        # Reshape for vectorized computation
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Create the full tensor and check
        T = np.zeros((n, m, p))
        for r in range(rank):
            u_r = U[:, r]  # shape (8,)
            v_r = V[:, r]  # shape (16,)
            w_r = W[:, r]  # shape (8,)
            # u_r indexed by i*m+j, v_r by j*p+k, w_r by k*n+i
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
        return np.allclose(T, 1.0, atol=1e-8)
    
    def residual_vec(params, rank):
        """Flatten U, V, W into params and return residual vector."""
        n, m, p = 2, 4, 4
        nm = n * m  # 8
        mp = m * p  # 16
        pn = p * n  # 8
        total = nm + mp + pn  # 32
        # params has shape (rank * total,)
        
        U = params[0:rank*nm].reshape(rank, nm).T  # (nm, rank)
        V = params[rank*nm:rank*nm+rank*mp].reshape(rank, mp).T  # (mp, rank)
        W = params[rank*nm+rank*mp:rank*nm+rank*mp+rank*pn].reshape(rank, pn).T  # (pn, rank)
        
        # Compute T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T = np.zeros((n, m, p))
        for r in range(rank):
            u_r = U[:, r]
            v_r = V[:, r]
            w_r = W[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
        
        return (T - 1.0).flatten()
    
    def try_rank(rank, n_trials=50, seed_base=42):
        """Try to find a rank-r decomposition using least_squares."""
        n, m, p = 2, 4, 4
        nm, mp, pn = n*m, m*p, p*n
        total = nm + mp + pn
        
        np.random.seed(seed_base)
        
        for trial in range(n_trials):
            if time_up():
                break
            
            # Initialize with Strassen-like structure plus random perturbation
            params = np.random.randn(rank * total) * 0.5
            
            try:
                result = least_squares(
                    residual_vec, 
                    params, 
                    args=(rank,),
                    method='trf',
                    max_nfev=2000,
                    xtol=1e-12,
                    ftol=1e-12
                )
            except Exception:
                continue
            
            if result.success or result.cost < 1e-10:
                # Extract U, V, W
                U = result.x[0:rank*nm].reshape(rank, nm).T
                V = result.x[rank*nm:rank*nm+rank*mp].reshape(rank, mp).T
                W = result.x[rank*nm+rank*mp:rank*nm+rank*mp+rank*pn].reshape(rank, pn).T
                
                if verify(U, V, W):
                    return U, V, W
            elif result.cost < 1e-6:
                # Close but not exact - try to clean up
                U = result.x[0:rank*nm].reshape(rank, nm).T
                V = result.x[rank*nm:rank*nm+rank*mp].reshape(rank, mp).T
                W = result.x[rank*nm+rank*mp:rank*nm+rank*mp+rank*pn].reshape(rank, pn).T
                if verify(U, V, W):
                    return U, V, W
        
        return None
    
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1
        
        return U, V, W
    
    def build_strassen_28():
        """Build the rank-28 Strassen-based decomposition."""
        U_222, V_222, W_222 = strassen_2x2x2()
        
        U_full = np.zeros((8, 28))
        V_full = np.zeros((16, 28))
        W_full = np.zeros((8, 28))
        
        # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        U_full[0:4, 0:7] = U_222
        V_full[0:4, 0:7] = V_222
        W_full[0:4, 0:7] = W_222
        
        # Block 2: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
        U_full[4:8, 7:14] = U_222
        V_full[8:12, 7:14] = V_222
        W_full[0:4, 7:14] = W_222
        
        # Block 3: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
        U_full[0:4, 14:21] = U_222
        V_full[4:8, 14:21] = V_222
        W_full[4:8, 14:21] = W_222
        
        # Block 4: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
        U_full[4:8, 21:28] = U_222
        V_full[12:16, 21:28] = V_222
        W_full[4:8, 21:28] = W_222
        
        return U_full, V_full, W_full
    
    # First, get the baseline rank-28 solution
    U28, V28, W28 = build_strassen_28()
    if not verify(U28, V28, W28):
        # Fallback to schoolbook
        rank = 32
        U_fb = np.zeros((8, rank))
        V_fb = np.zeros((16, rank))
        W_fb = np.zeros((8, rank))
        col = 0
        for i in range(2):
            for j in range(4):
                for k in range(4):
                    U_fb[i*4+j, col] = 1
                    V_fb[j*4+k, col] = 1
                    W_fb[k*2+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    best_rank = 28
    best_result = (U28, V28, W28)
    
    # Try to find lower rank decompositions
    for target_rank in [24, 25, 26, 27]:
        if time_up():
            break
        
        result = try_rank(target_rank, n_trials=30, seed_base=target_rank * 100)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break  # Found the lowest, stop
    
    # If we didn't find lower than 28, also try a smarter search for 24
    if best_rank == 28 and not time_up():
        # Try with Strassen-based initialization
        n, m, p = 2, 4, 4
        nm, mp, pn = 8, 16, 8
        total = nm + mp + pn
        
        for target_rank in [24, 25]:
            if time_up():
                break
            
            np.random.seed(target_rank * 777)
            for trial in range(20):
                if time_up():
                    break
                
                # Initialize from Strassen, then truncate and perturb
                params28 = np.concatenate([
                    U28.flatten(), V28.flatten(), W28.flatten()
                ])
                
                # Take first target_rank worth of parameters
                if target_rank * total < len(params28):
                    params = params28[:target_rank * total].copy()
                else:
                    params = np.zeros(target_rank * total)
                
                # Add small random perturbation
                params += np.random.randn(len(params)) * 0.1
                
                try:
                    result = least_squares(
                        residual_vec,
                        params,
                        args=(target_rank,),
                        method='trf',
                        max_nfev=3000,
                        xtol=1e-12,
                        ftol=1e-12
                    )
                except Exception:
                    continue
                
                if result.cost < 1e-10:
                    U = result.x[0:target_rank*nm].reshape(target_rank, nm).T
                    V = result.x[target_rank*nm:target_rank*nm+target_rank*mp].reshape(target_rank, mp).T
                    W = result.x[target_rank*nm+target_rank*mp:target_rank*nm+target_rank*mp+target_rank*pn].reshape(target_rank, pn).T
                    
                    if verify(U, V, W):
                        best_rank = target_rank
                        best_result = (U, V, W)
                        break
            if best_rank < 28:
                break
    
    U_best, V_best, W_best = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
