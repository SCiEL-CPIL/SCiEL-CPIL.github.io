import scipy
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s safety margin
    
    n, m, p = 3, 3, 6
    
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Construct target tensor T_target
    # T_target[a, b, c] = 1 if a,b,c form a valid multiplication term (i,j,k)
    # a = i*m + j, b = j*p + k, c = k*n + i
    T_target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def snap_and_verify(U, V, W):
        # Snap to nearest 0.5
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        # Check exactness
        R = np.einsum('ir,jr,kr->ijk', U_s, V_s, W_s)
        # Check if R matches T_target exactly
        if np.allclose(R, T_target, atol=1e-9):
            # Convert to lists
            U_list = U_s.tolist()
            V_list = V_s.tolist()
            W_list = W_s.tolist()
            return True, U_list, V_list, W_list
        return False, None, None, None

    def get_schoolbook(rank):
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    # Try ranks from 46 upwards
    # We know schoolbook is 54.
    # Let's try to find a better one.
    # The known lower bound is 47 (for 3x3x3 it's 23, but this is 3x3x6).
    # Actually, for 3x3x6, the rank is likely around 48-50.
    # Let's start from 46 and go up.
    
    # Use a more robust optimization strategy.
    # 1. Random init
    # 2. L-BFGS-B
    # 3. Snap and verify
    
    # We will iterate ranks from 46 to 53.
    # If we find a valid decomposition, return it.
    # If not, return schoolbook (54).
    
    for rank in range(46, 54):
        if time.time() > deadline - 30: break
        
        # Try multiple random initializations
        # More attempts for lower ranks as they are harder.
        num_attempts = 10 if rank < 48 else 5
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 15: break
            
            # Initialize
            # Use random init with small variance
            np.random.seed(42 + rank * 100 + attempt) # Deterministic for reproducibility in tests, but random in production?
            # Actually, better to use random seed to explore different basins.
            # But for reproducibility of the "search", maybe fixed seed is better?
            # The task says "score is the rank". It doesn't require determinism.
            # Let's use random seed.
            
            U = np.random.randn(dim_U, rank) * 0.5
            V = np.random.randn(dim_V, rank) * 0.5
            W = np.random.randn(dim_W, rank) * 0.5
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def obj(params):
                U_p = params[:dim_U*rank].reshape(dim_U, rank)
                V_p = params[dim_U*rank : dim_U*rank + dim_V*rank].reshape(dim_V, rank)
                W_p = params[dim_U*rank + dim_V*rank :].reshape(dim_W, rank)
                R = np.einsum('ir,jr,kr->ijk', U_p, V_p, W_p)
                return np.sum((R - T_target)**2)
            
            # L-BFGS-B
            try:
                # Use tighter tolerance and more iterations
                res = scipy.optimize.minimize(
                    obj, params, method='L-BFGS-B', 
                    options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                U_opt = res.x[:dim_U*rank].reshape(dim_U, rank)
                V_opt = res.x[dim_U*rank : dim_U*rank + dim_V*rank].reshape(dim_V, rank)
                W_opt = res.x[dim_U*rank + dim_V*rank :].reshape(dim_W, rank)
                
                success, U_s, V_s, W_s = snap_and_verify(U_opt, V_opt, W_opt)
                if success:
                    return {"U": U_s, "V": V_s, "W": W_s}
                    
            except Exception:
                pass

    # Fallback to schoolbook
    U, V, W = get_schoolbook(54)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
