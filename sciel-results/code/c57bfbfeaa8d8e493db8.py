import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin
    
    n, m, p = 2, 5, 5
    schoolbook_rank = n * m * p # 50
    
    # Build the target tensor T
    # T[x, y, z] where x=i*m+j, y=j*p+k, z=k*n+i
    # T is 1 if indices are consistent, else 0.
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                x = i * m + j
                y = j * p + k
                z = k * n + i
                T[x, y, z] = 1.0
                
    def verify(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Check exactness
        R = U.shape[1]
        # Use einsum for speed
        # Result[x, y, z] = sum_t U[x,t] V[y,t] W[z,t]
        # This is a tensor contraction.
        # For small R, we can do it in a loop or use einsum.
        # np.einsum('xt,yt,zt->xyz', U, V, W)
        try:
            res = np.einsum('xt,yt,zt->xyz', U, V, W)
            return np.allclose(res, T, atol=1e-9)
        except:
            return False
            
    def snap_to_half_int(arr):
        # Snap to nearest half-integer
        return np.round(arr * 2) / 2.0
        
    # Try to find a rank 49 solution
    # We use a simple local search on the real values, then snap.
    # If snapping fails, we try to adjust the snapped values to satisfy the equations.
    
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Generate schoolbook solution as fallback
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
                
    # Search for R=49
    R = 49
    # We will try a few random initializations and use gradient descent or simple iteration.
    # Given the time, a robust gradient descent is preferred.
    
    # We can use a "tensor train" or similar? No, just simple optimization.
    # Let's try to solve the linear system if we fix two matrices?
    # No, it's cubic.
    
    # Let's try a randomized search with "snapping" and "local repair".
    # 1. Random init from {-1, 0, 1}
    # 2. Compute residual
    # 3. If residual is small, snap and verify.
    # 4. If not, try to improve by changing one entry at a time.
    
    def random_search(R, max_iters=5000):
        U = np.random.randint(-1, 2, (n*m, R)) * 0.5
        V = np.random.randint(-1, 2, (m*p, R)) * 0.5
        W = np.random.randint(-1, 2, (p*n, R)) * 0.5
        
        # Local search
        current_error = np.sum((T - np.einsum('xt,yt,zt->xyz', U, V, W))**2)
        
        for i in range(max_iters):
            if time.time() > deadline:
                return None
            # Pick a random matrix and entry
            mat_idx = np.random.randint(3)
            if mat_idx == 0:
                idx = np.random.randint(n*m)
                t = np.random.randint(R)
                old_val = U[idx, t]
                # Try nearby values
                for delta in [-0.5, 0.5]:
                    new_val = old_val + delta
                    U[idx, t] = new_val
                    err = np.sum((T - np.einsum('xt,yt,zt->xyz', U, V, W))**2)
                    if err < current_error:
                        current_error = err
                        break
                    else:
                        U[idx, t] = old_val
            elif mat_idx == 1:
                idx = np.random.randint(m*p)
                t = np.random.randint(R)
                old_val = V[idx, t]
                for delta in [-0.5, 0.5]:
                    new_val = old_val + delta
                    V[idx, t] = new_val
                    err = np.sum((T - np.einsum('xt,yt,zt->xyz', U, V, W))**2)
                    if err < current_error:
                        current_error = err
                        break
                    else:
                        V[idx, t] = old_val
            else:
                idx = np.random.randint(p*n)
                t = np.random.randint(R)
                old_val = W[idx, t]
                for delta in [-0.5, 0.5]:
                    new_val = old_val + delta
                    W[idx, t] = new_val
                    err = np.sum((T - np.einsum('xt,yt,zt->xyz', U, V, W))**2)
                    if err < current_error:
                        current_error = err
                        break
                    else:
                        W[idx, t] = old_val
                        
            # Check for exact solution
            if current_error < 1e-9:
                return U, V, W
                
        # If not exact, try snapping the best found
        if current_error < 1.0: # Threshold
            U_s = snap_to_half_int(U)
            V_s = snap_to_half_int(V)
            W_s = snap_to_half_int(W)
            if verify(U_s, V_s, W_s):
                return U_s, V_s, W_s
        return None

    # Try R=49
    for attempt in range(5):
        if time.time() > deadline:
            break
        res = random_search(R, max_iters=1000)
        if res:
            U, V, W = res
            best_rank = R
            best_U, best_V, best_W = U, V, W
            break
            
    if best_rank < schoolbook_rank:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
