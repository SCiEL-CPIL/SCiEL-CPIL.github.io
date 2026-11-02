import numpy as np
import time
import random

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5-second safety margin before 120s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Schoolbook decomposition for 3x3 x 3x6 (rank 54)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Fast verification and residual using NumPy
    def compute_residual(U, V, W):
        """
        U: (nm, R)
        V: (mp, R)
        W: (pn, R)
        """
        R = U.shape[1]
        # Reshape to tensors
        # U[i*m+j, r] -> U_t[i, j, r]
        U_t = U.reshape(n, m, R)
        # V[j*p+k, r] -> V_t[j, k, r]
        V_t = V.reshape(m, p, R)
        # W[k*n+i, r] -> W_t[k, i, r]
        W_t = W.reshape(p, n, R)
        
        # Compute C[a, b, c, d, e, f] = sum_r U[a, b, r] V[c, d, r] W[e, f, r]
        # a=i2, b=j, c=j2, d=k2, e=k, f=i
        C = np.einsum('abr,cdr,efr->abcdef', U_t, V_t, W_t)
        
        # Construct Target T
        # T[a, b, c, d, e, f] = 1 if a==f, b==c, d==e else 0
        T = np.zeros((n, m, m, p, p, n))
        for a in range(n):
            for b in range(m):
                for d in range(p):
                    # c=b, e=d, f=a
                    T[a, b, b, d, d, a] = 1.0
                    
        diff = C - T
        return np.sum(diff**2), C, T

    # Local search with exact line search for one coordinate
    def local_search(U, V, W, max_iters=5000):
        """
        Perform coordinate descent with exact line search.
        """
        n, m, p = 3, 3, 6
        R = U.shape[1]
        
        # Precompute target T
        T = np.zeros((n, m, m, p, p, n))
        for a in range(n):
            for b in range(m):
                for d in range(p):
                    T[a, b, b, d, d, a] = 1.0
                    
        # Compute initial C and residual
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        C = np.einsum('abr,cdr,efr->abcdef', U_t, V_t, W_t)
        curr_res = np.sum((C - T)**2)
        
        best_res = curr_res
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Choose random matrix and entry
            mat_id = random.randint(0, 2)
            
            if mat_id == 0: # Update U
                a = random.randint(0, n - 1)
                b = random.randint(0, m - 1)
                r = random.randint(0, R - 1)
                
                # Affected slice: C[a, b, :, :, :, :]
                # Change in C is Delta * V[c, d, r] * W[e, f, r]
                # y[c, d, e, f] = V_t[c, d, r] * W_t[e, f, r]
                y = np.einsum('cr,fr->cf', V_t[:, :, r], W_t[:, :, r]) 
                # Wait, V_t shape (m, p, R). V_t[:, :, r] is (m, p).
                # W_t shape (p, n, R). W_t[:, :, r] is (p, n).
                # y should be broadcast to (m, p, p, n) i.e. (c, d, e, f)
                # V[c, d, r] * W[e, f, r]
                y = V_t[:, :, r][:, :, np.newaxis, np.newaxis] * W_t[:, :, r][np.newaxis, np.newaxis, :, :]
                
                x = C[a, b, :, :, :, :] - T[a, b, :, :, :, :]
                
                # Minimize sum (x + Delta * y)^2
                # Delta = - sum(x*y) / sum(y*y)
                num = np.sum(x * y)
                den = np.sum(y * y)
                
                if den < 1e-12:
                    continue
                    
                delta = -num / den
                
                # Update
                U[a * m + b, r] += delta
                U_t[a, b, r] += delta
                C[a, b, :, :, :, :] += delta * y
                
                new_res = curr_res + 2 * delta * num + delta * delta * den
                # Due to floating point, recompute if close
                if abs(new_res - curr_res) < 1e-6:
                     new_res = np.sum((C - T)**2)
                     
                curr_res = new_res
                
                if curr_res < best_res:
                    best_res = curr_res
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    
            elif mat_id == 1: # Update V
                c = random.randint(0, m - 1)
                d = random.randint(0, p - 1)
                r = random.randint(0, R - 1)
                
                # Affected slice: C[:, :, c, d, :, :]
                # Change in C is Delta * U[a, b, r] * W[e, f, r]
                # y[a, b, e, f] = U_t[a, b, r] * W_t[e, f, r]
                y = U_t[:, :, r][:, :, np.newaxis, np.newaxis] * W_t[:, :, r][np.newaxis, np.newaxis, :, :]
                
                x = C[:, :, c, d, :, :] - T[:, :, c, d, :, :]
                
                num = np.sum(x * y)
                den = np.sum(y * y)
                
                if den < 1e-12:
                    continue
                
                delta = -num / den
                
                V[c * p + d, r] += delta
                V_t[c, d, r] += delta
                C[:, :, c, d, :, :] += delta * y
                
                new_res = curr_res + 2 * delta * num + delta * delta * den
                if abs(new_res - curr_res) < 1e-6:
                     new_res = np.sum((C - T)**2)
                curr_res = new_res
                
                if curr_res < best_res:
                    best_res = curr_res
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    
            else: # Update W
                e = random.randint(0, p - 1)
                f = random.randint(0, n - 1)
                r = random.randint(0, R - 1)
                
                # Affected slice: C[:, :, :, :, e, f]
                # Change in C is Delta * U[a, b, r] * V[c, d, r]
                # y[a, b, c, d] = U_t[a, b, r] * V_t[c, d, r]
                y = U_t[:, :, r][:, :, np.newaxis, np.newaxis] * V_t[:, :, r][np.newaxis, np.newaxis, :, :]
                
                x = C[:, :, :, :, e, f] - T[:, :, :, :, e, f]
                
                num = np.sum(x * y)
                den = np.sum(y * y)
                
                if den < 1e-12:
                    continue
                
                delta = -num / delta if den > 0 else 0 # Typo in logic, fix below
                delta = -num / den
                
                W[e * n + f, r] += delta
                W_t[e, f, r] += delta
                C[:, :, :, :, e, f] += delta * y
                
                new_res = curr_res + 2 * delta * num + delta * delta * den
                if abs(new_res - curr_res) < 1e-6:
                     new_res = np.sum((C - T)**2)
                curr_res = new_res
                
                if curr_res < best_res:
                    best_res = curr_res
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
            
            if best_res < 1e-10:
                break
                
        return best_U, best_V, best_W, best_res

    # Main search loop
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try ranks below schoolbook
    for target_rank in range(best_rank - 1, 30, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Fewer restarts for lower ranks as they are harder
        restarts = 5 if target_rank < 40 else 10
        
        for _ in range(restarts):
            if time.time() > deadline - 5:
                break
                
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            U, V, W, res = local_search(U, V, W, max_iters=2000)
            
            if res < 1e-8:
                # Verify
                res_final, _, _ = compute_residual(U, V, W)
                if res_final < 1e-8:
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            # Try to go lower? The loop continues to next lower rank
            pass
    
    if time.time() > deadline - 2:
        return best_result
        
    return best_result
