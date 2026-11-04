    def search_rank(r, max_iters=5000):
        if time.time() > deadline - 10:
            return None
            
        np.random.seed(42)
        
        U = np.zeros((n*m, r), dtype=int)
        V = np.zeros((m*p, r), dtype=int)
        W = np.zeros((p*n, r), dtype=int)
        
        for col in range(r):
            U[:, col] = np.random.choice([0, 1], size=n*m, p=[0.8, 0.2])
            V[:, col] = np.random.choice([0, 1], size=m*p, p=[0.8, 0.2])
            W[:, col] = np.random.choice([0, 1], size=p*n, p=[0.8, 0.2])
            
        T = np.zeros((n, m, p), dtype=float)
        for col in range(r):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col]
                        
        error = np.sum((T - 1.0)**2)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            if error < 1e-9:
                if verify_exact(U.tolist(), V.tolist(), W.tolist()):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

            col = np.random.randint(r)
            which = np.random.randint(3)
            
            old_error = error
            
            if which == 0:
                idx = np.random.randint(n*m)
                i, j = idx // m, idx % m
                old_val = U[idx, col]
                U[idx, col] = 1 - old_val
                delta = (1 - old_val) - old_val
                
                # Save old T slice
                old_T_slice = T[i, j, :].copy()
                
                for k in range(p):
                    T[i, j, k] += delta * V[j*p+k, col] * W[k*n+i, col]
                    
                new_error = error - np.sum((old_T_slice - 1.0)**2) + np.sum((T[i, j, :] - 1.0)**2)
                
                if new_error < error:
                    error = new_error
                else:
                    U[idx, col] = old_val
                    T[i, j, :] = old_T_slice
                    
            elif which == 1:
                idx = np.random.randint(m*p)
                j, k = idx // p, idx % p
                old_val = V[idx, col]
                V[idx, col] = 1 - old_val
                delta = (1 - old_val) - old_val
                
                # T[i, j, k] for all i
                # Save old T[:, j, k]
                old_T_slice = T[:, j, k].copy()
                
                for i in range(n):
                    T[i, j, k] += U[i*m+j, col] * delta * W[k*n+i, col]
                    
                new_error = error - np.sum((old_T_slice - 1.0)**2) + np.sum((T[:, j, k] - 1.0)**2)
                
                if new_error < error:
                    error = new_error
                else:
                    V[idx, col] = old_val
                    T[:, j, k] = old_T_slice
                    
            else:
                idx = np.random.randint(p*n)
                k, i = idx // n, idx % n
                old_val = W[idx, col]
                W[idx, col] = 1 - old_val
                delta = (1 - old_val) - old_val
                
                # T[i, j, k] for all j
                # Save old T[i, :, k]
                old_T_slice = T[i, :, k].copy()
                
                for j in range(m):
                    T[i, j, k] += U[i*m+j, col] * V[j*p+k, col] * delta
                    
                new_error = error - np.sum((old_T_slice - 1.0)**2) + np.sum((T[i, :, k] - 1.0)**2)
                
                if new_error < error:
                    error = new_error
                else:
                    W[idx, col] = old_val
                    T[i, :, k] = old_T_slice

        return None
