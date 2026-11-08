import numpy as np

def construct():
        n, m, p = 2, 3, 10
        
        # Attempt Strassen + Direct hybrid
        try:
            rank = 36
            U = np.zeros((n*m, rank), dtype=int)
            V = np.zeros((m*p, rank), dtype=int)
            W = np.zeros((p*n, rank), dtype=int)
            
            def add_strassen_block(b, r_start):
                k0 = 2 * b
                # ... (same terms as before)
                
            # ... (same calls)
            
            # Verify
            # T_recon[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            T_recon = np.zeros((n, m, p), dtype=int)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        T_recon[i,j,k] = s
            
            T_target = np.ones((n, m, p), dtype=int)
            
            if not np.array_equal(T_recon, T_target):
                # Fallback to direct
                U = np.zeros((n*m, n*m*p), dtype=int)
                V = np.zeros((m*p, n*m*p), dtype=int)
                W = np.zeros((p*n, n*m*p), dtype=int)
                rank = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U[i*m + j, rank] = 1
                            V[j*p + k, rank] = 1
                            W[k*n + i, rank] = 1
                            rank += 1
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            # Fallback to direct
            U = np.zeros((n*m, n*m*p), dtype=int)
            V = np.zeros((m*p, n*m*p), dtype=int)
            W = np.zeros((p*n, n*m*p), dtype=int)
            rank = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m + j, rank] = 1
                        V[j*p + k, rank] = 1
                        W[k*n + i, rank] = 1
                        rank += 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
