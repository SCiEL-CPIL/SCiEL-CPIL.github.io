    def find_rank_3x3x2():
        """Search for low-rank decomposition of 3x3x2 tensor."""
        n2, m2, p2 = 3, 3, 2
        nm2 = n2 * m2 # 9
        mp2 = m2 * p2 # 6
        pn2 = p2 * n2 # 6
        
        T2 = np.zeros((nm2, mp2, pn2))
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    T2[i*m2+j, j*p2+k, k*n2+i] = 1.0
        T2_flat = T2.reshape(nm2 * mp2, pn2)
        
        best_r2 = 17
        best_U2, best_V2, best_W2 = None, None, None
        
        # Try ranks from 16 down to 10
        for r in range(16, 9, -1):
            if time.time() > deadline - 20:
                break
            
            # Use integer/half-integer random initializations
            for attempt in range(200):
                if time.time() > deadline - 15:
                    break
                
                np.random.seed(12345 + r * 100 + attempt)
                
                # Random dense initialization with integers/half-integers
                # Values in {-2, -1, -0.5, 0, 0.5, 1, 2}
                U2 = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(nm2, r))
                V2 = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(mp2, r))
                
                K2 = np.zeros((nm2 * mp2, r))
                for c in range(r):
                    K2[:, c] = np.outer(U2[:, c], V2[:, c]).flatten()
                
                try:
                    W2_sol, residuals, rank_K, sv = np.linalg.lstsq(K2, T2_flat, rcond=1e-12)
                except np.linalg.LinAlgError:
                    continue
                
                residual = np.abs(K2 @ W2_sol - T2_flat).max()
                
                if residual < 1e-9:
                    # Check if W2_sol can be represented with integers/half-integers
                    W2_check = W2_sol * 2
                    if np.allclose(W2_check, np.round(W2_check), atol=1e-6):
                        W2_final = np.round(W2_check) / 2
                        
                        # Verify
                        recon2 = np.zeros((nm2, mp2, pn2))
                        for c in range(r):
                            recon2 += np.einsum('a,b,c->abc', U2[:,c], V2[:,c], W2_final[:,c])
                        if np.abs(recon2 - T2).max() < 1e-8:
                            best_r2 = r
                            best_U2 = U2.copy()
                            best_V2 = V2.copy()
                            best_W2 = W2_final.copy()
                            break
            
            if best_r2 < 17:
                break
        
        if best_U2 is not None:
            return best_U2, best_V2, best_W2, best_r2
        return None, None, None, 17
