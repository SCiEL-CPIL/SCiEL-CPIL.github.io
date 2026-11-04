    def try_rank_optimized(r):
        time_limit = 10.0
        start = time.time()
        
        for seed in range(20):
            if time.time() - start > time_limit:
                break
            np.random.seed(100 + seed)
            
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Normalize
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            for step in range(100):
                # ALS steps
                # ... (same as before but using vectorized loss for check)
                
                # Check loss
                loss = compute_loss(U, V, W)
                if loss < 1e-10:
                    # Try rounding
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        
            # Also check if current U,V,W are close to half-integers?
            # No, we need exact.
            
        return None
    