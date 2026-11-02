import scipy.optimize as opt

def _try_rank(T, n, m, p, r):
    dim_u, dim_v, dim_w = T.shape
    T_flat = T.flatten()
    
    def objective(params):
        U = params[:dim_u*r].reshape(dim_u, r)
        V = params[dim_u*r:dim_u*r+dim_v*r].reshape(dim_v, r)
        W = params[dim_u*r+dim_v*r:].reshape(dim_w, r)
        S = np.einsum('ur,vr,wr->uvw', U, V, W)
        return (T_flat - S.flatten())
        
    # Initial guesses
    for _ in range(10):
        if time.time() > time.time() + 5: # crude check
             break
        x0 = np.random.randn(dim_u*r + dim_v*r + dim_w*r) * 0.1
        # Use least_squares
        try:
            res = opt.least_squares(objective, x0, method='lm') # Levenberg-Marquardt
            if res.cost < 1e-10:
                # Extract U, V, W
                U = res.x[:dim_u*r].reshape(dim_u, r)
                V = res.x[dim_u*r:dim_u*r+dim_v*r].reshape(dim_v, r)
                W = res.x[dim_u*r+dim_v*r:].reshape(dim_w, r)
                
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify
                S_check = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
                if np.allclose(S_check, T, atol=1e-9):
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
        except Exception:
            pass
            
    return None
