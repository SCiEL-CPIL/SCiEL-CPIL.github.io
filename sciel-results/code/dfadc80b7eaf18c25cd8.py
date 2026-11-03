    def try_rank_reduction(target_rank, n_restarts=10, seed=0):
        if time.time() > deadline:
            return None
        
        rng = np.random.RandomState(seed)
        best_err = float('inf')
        best_solution = None
        R = target_rank

        def objective_and_grad(params):
            U = params[:n*m*R].reshape(n*m, R)
            V = params[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
            W = params[n*m*R+m*p*R:].reshape(p*n, R)
            recon = np.einsum('at,bt,ct->abc', U, V, W)
            F = recon - T
            err = np.sum(F * F)
            gU = np.einsum('abc,bt,ct->at', F, V, W)
            gV = np.einsum('abc,at,ct->bt', F, U, W)
            gW = np.einsum('abc,at,bt->ct', F, U, V)
            grad = np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
            return err, grad

        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            if target_rank <= 28:
                indices = rng.choice(28, target_rank, replace=False)
                U = U28[:, indices].copy()
                V = V28[:, indices].copy()
                W = W28[:, indices].copy()
                U += rng.randn(n * m, target_rank) * 0.1
                V += rng.randn(m * p, target_rank) * 0.1
                W += rng.randn(p * n, target_rank) * 0.1
            else:
                U = rng.randn(n * m, target_rank)
                V = rng.randn(m * p, target_rank)
                W = rng.randn(p * n, target_rank)
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = opt.minimize(objective_and_grad, params, jac=True, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            except:
                continue
            
            if res.fun < best_err:
                params_opt = res.x
                U_opt = params_opt[:n*m*R].reshape(n*m, R)
                V_opt = params_opt[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                W_opt = params_opt[n*m*R+m*p*R:].reshape(p*n, R)
                
                recon = np.einsum('at,bt,ct->abc', U_opt, V_opt, W_opt)
                err = np.sum((recon - T)**2)
                
                if err < best_err:
                    best_err = err
                    best_solution = (U_opt, V_opt, W_opt)
                    
                if best_err < 1e-10:
                    break
        
        if best_err > 1e-6:
            return None
        
        Uf, Vf, Wf = best_solution
        
        Us = np.round(Uf * 2) / 2
        Vs = np.round(Vf * 2) / 2
        Ws = np.round(Wf * 2) / 2
        if verify(Us, Vs, Ws, tol=1e-8):
            return Us, Vs, Ws, target_rank
        
        Ui = np.round(Uf)
        Vi = np.round(Vf)
        Wi = np.round(Wf)
        if verify(Ui, Vi, Wi, tol=1e-8):
            return Ui, Vi, Wi, target_rank
        
        Uq = np.round(Uf * 4) / 4
        Vq = np.round(Vf * 4) / 4
        Wq = np.round(Wf * 4) / 4
        if verify(Uq, Vq, Wq, tol=1e-8):
            return Uq, Vq, Wq, target_rank
        
        return None
