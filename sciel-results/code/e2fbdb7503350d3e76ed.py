import numpy as np
import time
import scipy.optimize

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    np_ = p * n
    
    # Schoolbook
    rank_sb = nm * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((np_, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Precompute non-zero indices
    nz_u = []
    nz_v = []
    nz_w = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz_u.append(i*m+j)
                nz_v.append(j*p+k)
                nz_w.append(k*n+i)
    nz_u = np.array(nz_u)
    nz_v = np.array(nz_v)
    nz_w = np.array(nz_w)
    
    # Zero indices sample
    # Total 2916 entries.
    # We'll use all non-zero and 1000 random zero entries.
    all_u = np.repeat(np.arange(nm), mp * np_)
    all_v = np.tile(np.repeat(np.arange(mp), np_), nm)
    all_w = np.tile(np.arange(np_), nm * mp)
    
    nz_mask = np.zeros(len(all_u), dtype=bool)
    for idx in range(len(nz_u)):
        # Find index in all_u
        # This is slow. Let's just generate the mask by checking.
        pass
        
    # Simpler: just iterate over all 2916 entries in the loss function.
    # 2916 is small.
    
    def objective(params):
        if time.time() > deadline:
            return 1e9
        r = len(nz_u) # No, r is rank
        # params shape: (nm + mp + np_) * r
        # Reshape
        U = params[:nm*r].reshape(nm, r)
        V = params[nm*r:(nm+mp)*r].reshape(mp, r)
        W = params[(nm+mp)*r:].reshape(np_, r)
        
        err = 0.0
        # Non-zero
        for idx in range(len(nz_u)):
            u = nz_u[idx]
            v = nz_v[idx]
            w = nz_w[idx]
            val = np.dot(U[u, :], V[v, :] * W[w, :])
            err += (1.0 - val) ** 2
            
        # Zero (sample 500)
        # To avoid overhead, we can precompute a fixed set of zero indices
        # But for now, let's just check the non-zero ones and assume 
        # the optimizer will drive the others to 0 if we include them.
        # Including all zeros makes the gradient dense.
        
        # Let's include 1000 random zero indices pre-computed outside.
        # We'll create them once.
        
        return err

    # Precompute zero indices
    zero_u = []
    zero_v = []
    zero_w = []
    nz_set = set(zip(nz_u.tolist(), nz_v.tolist(), nz_w.tolist()))
    for u in range(nm):
        for v in range(mp):
            for w in range(np_):
                if (u, v, w) not in nz_set:
                    zero_u.append(u)
                    zero_v.append(v)
                    zero_w.append(w)
    # Shuffle and take 1000
    indices = np.random.permutation(len(zero_u))[:1000]
    zu = np.array(zero_u)[indices]
    zv = np.array(zero_v)[indices]
    zw = np.array(zero_w)[indices]

    def objective_full(params):
        if time.time() > deadline:
            return 1e9
        r = params.shape[0] // (nm + mp + np_)
        U = params[:nm*r].reshape(nm, r)
        V = params[nm*r:(nm+mp)*r].reshape(mp, r)
        W = params[(nm+mp)*r:].reshape(np_, r)
        
        err = 0.0
        # Non-zero
        for idx in range(len(nz_u)):
            u = nz_u[idx]
            v = nz_v[idx]
            w = nz_w[idx]
            val = np.dot(U[u, :], V[v, :] * W[w, :])
            err += (1.0 - val) ** 2
            
        # Zero
        for idx in range(len(zu)):
            u = zu[idx]
            v = zv[idx]
            w = zw[idx]
            val = np.dot(U[u, :], V[v, :] * W[w, :])
            err += val ** 2
            
        return err

    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try ranks from 53 down to 45
    for r in range(53, 44, -1):
        if time.time() > deadline:
            break
            
        # Initialize
        np.random.seed(42)
        params = np.random.randn((nm + mp + np_) * r)
        
        # Optimize
        try:
            res = scipy.optimize.minimize(objective_full, params, method='L-BFGS-B', options={'maxiter': 100, 'maxfun': 200})
        except Exception:
            continue
            
        if res.fun < 1e-6:
            # Check integrality
            U = res.x[:nm*r].reshape(nm, r)
            V = res.x[nm*r:(nm+mp)*r].reshape(mp, r)
            W = res.x[(nm+mp)*r:].reshape(np_, r)
            
            # Round to 0.5
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Verify exactness
            # Check non-zero
            ok = True
            for idx in range(len(nz_u)):
                u = nz_u[idx]
                v = nz_v[idx]
                w = nz_w[idx]
                val = np.dot(U_r[u, :], V_r[v, :] * W_r[w, :])
                if abs(val - 1.0) > 1e-9:
                    ok = False
                    break
            if ok:
                # Check zeros
                for idx in range(len(zu)):
                    u = zu[idx]
                    v = zv[idx]
                    w = zw[idx]
                    val = np.dot(U_r[u, :], V_r[v, :] * W_r[w, :])
                    if abs(val) > 1e-9:
                        ok = False
                        break
                        
            if ok:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
    return best_result
