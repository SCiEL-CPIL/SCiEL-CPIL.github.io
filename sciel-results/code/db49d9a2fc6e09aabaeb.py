import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = 150.0 - 10.0  # 10-second safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T as a 3D array
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
        
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W form a valid rank-r decomposition."""
        r = U.shape[1]
        T_rec = np.zeros((nm, mp, pn))
        for r_idx in range(r):
            T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        return np.allclose(T_rec, T, atol=tol)
    
    def try_scipy_optimization(r, max_iters=100, tol=1e-10):
        """Try to find a rank-r decomposition using scipy L-BFGS-B."""
        if time.time() > deadline:
            return None
            
        # Flatten parameters
        # U: nm x r, V: mp x r, W: pn x r
        # Total params: r * (nm + mp + pn)
        
        def unpack(params):
            U = params[:r*nm].reshape(nm, r)
            V = params[r*nm:r*(nm+mp)].reshape(mp, r)
            W = params[r*(nm+mp):].reshape(pn, r)
            return U, V, W
            
        def objective(params):
            U, V, W = unpack(params)
            T_rec = np.zeros((nm, mp, pn))
            for r_idx in range(r):
                T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
            error = T_rec - T
            return np.sum(error**2)
            
        def gradient(params):
            U, V, W = unpack(params)
            T_rec = np.zeros((nm, mp, pn))
            for r_idx in range(r):
                T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
            error = T_rec - T
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r_idx in range(r):
                grad_U[:, r_idx] = np.einsum('abc,b,c->a', error, V[:, r_idx], W[:, r_idx])
                grad_V[:, r_idx] = np.einsum('abc,a,c->b', error, U[:, r_idx], W[:, r_idx])
                grad_W[:, r_idx] = np.einsum('abc,a,b->c', error, U[:, r_idx], V[:, r_idx])
                
            # Scale gradients by 2 for sum of squares
            grad_U *= 2
            grad_V *= 2
            grad_W *= 2
            
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
        # Initialize with random sparse structure
        np.random.seed(42)
        # Start with small random values
        init_U = np.random.randn(nm, r) * 0.1
        init_V = np.random.randn(mp, r) * 0.1
        init_W = np.random.randn(pn, r) * 0.1
        
        init_params = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
        
        try:
            res = minimize(objective, init_params, jac=gradient, method='L-BFGS-B', 
                           options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-8})
        except Exception:
            return None
            
        if res.success or res.fun < 1e-8:
            U, V, W = unpack(res.x)
            # Try to round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if check_decomposition(U_rounded, V_rounded, W_rounded):
                return (U_rounded, V_rounded, W_rounded)
                
            # Check if continuous solution is valid (though task asks for int/half-int)
            # If not, we might fail. But usually if error is tiny, rounding works.
            if check_decomposition(U, V, W):
                 # If continuous works but rounded doesn't, we can't return it directly
                 # unless we accept float. But task says "integer/half-integer entries".
                 # So we must return rounded if it works.
                 pass
                 
        return None

    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # Try ranks
    # 63 is schoolbook.
    # Try to beat 63.
    ranks_to_try = [62, 60, 58, 56, 54, 52, 50]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        result = try_scipy_optimization(r, max_iters=50)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = (U, V, W)
            break
            
    U, V, W = best_result
    
    # Convert to lists of lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
