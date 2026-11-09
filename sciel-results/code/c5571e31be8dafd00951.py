import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # 5-second safety margin
    
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
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_rec, T, atol=tol)
    
    def round_to_half_integer(x):
        """Round to nearest integer or half-integer."""
        return np.round(x * 2) / 2.0
    
    def structured_init(r, seed):
        """Generate structured initializations based on tensor structure."""
        np.random.seed(seed)
        
        # Strategy 1: Random
        x0 = np.random.randn(r * (nm + mp + pn)) * 0.5
        
        # Strategy 2: Block-structured
        # Try to initialize with patterns that respect the (i,j,k) structure
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        
        # Assign random signs and scales to encourage non-zero entries
        for col in range(r):
            # Randomly pick which (i,j,k) combinations this rank term covers
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            
            sign = np.random.choice([-1, 1])
            scale = np.random.choice([0.5, 1.0])
            
            U[i * m + j, col] = sign * scale
            V[j * p + k, col] = sign * scale
            W[k * n + i, col] = sign * scale
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        return x0
    
    def optimize_rank(r, max_restarts=15):
        """
        Try to find a rank-r decomposition using optimization.
        """
        if time.time() > deadline:
            return None
        
        dim = r * (nm + mp + pn)
        
        def unpack(x):
            U = x[:r*nm].reshape(nm, r)
            V = x[r*nm:r*(nm+mp)].reshape(mp, r)
            W = x[r*(nm+mp):].reshape(pn, r)
            return U, V, W
        
        def loss_and_grad(x):
            U, V, W = unpack(x)
            
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            error = T_rec - T
            loss = np.sum(error**2)
            
            grad_U = np.einsum('abc,br,cr->ar', error, V, W)
            grad_V = np.einsum('abc,ar,cr->br', error, U, W)
            grad_W = np.einsum('abc,ar,br->cr', error, U, V)
            
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            return loss, grad
        
        best_result = None
        best_loss = float('inf')
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Alternate between random and structured initialization
            if restart % 3 == 0:
                x0 = structured_init(r, 42 + restart * 13 + r)
            elif restart % 3 == 1:
                np.random.seed(42 + restart * 13 + r)
                x0 = np.random.randn(dim) * 0.3
            else:
                np.random.seed(42 + restart * 13 + r + 100)
                x0 = np.random.randn(dim) * 0.7
            
            try:
                res = minimize(loss_and_grad, x0, method='L-BFGS-B', jac=True, 
                               options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
            except Exception:
                continue
            
            if res.fun < best_loss:
                U, V, W = unpack(res.x)
                
                # Round to nearest half-integer
                U_rounded = round_to_half_integer(U)
                V_rounded = round_to_half_integer(V)
                W_rounded = round_to_half_integer(W)
                
                # Check if rounded solution is exact
                if check_decomposition(U_rounded, V_rounded, W_rounded):
                    return (U_rounded, V_rounded, W_rounded)
                
                # Local discrete search
                U_cur = U_rounded.copy()
                V_cur = V_rounded.copy()
                W_cur = W_rounded.copy()
                
                T_rec_cur = np.einsum('ar,br,cr->abc', U_cur, V_cur, W_cur)
                err_cur = np.sum((T_rec_cur - T)**2)
                
                # Try local moves
                for step in range(20):
                    if time.time() > deadline:
                        break
                    mat_choice = np.random.randint(3)
                    if mat_choice == 0:
                        idx = np.random.randint(U_cur.size)
                        row, col = np.unravel_index(idx, U_cur.shape)
                        for delta in [-0.5, 0.5]:
                            orig_val = U_cur[row, col]
                            U_cur[row, col] = orig_val + delta
                            T_rec_test = np.einsum('ar,br,cr->abc', U_cur, V_cur, W_cur)
                            err_test = np.sum((T_rec_test - T)**2)
                            if err_test < err_cur - 1e-12:
                                err_cur = err_test
                                if check_decomposition(U_cur, V_cur, W_cur):
                                    return (U_cur, V_cur, W_cur)
                            else:
                                U_cur[row, col] = orig_val
                    elif mat_choice == 1:
                        idx = np.random.randint(V_cur.size)
                        row, col = np.unravel_index(idx, V_cur.shape)
                        for delta in [-0.5, 0.5]:
                            orig_val = V_cur[row, col]
                            V_cur[row, col] = orig_val + delta
                            T_rec_test = np.einsum('ar,br,cr->abc', U_cur, V_cur, W_cur)
                            err_test = np.sum((T_rec_test - T)**2)
                            if err_test < err_cur - 1e-12:
                                err_cur = err_test
                                if check_decomposition(U_cur, V_cur, W_cur):
                                    return (U_cur, V_cur, W_cur)
                            else:
                                V_cur[row, col] = orig_val
                    else:
                        idx = np.random.randint(W_cur.size)
                        row, col = np.unravel_index(idx, W_cur.shape)
                        for delta in [-0.5, 0.5]:
                            orig_val = W_cur[row, col]
                            W_cur[row, col] = orig_val + delta
                            T_rec_test = np.einsum('ar,br,cr->abc', U_cur, V_cur, W_cur)
                            err_test = np.sum((T_rec_test - T)**2)
                            if err_test < err_cur - 1e-12:
                                err_cur = err_test
                                if check_decomposition(U_cur, V_cur, W_cur):
                                    return (U_cur, V_cur, W_cur)
                            else:
                                W_cur[row, col] = orig_val
                                
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_result = (U_rounded, V_rounded, W_rounded)
        
        return None
    
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # Try ranks from 62 down to 50
    ranks_to_try = [62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        result = optimize_rank(r, max_restarts=10)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = (U, V, W)
            # Continue to lower ranks if time permits
    
    U, V, W = best_result
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
