"""
Matrix multiplication 3,3,6 tensor decomposition.
Strategy: Exploit the structure that 3x6 = two 3x3 blocks.
Try to find rank < 54 by sharing factors between the two 3x3 multiplications.
"""
import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n, m, p = 3, 3, 6
    dim_u = n * m  # 9
    dim_v = m * p  # 18
    dim_w = p * n  # 18
    
    # Build target tensor
    T_target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    def verify(U, V, W, tol=1e-8):
        try:
            T_recon = np.einsum('at,bt,ct->abc', U, V, W)
            return np.allclose(T_recon, T_target, atol=tol, rtol=0)
        except Exception:
            return False
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, r_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb.copy(), V_sb.copy(), W_sb.copy(), r_sb
    
    def try_random_optimization(target_rank, num_trials=30, max_iter=150):
        """Try random initialization + gradient descent for rank reduction."""
        from scipy.optimize import minimize
        
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(12345 + trial * 7)
            
            U = np.random.randn(dim_u, target_rank) * 0.5
            V = np.random.randn(dim_v, target_rank) * 0.5
            W = np.random.randn(dim_w, target_rank) * 0.5
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                Ux = x[:dim_u*target_rank].reshape(dim_u, target_rank)
                Vx = x[dim_u*target_rank : dim_u*target_rank + dim_v*target_rank].reshape(dim_v, target_rank)
                Wx = x[dim_u*target_rank + dim_v*target_rank : ].reshape(dim_w, target_rank)
                T_recon = np.einsum('at,bt,ct->abc', Ux, Vx, Wx)
                error = T_target - T_recon
                return np.sum(error**2)
            
            def grad(x):
                Ux = x[:dim_u*target_rank].reshape(dim_u, target_rank)
                Vx = x[dim_u*target_rank : dim_u*target_rank + dim_v*target_rank].reshape(dim_v, target_rank)
                Wx = x[dim_u*target_rank + dim_v*target_rank : ].reshape(dim_w, target_rank)
                T_recon = np.einsum('at,bt,ct->abc', Ux, Vx, Wx)
                error = T_target - T_recon
                dU = 2 * np.einsum('abc,bt,ct->at', error, Vx, Wx)
                dV = 2 * np.einsum('abc,at,ct->bt', error, Ux, Wx)
                dW = 2 * np.einsum('abc,at,bt->ct', error, Ux, Vx)
                return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
            
            try:
                res = minimize(
                    objective, x0, jac=grad, method='L-BFGS-B',
                    options={'maxiter': max_iter, 'gtol': 1e-14, 'maxfun': 5000}
                )
            except Exception:
                continue
            
            if time.time() > deadline - 2:
                break
                
            U_f = res.x[:dim_u*target_rank].reshape(dim_u, target_rank)
            V_f = res.x[dim_u*target_rank : dim_u*target_rank + dim_v*target_rank].reshape(dim_v, target_rank)
            W_f = res.x[dim_u*target_rank + dim_v*target_rank : ].reshape(dim_w, target_rank)
            
            if verify(U_f, V_f, W_f, tol=1e-7):
                # Try to snap to rational values
                for denom in [1, 2, 4]:
                    U_s = np.round(U_f * denom) / denom
                    V_s = np.round(V_f * denom) / denom
                    W_s = np.round(W_f * denom) / denom
                    if verify(U_s, V_s, W_s, tol=1e-6):
                        return U_s, V_s, W_s, target_rank
                return U_f, V_f, W_f, target_rank
        
        return None
    
    def try_structured_decomposition(target_rank):
        """
        Exploit the structure: 3x6 matrix = [B1 | B2] where B1, B2 are 3x3.
        C = A * [B1 | B2] = [A*B1 | A*B2]
        
        Try to find a shared factorization where some rank terms are shared
        between the two 3x3 multiplications.
        
        Approach: Assume rank-r decomposition where:
        - Some terms contribute to both C[:, 0:3] and C[:, 3:6]
        - Some terms contribute only to one side
        
        Parameterize: 
        U: (9, r), V: (18, r), W: (18, r)
        
        For the V matrix, rows 0-8 correspond to j=0, k=0,1,2 (first 3x3 block)
        rows 9-17 correspond to j=0, k=3,4,5 (second 3x3 block)
        
        Similarly for W: rows 0-5 for k=0, i=0,1,2; rows 6-11 for k=1; rows 12-17 for k=2
        Wait, W index is k*n+i, so:
        k=0: i=0,1,2 -> rows 0,1,2
        k=1: i=0,1,2 -> rows 3,4,5
        k=2: i=0,1,2 -> rows 6,7,8
        k=3: i=0,1,2 -> rows 9,10,11
        k=4: i=0,1,2 -> rows 12,13,14
        k=5: i=0,1,2 -> rows 15,16,17
        
        For V: index is j*p+k, so:
        j=0: k=0,1,2,3,4,5 -> rows 0,1,2,3,4,5
        j=1: k=0,1,2,3,4,5 -> rows 6,7,8,9,10,11
        j=2: k=0,1,2,3,4,5 -> rows 12,13,14,15,16,17
        """
        if time.time() > deadline - 10:
            return None
        
        from scipy.optimize import minimize
        
        num_trials = 15
        
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(999 + trial * 11)
            
            U = np.random.randn(dim_u, target_rank) * 0.3
            V = np.random.randn(dim_v, target_rank) * 0.3
            W = np.random.randn(dim_w, target_rank) * 0.3
            
            # Add structure: encourage some terms to be "shared"
            # by making some V entries symmetric between first and second half
            for t in range(target_rank // 2):
                # Make V rows for k=0,1,2 similar to k=3,4,5 for some terms
                for j in range(m):
                    V[j*p + 0, t] = V[j*p + 3, t]
                    V[j*p + 1, t] = V[j*p + 4, t]
                    V[j*p + 2, t] = V[j*p + 5, t]
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                Ux = x[:dim_u*target_rank].reshape(dim_u, target_rank)
                Vx = x[dim_u*target_rank : dim_u*target_rank + dim_v*target_rank].reshape(dim_v, target_rank)
                Wx = x[dim_u*target_rank + dim_v*target_rank : ].reshape(dim_w, target_rank)
                T_recon = np.einsum('at,bt,ct->abc', Ux, Vx, Wx)
                error = T_target - T_recon
                return np.sum(error**2)
            
            def grad(x):
                Ux = x[:dim_u*target_rank].reshape(dim_u, target_rank)
                Vx = x[dim_u*target_rank : dim_u*target_rank + dim_v*target_rank].reshape(dim_v, target_rank)
                Wx = x[dim_u*target_rank + dim_v*target_rank : ].reshape(dim_w, target_rank)
                T_recon = np.einsum('at,bt,ct->abc', Ux, Vx, Wx)
                error = T_target - T_recon
                dU = 2 * np.einsum('abc,bt,ct->at', error, Vx, Wx)
                dV = 2 * np.einsum('abc,at,ct->bt', error, Ux, Wx)
                dW = 2 * np.einsum('abc,at,bt->ct', error, Ux, Vx)
                return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
            
            try:
                res = minimize(
                    objective, x0, jac=grad, method='L-BFGS-B',
                    options={'maxiter': 200, 'gtol': 1e-14, 'maxfun': 8000}
                )
            except Exception:
                continue
            
            if time.time() > deadline - 2:
                break
                
            U_f = res.x[:dim_u*target_rank].reshape(dim_u, target_rank)
            V_f = res.x[dim_u*target_rank : dim_u*target_rank + dim_v*target_rank].reshape(dim_v, target_rank)
            W_f = res.x[dim_u*target_rank + dim_v*target_rank : ].reshape(dim_w, target_rank)
            
            if verify(U_f, V_f, W_f, tol=1e-7):
                for denom in [1, 2, 4, 8]:
                    U_s = np.round(U_f * denom) / denom
                    V_s = np.round(V_f * denom) / denom
                    W_s = np.round(W_f * denom) / denom
                    if verify(U_s, V_s, W_s, tol=1e-6):
                        return U_s, V_s, W_s, target_rank
                return U_f, V_f, W_f, target_rank
        
        return None
    
    # Try various ranks in increasing order
    # Known: rank(3x3x3) = 23, so rank(3x3x6) >= 23
    # Upper bound: 54 (schoolbook)
    # Try: 46, 48, 50, 52 first (structured), then random
    
    ranks_to_try = [46, 48, 50, 52]
    
    for r in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        # First try structured
        res = try_structured_decomposition(r)
        if res is not None:
            best_U, best_V, best_W, best_rank = res
            break
        
        # Then try random
        if time.time() > deadline - 10:
            break
        res = try_random_optimization(r, num_trials=10, max_iter=100)
        if res is not None:
            best_U, best_V, best_W, best_rank = res
            break
    
    # If still at schoolbook rank, try a few more random attempts at lower ranks
    if best_rank == 54 and time.time() < deadline - 20:
        for r in [44, 42, 40]:
            if time.time() > deadline - 10:
                break
            res = try_random_optimization(r, num_trials=5, max_iter=80)
            if res is not None:
                best_U, best_V, best_W, best_rank = res
                break
    
    # Convert to lists with rational approximation
    def to_list(mat):
        res = []
        for row in mat:
            r = []
            for v in row:
                if not np.isfinite(v):
                    r.append(0.0)
                    continue
                try:
                    f = Fraction(v).limit_denominator(100)
                    if abs(f - v) < 1e-5:
                        if f.denominator == 1:
                            r.append(int(f.numerator))
                        else:
                            r.append(f.numerator / f.denominator)
                    else:
                        r.append(float(v))
                except (ValueError, OverflowError):
                    r.append(float(v))
            res.append(r)
        return res
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
