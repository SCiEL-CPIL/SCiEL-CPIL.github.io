import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 130s to be safe (150s limit)
    
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, R), V (m*p, R), W (p*n, R)
    # n=3, m=3, p=4
    # U: 9 x R
    # V: 12 x R
    # W: 12 x R
    
    # Construct target tensor T (dense)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def rationalize_matrix(M, tol=1e-5):
        # M is a numpy array
        # Check if entries are half-integers (2*M is integer)
        # Returns M * 2 rounded to nearest integer if valid, else None
        M_scaled = 2.0 * M
        M_int = np.round(M_scaled)
        if not np.allclose(M_scaled, M_int, atol=tol):
            return None
        return M_int

    def verify_and_convert(U, V, W, T, tol=1e-5):
        # U, V, W are numpy arrays of floats
        # Returns dict of lists if valid, else None
        U_int = rationalize_matrix(U, tol)
        V_int = rationalize_matrix(V, tol)
        W_int = rationalize_matrix(W, tol)
        
        if U_int is None or V_int is None or W_int is None:
            return None
            
        # Compute tensor product of integer scaled matrices
        # T_calc[a,b,c] = sum_r U_int[a,r] V_int[b,r] W_int[c,r]
        T_calc = np.einsum('ar,br,cr->abc', U_int, V_int, W_int, optimize=True)
        T_target = 8.0 * T
        
        # Check if T_calc matches T_target
        if not np.allclose(T_calc, T_target, atol=1e-6):
            return None
            
        # Construct output
        U_out = (U_int / 2.0).tolist()
        V_out = (V_int / 2.0).tolist()
        W_out = (W_int / 2.0).tolist()
        
        return {"U": U_out, "V": V_out, "W": W_out}

    def compute_residual(U, V, W, T):
        """Compute residual norm efficiently using einsum."""
        T_calc = np.zeros_like(T)
        R = U.shape[1]
        for r in range(R):
            T_calc += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.linalg.norm(T_calc - T)

    def als_decompose(T, R, max_iters=80, tol=1e-10, init_type='random'):
        """ALS decomposition with vectorized updates."""
        n, m, p = DIMENSIONS
        nm, mp, pn = n*m, m*p, p*n
        
        if init_type == 'random':
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(pn, R)
        elif init_type == 'structured':
            # Use structured initialization based on the tensor structure
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            W = np.zeros((pn, R))
            # Initialize with partial schoolbook patterns
            for r in range(R):
                # Randomly assign some indices
                i = r % n
                j = (r // n) % m
                k = (r // (n*m)) % p
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
            # Add some random perturbation
            U += 0.1 * np.random.randn(nm, R)
            V += 0.1 * np.random.randn(mp, R)
            W += 0.1 * np.random.randn(pn, R)
        else:
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(pn, R)
        
        prev_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
                
            # Update U: for each a, solve least squares
            # A[a, (b,c), r] = V[b,r] * W[c,r]
            # We can vectorize: for each a, build matrix A_a of shape (mp*pn, R)
            for a in range(nm):
                # Build A matrix: shape (mp*pn, R)
                # A[idx, r] = V[b, r] * W[c, r] where idx = b*pn + c
                A = np.einsum('br,cr->bc r', V, W).reshape(mp*pn, R)
                y = T[a].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                U[a, :] = sol
            
            # Update V: for each b, solve least squares
            for b in range(mp):
                A = np.einsum('ar,cr->ac r', U, W).reshape(nm*pn, R)
                y = T[:, b, :].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                V[b, :] = sol
            
            # Update W: for each c, solve least squares
            for c in range(pn):
                A = np.einsum('ar,br->ab r', U, V).reshape(nm*mp, R)
                y = T[:, :, c].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                W[c, :] = sol
            
            # Check convergence
            err = compute_residual(U, V, W, T)
            if err < tol:
                return U, V, W, err
            if abs(prev_err - err) < 1e-12 and it > 10:
                return U, V, W, err
            prev_err = err
                
        return U, V, W, err

    def gradient_polish(U, V, W, T, max_iters=200, tol=1e-12):
        """Gradient descent polish for exact decomposition."""
        n, m, p = DIMENSIONS
        nm, mp, pn = n*m, m*p, p*n
        R = U.shape[1]
        
        lr = 0.01
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Compute gradients
            # dL/dU[a,r] = sum_{b,c} (T_calc[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
            # dL/dV[b,r] = sum_{a,c} (T_calc[a,b,c] - T[a,b,c]) * U[a,r] * W[c,r]
            # dL/dW[c,r] = sum_{a,b} (T_calc[a,b,c] - T[a,b,c]) * U[a,r] * V[b,r]
            
            T_calc = np.zeros((nm, mp, pn))
            for r in range(R):
                T_calc += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            diff = T_calc - T
            
            # Gradient for U
            grad_U = np.einsum('abc,br,cr->ar', diff, V, W)
            # Gradient for V
            grad_V = np.einsum('abc,ar,cr->br', diff, U, W)
            # Gradient for W
            grad_W = np.einsum('abc,ar,br->cr', diff, U, V)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            err = np.linalg.norm(diff)
            if err < tol:
                break
            
            # Adaptive learning rate
            if it % 20 == 0:
                lr = max(0.001, lr * 0.9)
        
        final_err = compute_residual(U, V, W, T)
        return U, V, W, final_err

    def try_rational_and_verify(U, V, W, T, tol=1e-5):
        """Try to rationalize and verify."""
        return verify_and_convert(U, V, W, T, tol=tol)

    # Search for minimal R
    best_result = None
    
    for R in range(18, 37):
        if time.time() > deadline - 20:
            break
        
        # Try different initialization strategies and restarts
        init_types = ['random', 'structured', 'random', 'structured', 'random']
        
        for restart in range(len(init_types)):
            if time.time() > deadline - 15:
                break
            
            init_type = init_types[restart]
            U, V, W, err = als_decompose(T, R, max_iters=50, tol=1e-10, init_type=init_type)
            
            if err < 1e-6:
                # Try gradient polish
                U, V, W, err = gradient_polish(U, V, W, T, max_iters=100, tol=1e-12)
                
                if err < 1e-8:
                    result = try_rational_and_verify(U, V, W, T, tol=1e-5)
                    if result:
                        return result
                    # Try with tighter tolerance
                    result = try_rational_and_verify(U, V, W, T, tol=1e-6)
                    if result:
                        return result
                    # Try with even tighter tolerance
                    result = try_rational_and_verify(U, V, W, T, tol=1e-7)
                    if result:
                        return result
        
        # If we get close but can't rationalize, try more restarts
        if R <= 20 and time.time() < deadline - 30:
            for extra_restart in range(5):
                if time.time() > deadline - 20:
                    break
                U, V, W, err = als_decompose(T, R, max_iters=80, tol=1e-10, init_type='random')
                if err < 1e-8:
                    U, V, W, err = gradient_polish(U, V, W, T, max_iters=150, tol=1e-12)
                    if err < 1e-9:
                        result = try_rational_and_verify(U, V, W, T, tol=1e-6)
                        if result:
                            return result
    
    # Fallback to schoolbook
    return schoolbook()
