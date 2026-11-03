"""Matrix multiplication 2x3x3 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

def _build_target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _tensor_from_factors(U, V, W, n, m, p):
    """Compute sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] flattened to (n*m*p,).
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    Target T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    """
    T = np.zeros(n * m * p, dtype=np.float64)
    for c in range(U.shape[1]):
        uc = U[:, c]  # (n*m,)
        vc = V[:, c]  # (m*p,)
        wc = W[:, c]  # (p*n,)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m * p + j * p + k] += uc[i * m + j] * vc[j * p + k] * wc[k * n + i]
    return T

def _is_close(T_computed, T_target, tol=1e-6):
    return np.allclose(T_computed, T_target, atol=tol, rtol=1e-6)

def _try_rationalize(U, V, W, max_denom=100):
    """Try to find small rational approximations."""
    from fractions import Fraction
    U_rat = [[Fraction(0) for _ in range(U.shape[1])] for _ in range(U.shape[0])]
    V_rat = [[Fraction(0) for _ in range(V.shape[1])] for _ in range(V.shape[0])]
    W_rat = [[Fraction(0) for _ in range(W.shape[1])] for _ in range(W.shape[0])]
    
    for i in range(U.shape[0]):
        for c in range(U.shape[1]):
            val = U[i, c]
            if abs(val) < 1e-9:
                U_rat[i][c] = Fraction(0)
            else:
                best = Fraction(0, 1)
                best_err = abs(val)
                for d in range(1, max_denom + 1):
                    num = round(val * d)
                    f = Fraction(num, d)
                    err = abs(val - float(f))
                    if err < best_err:
                        best_err = err
                        best = f
                    if best_err < 1e-10:
                        break
                U_rat[i][c] = best
    
    for i in range(V.shape[0]):
        for c in range(V.shape[1]):
            val = V[i, c]
            if abs(val) < 1e-9:
                V_rat[i][c] = Fraction(0)
            else:
                best = Fraction(0, 1)
                best_err = abs(val)
                for d in range(1, max_denom + 1):
                    num = round(val * d)
                    f = Fraction(num, d)
                    err = abs(val - float(f))
                    if err < best_err:
                        best_err = err
                        best = f
                    if best_err < 1e-10:
                        break
                V_rat[i][c] = best
    
    for i in range(W.shape[0]):
        for c in range(W.shape[1]):
            val = W[i, c]
            if abs(val) < 1e-9:
                W_rat[i][c] = Fraction(0)
            else:
                best = Fraction(0, 1)
                best_err = abs(val)
                for d in range(1, max_denom + 1):
                    num = round(val * d)
                    f = Fraction(num, d)
                    err = abs(val - float(f))
                    if err < best_err:
                        best_err = err
                        best = f
                    if best_err < 1e-10:
                        break
                W_rat[i][c] = best
    
    # Verify with rationals
    T_check = np.zeros(n * m * p, dtype=np.float64)
    n, m, p = DIMENSIONS
    for c in range(U.shape[1]):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_check[i * m * p + j * p + k] += float(U_rat[i * m + j][c]) * float(V_rat[j * p + k][c]) * float(W_rat[k * n + i][c])
    
    if _is_close(T_check, _build_target_tensor(), tol=1e-8):
        return U_rat, V_rat, W_rat
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    T_target = _build_target_tensor().flatten()
    
    deadline = time.time() + 140  # 10s safety margin
    
    # Try rank 17 with multiple random initializations
    rng = np.random.RandomState(42)
    
    for trial in range(2000):
        if time.time() > deadline - 10:
            break
        
        rank = 17
        # Random initialization
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        # Simple iterative refinement (like alternating least squares for tensor)
        # We want: T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
        # For fixed V, W: each row of U can be solved linearly
        # This is a multilinear system - use simple gradient-free approach
        
        # Use a simple multiplicative update approach
        best_err = np.inf
        U_best, V_best, W_best = U.copy(), V.copy(), W.copy()
        
        for it in range(50):
            if time.time() > deadline - 15:
                break
            
            # Update U: for each row (i*m+j), solve linear system
            # T[i*m+j, j*p+k] * W[k*n+i] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i]
            # This is coupled, so use simple projection approach
            
            # Compute current tensor
            T_cur = _tensor_from_factors(U, V, W, n, m, p)
            err = np.linalg.norm(T_cur - T_target)
            if err < best_err:
                best_err = err
                U_best, V_best, W_best = U.copy(), V.copy(), W.copy()
            
            if err < 1e-7:
                break
            
            # Simple gradient step for tensor factorization
            # dT/dU[i*m+j,c] = V[j*p+k,c] * W[k*n+i,c]
            # Use simple fixed-point: adjust U, V, W to reduce error
            
            # For each factor, project onto the residual
            # U_new[i*m+j,c] = U[i*m+j,c] + lr * sum_{j,k} residual[i*m+j,j*p+k] * V[j*p+k,c] * W[k*n+i,c]
            lr = 0.1
            for c in range(rank):
                # Update U column c
                for i in range(n):
                    for j in range(m):
                        grad = 0.0
                        for k in range(p):
                            idx = i * m * p + j * p + k
                            grad += (T_target[idx] - T_cur[idx]) * V[j * p + k, c] * W[k * n + i, c]
                        U[i * m + j, c] += lr * grad
                
                # Update V column c
                for j in range(m):
                    for k in range(p):
                        grad = 0.0
                        for i in range(n):
                            idx = i * m * p + j * p + k
                            grad += (T_target[idx] - T_cur[idx]) * U[i * m + j, c] * W[k * n + i, c]
                        V[j * p + k, c] += lr * grad
                
                # Update W column c
                for k in range(p):
                    for i in range(n):
                        grad = 0.0
                        for j in range(m):
                            idx = i * m * p + j * p + k
                            grad += (T_target[idx] - T_cur[idx]) * U[i * m + j, c] * V[j * p + k, c]
                        W[k * n + i, c] += lr * grad
        
        # Check if we found a solution
        T_final = _tensor_from_factors(U_best, V_best, W_best, n, m, p)
        if np.allclose(T_final, T_target, atol=1e-6, rtol=1e-6):
            # Try to rationalize
            result = _try_rationalize(U_best, V_best, W_best)
            if result is not None:
                U_rat, V_rat, W_rat = result
                U_list = [[float(x) for x in row] for row in U_rat]
                V_list = [[float(x) for x in row] for row in V_rat]
                W_list = [[float(x) for x in row] for row in W_rat]
                # Convert to int if all are integers
                U_out = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U_list]
                V_out = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V_list]
                W_out = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W_list]
                return {"U": U_out, "V": V_out, "W": W_out}
            else:
                # Return float version
                return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
    
    # Fallback: schoolbook rank 18
    return _schoolbook()
