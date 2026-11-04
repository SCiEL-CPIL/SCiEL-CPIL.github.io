"""Matrix multiplication 2x3 by 3x3 - rank-minimal tensor decomposition."""
import time
import numpy as np
from numpy.linalg import svd

DIMENSIONS = [2, 3, 3]

def construct():
    start_time = time.time()
    timeout = 140.0  # safety margin under 150s limit

    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 9
    pn = p * n  # 6

    # Build the multiplication tensor T with T[i*mp + j*p + k] = 1 if corresponding
    # Actually: T[(i,j,k)] = 1 for all i,j,k. Reshape to tensor shape (n,m,p).
    T = np.ones((n, m, p))

    # The rank-1 decomposition condition:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  for all i,j,k
    # where c ranges over rank
    # Equivalently, for each c, define the rank-1 tensor T_c[i,j,k] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]

    def verify(U, V, W, tol=1e-10):
        """Check if U, V, W give exact decomposition. Returns True if valid."""
        # For each c, outer product of U_c, V_c, W_c
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def verify_fast(U, V, W, tol=1e-10):
        """Vectorized verification."""
        n, m, p = DIMENSIONS
        U_np = np.array(U, dtype=float) if not isinstance(U, np.ndarray) else U
        V_np = np.array(V, dtype=float) if not isinstance(V, np.ndarray) else V
        W_np = np.array(W, dtype=float) if not isinstance(W, np.ndarray) else W
        rank = U_np.shape[1]
        T_recon = np.zeros((n, m, p))
        for c in range(rank):
            uc = U_np[:, c]  # nm
            vc = V_np[:, c]  # mp
            wc = W_np[:, c]  # pn
            T_recon += np.einsum('ab,bc,ca->abc', uc.reshape(n, m, 1), vc.reshape(1, m, p), wc.reshape(p, 1, n).squeeze())
            # Actually let me redo: U[i*m+j] * V[j*p+k] * W[k*n+i]
        # Let me use direct approach
        T_recon = np.zeros((n, m, p))
        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += U_np[i*m + j, c] * V_np[j*p + k, c] * W_np[k*n + i, c]
        return np.allclose(T_recon, np.ones((n, m, p)), atol=tol)

    # Strategy: Start from rank-18 schoolbook, remove one term, optimize
    # Rank-17 is known to exist for mu(2,3,3).
    
    # Better approach: parameterize and use least-squares / gradient descent
    # For rank r, we need to find U(nm,r), V(mp,r), W(pn,r) such that
    # sum_c U[:,c] * V[:,c] * W[:,c] (with appropriate index mapping) = T
    
    # Let me use a different formulation. The tensor can be viewed as:
    # For fixed i, the slice T_i is an m x p matrix.
    # T_i[j,k] = sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c])
    # Define for each i and c: W_i[c] = W[0*n+i, c], W_i[k*n+i, c] varies with k
    # This is tricky because W depends on both k and i.
    
    # Let me use the standard tensor approach:
    # Flatten T to a 3D array and use the identity:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # I'll use scipy's optimize or manual gradient descent.
    from scipy.optimize import minimize
    
    def objective(params, n, m, p):
        """params: flat array of length rank*(nm + mp + pn)"""
        rank = (len(params)) // (nm + mp + pn)
        U = params[0:nm*rank].reshape(nm, rank)
        V = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
        W = params[nm*rank+mp*rank:].reshape(pn, rank)
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (s - 1.0)**2
        return err
    
    def objective_grad(params, n, m, p):
        rank = len(params) // (nm + mp + pn)
        U = params[0:nm*rank].reshape(nm, rank)
        V = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
        W = params[nm*rank+mp*rank:].reshape(pn, rank)
        
        # Compute gradient numerically? No, let's do analytical.
        # Actually, let me just compute value and use L-BFGS-B which needs gradient
        # For simplicity, use finite differences with a wrapper
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (s - 1.0)**2
        return err

    def numerical_grad(params, n, m, p, eps=1e-7):
        """Compute numerical gradient."""
        grad = np.zeros_like(params)
        f0 = objective(params, n, m, p)
        for idx in range(len(params)):
            params2 = params.copy()
            params2[idx] += eps
            f1 = objective(params2, n, m, p)
            params2[idx] -= 2*eps
            f2 = objective(params2, n, m, p)
            params2[idx] += eps  # restore
            grad[idx] = (f1 - f2) / (2*eps)
        return grad

    # This numerical gradient will be too slow. Let me use a better approach.
    
    # Alternative: Use the fact that for fixed V and W, U is determined linearly, etc.
    # Or use alternating least squares.
    
    # ALS approach:
    # Fix V, W -> solve for U (linear system)
    # Fix U, W -> solve for V (linear system)
    # Fix U, V -> solve for W (linear system)
    
    def als_step(rank, n, m, p, max_iters=500, tol=1e-12):
        """Alternating least squares to find rank-r decomposition."""
        # Initialize randomly or from schoolbook
        rng = np.random.default_rng(42)
        U = rng.normal(size=(nm, rank))
        V = rng.normal(size=(mp, rank))
        W = rng.normal(size=(pn, rank))
        
        for it in range(max_iters):
            if time.time() - start_time > timeout:
                break
            
            # Solve for U given V, W
            # For each (i,j): sum_c U[i*m+j,c] * (V[j*p+k,c]*W[k*n+i,c] for all k) ... 
            # Actually, fix i,j. The equation is: for all k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This gives p equations for rank unknowns U[i*m+j, :]. 
            # Stack: for k=0..p-1: dot(U_ij, V_jk * W_ki) = 1
            # where V_jk = V[j*p + k, :], W_ki = W[k*n + i, :]
            # Matrix form: M[i,j] @ U[i*m+j,:] = 1 (vector of length p)
            # M[i,j] is p x rank, with M[k, c] = V[j*p+k, c] * W[k*n+i, c]
            
            A_U = np.zeros((n * m, p, rank))  # A_U[i*m+j, k, c] = V[j*p+k,c]*W[k*n+i,c]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        A_U[i*m+j, k, :] = V[j*p+k, :] * W[k*n+i, :]
            
            # Solve A_U[i*m+j, k, :] @ U[i*m+j, :] = 1 for k=0..p-1
            # This is an underdetermined system (rank unknowns, p equations)
            # Least squares solution
            for i in range(n):
                for j in range(m):
                    M = A_U[i*m+j]  # p x rank
                    # min ||M @ u - 1||^2
                    if rank >= p:
                        u, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    else:
                        u, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[i*m+j, :] = u
            
            # Solve for V given U, W
            # For fixed j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i
            # M_V[j*p+k, i, c] = U[i*m+j,c]*W[k*n+i,c], solve M_V @ v = 1 (vector of length n)
            A_V = np.zeros((m * p, n, rank))
            for j in range(m):
                for k in range(p):
                    for i in range(n):
                        A_V[j*p+k, i, :] = U[i*m+j, :] * W[k*n+i, :]
            for j in range(m):
                for k in range(p):
                    M = A_V[j*p+k]  # n x rank
                    v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V[j*p+k, :] = v
            
            # Solve for W given U, V
            # For fixed k,i: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all j
            # M_W[k*n+i, j, c] = U[i*m+j,c]*V[j*p+k,c], solve M_W @ w = 1 (vector of length m)
            A_W = np.zeros((p * n, m, rank))
            for k in range(p):
                for i in range(n):
                    for j in range(m):
                        A_W[k*n+i, j, :] = U[i*m+j, :] * V[j*p+k, :]
            for k in range(p):
                for i in range(n):
                    M = A_W[k*n+i]  # m x rank
                    w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W[k*n+i, :] = w
            
            # Check convergence
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        max_err = max(max_err, abs(s - 1.0))
            if max_err < tol:
                return U, V, W, max_err, True
        
        # Final error
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(s - 1.0))
        return U, V, W, max_err, max_err < tol

    # Try rank 17 first with multiple restarts
    rank_to_try = 17
    rng = np.random.default_rng(12345)
    
    best_result = None
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() - start_time > timeout:
            break
        
        # Initialize
        U_init = rng.normal(size=(nm, rank_to_try))
        V_init = rng.normal(size=(mp, rank_to_try))
        W_init = rng.normal(size=(pn, rank_to_try))
        
        # Run ALS
        U, V, W, err, success = als_step(rank_to_try, n, m, p, max_iters=200, tol=1e-14)
        
        if success and err < 1e-12:
            # Found a solution!
            # Try to clean up: snap to rationals if close
            best_result = (U, V, W, err)
            break
        
        if best_result is None or err < best_result[3]:
            best_result = (U, V, W, err)
    
    # If we found a valid rank-17 solution, use it
    if best_result is not None and best_result[3] < 1e-10:
        U, V, W, err = best_result
        rank = rank_to_try
    else:
        # Fall back to schoolbook rank 18
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
    
    # Convert to lists with integer or half-integer rounding if close
    def to_rational_list(arr):
        """Round to nearest integer or half-integer."""
        result = []
        for row in arr:
            new_row = []
            for val in row:
                # Try integer first
                if abs(val - round(val)) < 1e-8:
                    new_row.append(int(round(val)))
                elif abs(val - (2*round(val/2)*0.5)) < 1e-8:
                    # Half-integer
                    h = round(val * 2) / 2
                    if h == int(h):
                        new_row.append(int(h))
                    else:
                        new_row.append(h)
                else:
                    new_row.append(val)
            result.append(new_row)
        return result

    U_list = to_rational_list(np.array(U, dtype=float))
    V_list = to_rational_list(np.array(V, dtype=float))
    W_list = to_rational_list(np.array(W, dtype=float))
    
    # Final verification
    if not verify(U_list, V_list, W_list, tol=1e-8):
        # If rounding broke it, use the float values
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
