import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def _verify_half_integer(arr, tol=1e-9):
    doubled = arr * 2
    rounded = np.round(doubled)
    if np.max(np.abs(doubled - rounded)) > tol:
        return False
    return True

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_to_half(arr):
    return np.round(arr * 2) / 2.0

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _als_decompose(n, m, p, r, max_iters=50, tol=1e-10, seed=0):
    """
    Alternating Least Squares for tensor decomposition.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    np.random.seed(seed)
    
    # Initialize
    U = np.random.randn(n*m, r)
    V = np.random.randn(m*p, r)
    W = np.random.randn(p*n, r)
    
    # Precompute indices
    idx_U = []
    idx_V = []
    idx_W = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_U.append(i*m+j)
                idx_V.append(j*p+k)
                idx_W.append(k*n+i)
    idx_U = np.array(idx_U)
    idx_V = np.array(idx_V)
    idx_W = np.array(idx_W)
    
    prev_err = float('inf')
    
    for it in range(max_iters):
        # Update U: Fix V, W. Solve for U rows.
        # For each row i*m+j, we want to minimize sum_k (1 - sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c])^2
        # This is a linear least squares problem for each row of U.
        # Let's vectorize.
        # T is 1 for all (i,j,k).
        # For a fixed U row u_idx, the equation is:
        # sum_c U[u_idx, c] * (V[v_idx, c] * W[w_idx, c]) = 1
        # This must hold for all k associated with this (i,j).
        # Actually, for a fixed (i,j), k varies.
        # So for each (i,j), we have p equations:
        # sum_c U[i*m+j, c] * X_{k,c} = 1, where X_{k,c} = V[j*p+k, c] * W[k*n+i, c]
        
        new_U = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                u_idx = i*m+j
                # Construct matrix X of shape (p, r)
                # X[k, c] = V[j*p+k, c] * W[k*n+i, c]
                X = np.zeros((p, r))
                for k in range(p):
                    X[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve X @ U[u_idx, :] = 1
                # Least squares: U[u_idx, :] = pinv(X) @ 1
                try:
                    new_U[u_idx, :] = np.linalg.lstsq(X, np.ones(p), rcond=None)[0]
                except:
                    new_U[u_idx, :] = U[u_idx, :]
        U = new_U
        
        # Update V: Fix U, W.
        # For fixed (j,k), i varies.
        # sum_c V[j*p+k, c] * (U[i*m+j, c] * W[k*n+i, c]) = 1
        new_V = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                v_idx = j*p+k
                # Construct matrix Y of shape (n, r)
                # Y[i, c] = U[i*m+j, c] * W[k*n+i, c]
                Y = np.zeros((n, r))
                for i in range(n):
                    Y[i, :] = U[i*m+j, :] * W[k*n+i, :]
                # Solve Y @ V[v_idx, :] = 1
                try:
                    new_V[v_idx, :] = np.linalg.lstsq(Y, np.ones(n), rcond=None)[0]
                except:
                    new_V[v_idx, :] = V[v_idx, :]
        V = new_V
        
        # Update W: Fix U, V.
        # For fixed (k,i), j varies.
        # sum_c W[k*n+i, c] * (U[i*m+j, c] * V[j*p+k, c]) = 1
        new_W = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                w_idx = k*n+i
                # Construct matrix Z of shape (m, r)
                # Z[j, c] = U[i*m+j, c] * V[j*p+k, c]
                Z = np.zeros((m, r))
                for j in range(m):
                    Z[j, :] = U[i*m+j, :] * V[j*p+k, :]
                # Solve Z @ W[w_idx, :] = 1
                try:
                    new_W[w_idx, :] = np.linalg.lstsq(Z, np.ones(m), rcond=None)[0]
                except:
                    new_W[w_idx, :] = W[w_idx, :]
        W = new_W
        
        # Calculate error
        err = 0.0
        for idx in range(len(idx_U)):
            u_idx = idx_U[idx]
            v_idx = idx_V[idx]
            w_idx = idx_W[idx]
            val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
            err += (val - 1.0)**2
            
        if abs(prev_err - err) < tol:
            break
        prev_err = err
        
    return U, V, W

def _optimize_rank(r, n, m, p, time_limit=10):
    start_time = time.time()
    best_loss = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Try multiple seeds
    for seed in range(10):
        if time.time() - start_time > time_limit:
            break
            
        U, V, W = _als_decompose(n, m, p, r, max_iters=30, seed=seed)
        
        # Calculate loss
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    loss += (val - 1.0)**2
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U, V, W
            
            # Check if we can snap to half-integers and verify
            U_snap = _snap_to_half(U)
            V_snap = _snap_to_half(V)
            W_snap = _snap_to_half(W)
            
            if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-6):
                return U_snap, V_snap, W_snap, True
                
    # If we didn't find a snapped solution, try one more time with the best raw solution
    if best_U is not None and best_loss < 1e-4:
        U_snap = _snap_to_half(best_U)
        V_snap = _snap_to_half(best_V)
        W_snap = _snap_to_half(best_W)
        if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-6):
            return U_snap, V_snap, W_snap, True
            
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    U_school, V_school, W_school = _make_schoolbook(n, m, p)
    
    # Lower bound
    lower_bound = 24
    
    for r in range(lower_bound, n*m*p + 1):
        if time.time() > deadline - 8:
            break
            
        U, V, W, success = _optimize_rank(r, n, m, p, time_limit=min(10, deadline - time.time() - 5))
        
        if success:
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
            
    U_list = [[float(x) for x in row] for row in U_school]
    V_list = [[float(x) for x in row] for row in V_school]
    W_list = [[float(x) for x in row] for row in W_school]
    return {"U": U_list, "V": V_list, "W": W_list}
