import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition: sum_r W[k*n+i,r]*U[i*m+j1,r]*V[j2*p+k,r] == delta_{j1,j2}"""
    r = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = np.dot(W[k*n+i, :], U[i*m+j1, :] * V[j2*p+k, :])
                    target = 1.0 if j1 == j2 else 0.0
                    if abs(s - target) > 1e-8:
                        return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Vectorized verification using matrix products."""
    r = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    # For each (i, k): sum_r W3[k,i,r] * U3[i,j1,r] * V3[j2,k,r] should be delta_{j1,j2}
    # Compute M[i,k,j1,j2] = sum_r W3[k,i,r]*U3[i,j1,r]*V3[j2,k,r]
    # = sum_r W3[k,i,r] * U3[i,j1,r] * V3[j2,k,r]
    # Let's compute this as: for each r, contribution is W3[:,:,r] outer U3[:,:,r] outer V3[:,:,r]
    # But that's 4D. Instead:
    # M[i,k,j1,j2] = sum_r W3[k,i,r] * U3[i,j1,r] * V3[j2,k,r]
    # = sum_j1' sum_j2' ... no.
    # Let's do it in chunks. For fixed (i,k):
    #   vec = W3[k,i,:] * U3[i,:,:] * V3[:,k,:]  -- no, that's element-wise over r.
    # Actually: sum_r W3[k,i,r] * U3[i,j1,r] * V3[j2,k,r]
    # For fixed (i,k), this is a matrix M_{j1,j2} = sum_r (W3[k,i,r]*U3[i,j1,r]) * V3[j2,k,r]
    # = (U3[i,:,r] * W3[k,i,r]) as row vector times V3[:,k,r] as column vector, summed over r.
    # = sum_r (W3[k,i,r] * U3[i,:,r]) @ V3[:,k,r]
    # Let's compute: for each r, form outer product of (W3[k,i,r]*U3[i,:,r]) with V3[:,k,r]
    # Sum over r.
    # Vectorized: 
    # A[i,k,j1] = sum_r W3[k,i,r] * U3[i,j1,r]  -- this is (W3[:,:,r] * U3[:,:,r]) summed? No.
    # A[i,k,j1] = sum_r W3[k,i,r]*U3[i,j1,r]
    # B[i,k,j2] = V3[j2,k,r] -- but this depends on r too.
    # So M[i,k,j1,j2] = sum_r W3[k,i,r]*U3[i,j1,r]*V3[j2,k,r]
    # Let's just loop over i,k and use matrix multiplication.
    for i in range(n):
        for k in range(p):
            # M[j1,j2] = sum_r W3[k,i,r] * U3[i,j1,r] * V3[j2,k,r]
            # = sum_r (W3[k,i,r] * U3[i,j1,r]) * V3[j2,k,r]
            # Let a_r = W3[k,i,r] * U3[i,:,r]  (shape m)
            # Let b_r = V3[:,k,r]  (shape m)
            # M = sum_r a_r @ b_r  (outer product, m x m)
            a = W3[k, i, :] * U3[i, :, :]  # (m, r)
            b = V3[:, k, :]  # (m, r)
            M = a @ b.T  # (m, m)
            target = np.eye(m)
            if not np.allclose(M, target, atol=1e-8):
                return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W, rank

def _als_optimize(U, V, W, n, m, p, max_iters=100, tol=1e-10):
    """ALS optimization for the tensor decomposition. Returns (U, V, W, converged)."""
    r = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    prev_err = np.inf
    for it in range(max_iters):
        # Update W: for each (k,i), solve least squares
        for k in range(p):
            for i in range(n):
                # Equations: for all (j1,j2): sum_r W[k*n+i,r]*U[i*m+j1,r]*V[j2*p+k,r] = delta_{j1,j2}
                # Linear in W[k*n+i,:]: A @ w = b where A[j1*m+j2, r] = U[i*m+j1,r]*V[j2*p+k,r]
                A_W = np.zeros((m*m, r))
                b = np.zeros(m*m)
                for j1 in range(m):
                    for j2 in range(m):
                        A_W[j1*m+j2, :] = U[i*m+j1, :] * V[j2*p+k, :]
                        if j1 == j2:
                            b[j1*m+j2] = 1.0
                w, *_ = np.linalg.lstsq(A_W, b, rcond=None)
                W[k*n+i, :] = w
        
        # Update U: for each (i,j1), solve least squares
        for i in range(n):
            for j1 in range(m):
                # Equations: for all (k,j2): sum_r W[k*n+i,r]*U[i*m+j1,r]*V[j2*p+k,r] = delta_{j1,j2}
                # Linear in U[i*m+j1,:]: A @ u = b where A[(k,j2), r] = W[k*n+i,r]*V[j2*p+k,r]
                A_U = np.zeros((p*m, r))
                b = np.zeros(p*m)
                for k in range(p):
                    for j2 in range(m):
                        A_U[k*m+j2, :] = W[k*n+i, :] * V[j2*p+k, :]
                        if j1 == j2:
                            b[k*m+j2] = 1.0
                u, *_ = np.linalg.lstsq(A_U, b, rcond=None)
                U[i*m+j1, :] = u
        
        # Update V: for each (j2,k), solve least squares
        for j2 in range(m):
            for k in range(p):
                # Equations: for all (i,j1): sum_r W[k*n+i,r]*U[i*m+j1,r]*V[j2*p+k,r] = delta_{j1,j2}
                # Linear in V[j2*p+k,:]: A @ v = b where A[(i,j1), r] = W[k*n+i,r]*U[i*m+j1,r]
                A_V = np.zeros((n*m, r))
                b = np.zeros(n*m)
                for i in range(n):
                    for j1 in range(m):
                        A_V[i*m+j1, :] = W[k*n+i, :] * U[i*m+j1, :]
                        if j1 == j2:
                            b[i*m+j1] = 1.0
                v, *_ = np.linalg.lstsq(A_V, b, rcond=None)
                V[j2*p+k, :] = v
        
        # Compute error
        err = 0.0
        for i in range(n):
            for k in range(p):
                a = W[k*n+i, :] * U[i, :, :]  # (m, r)
                b = V[:, k, :]  # (m, r)
                M = a @ b.T
                err += np.sum((M - np.eye(m))**2)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W, (prev_err < 1e-8)

def _gradient_descent(U, V, W, n, m, p, lr=0.01, max_iters=500):
    """Gradient descent on the loss function."""
    r = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iters):
        # Compute gradients
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        gW = np.zeros_like(W)
        
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        target = 1.0 if j1 == j2 else 0.0
                        # d/dU[i*m+j1,r] = 2*(s - target)*W[k*n+i,r]*V[j2*p+k,r]
                        # d/dV[j2*p+k,r] = 2*(s - target)*W[k*n+i,r]*U[i*m+j1,r]
                        # d/dW[k*n+i,r] = 2*(s - target)*U[i*m+j1,r]*V[j2*p+k,r]
                        s = np.dot(W[k*n+i, :], U[i*m+j1, :] * V[j2*p+k, :])
                        diff = 2 * (s - target)
                        gU[i*m+j1, :] += diff * W[k*n+i, :] * V[j2*p+k, :]
                        gV[j2*p+k, :] += diff * W[k*n+i, :] * U[i*m+j1, :]
                        gW[k*n+i, :] += diff * U[i*m+j1, :] * V[j2*p+k, :]
        
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s limit
    
    # Schoolbook baseline
    U0, V0, W0, rank0 = _schoolbook(n, m, p)
    best = (rank0, U0.copy(), V0.copy(), W0.copy())
    
    np.random.seed(42)
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, n*p) = max(15, 25, 15) = 25
    # But tensor rank is likely higher. Let's try a range.
    min_rank = max(n*m, m*p, n*p)
    # Try ranks from min_rank to rank0-1
    candidate_ranks = list(range(min_rank, rank0))
    # Sort by difficulty: try smaller ranks first (harder but more valuable)
    # Actually, let's try a subset to save time
    # Try every other rank, plus some key ones
    candidate_ranks = [r for r in candidate_ranks if r % 2 == 0] + [rank0 - 1, rank0 - 2, rank0 - 3]
    candidate_ranks = sorted(set(candidate_ranks))
    
    for r in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        # Try multiple random restarts
        max_restarts = 5
        if time.time() > deadline - 30:
            max_restarts = 2
        if time.time() > deadline - 15:
            max_restarts = 1
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # ALS optimization
            U, V, W, converged = _als_optimize(U, V, W, n, m, p, max_iters=50)
            
            if converged:
                # Try rounding to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if _verify_fast(U_r, V_r, W_r, n, m, p):
                    if r < best[0]:
                        best = (r, U_r.copy(), V_r.copy(), W_r.copy())
                        print(f"Found rank {r} decomposition!")
                    break  # Found a valid one for this r, try smaller r
                else:
                    # Try without rounding (exact float)
                    if _verify_fast(U, V, W, n, m, p):
                        # Entries may not be half-integers, but let's try
                        # Round to nearest half-integer and re-verify
                        U_r = np.round(U * 2) / 2
                        V_r = np.round(V * 2) / 2
                        W_r = np.round(W * 2) / 2
                        if _verify_fast(U_r, V_r, W_r, n, m, p):
                            if r < best[0]:
                                best = (r, U_r.copy(), V_r.copy(), W_r.copy())
                            break
                        else:
                            # Use the float values directly (they should be close to half-integers)
                            if r < best[0]:
                                best = (r, U.copy(), V.copy(), W.copy())
                            break
    
    r, U, V, W = best
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
