"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

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

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check all ones positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                if abs(val - 1) > tol:
                    return False
    
    # Check some zeros
    rng = np.random.RandomState(42)
    for _ in range(2000):
        a = rng.randint(0, n*m)
        b = rng.randint(0, m*p)
        c = rng.randint(0, p*n)
        i1, j1 = divmod(a, m)
        j2, k1 = divmod(b, p)
        k2, i2 = divmod(c, n)
        if i1 == i2 and j1 == j2 and k1 == k2:
            continue  # This is a ones position
        val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
        if abs(val) > tol:
            return False
    return True

def _try_rank(target_rank, time_limit):
    """Try to find a decomposition of the given rank."""
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Build the tensor as a matrix for least squares
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This is a 3-way tensor, hard to do least squares directly.
    # 
    # Alternative: use iterative optimization
    # Initialize U, V, W randomly
    rng = np.random.RandomState(0)
    
    best_loss = float('inf')
    best_U = best_V = best_W = None
    
    for attempt in range(10):
        if time.time() - start > time_limit:
            break
        
        U = rng.randn(n*m, target_rank) * 0.1
        V = rng.randn(m*p, target_rank) * 0.1
        W = rng.randn(p*n, target_rank) * 0.1
        
        # Iterate: fix two, solve for third
        for it in range(200):
            if time.time() - start > time_limit:
                break
            
            # Update U: for each a, U[a,:] should satisfy
            # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all b,c
            # This is a linear system in U[a,:]
            # Let M[a] be a matrix where M[a][bc, r] = V[b,r] * W[c,r]
            # Then U[a,:] = argmin ||M[a] @ U[a,:] - T[a,:]||^2
            
            # Build the linear system
            # Number of (b,c) pairs: m*p * p*n = 18 * 18 = 324
            # For each a, solve: (V_b * W_c) outer product matrix @ U[a,:] = T[a,b,c]
            
            # Precompute for efficiency
            for a in range(n*m):
                i1, j1 = divmod(a, m)
                # Build system: for each (b,c), coefficient is V[b,r]*W[c,r]
                # This is a 324 x target_rank system
                A_sys = np.zeros((m*p*p*n, target_rank))
                b_idx = 0
                for b in range(m*p):
                    for c in range(p*n):
                        A_sys[b_idx, :] = V[b, :] * W[c, :]
                        b_idx += 1
                
                # RHS: T[a,b,c] for all b,c
                rhs = np.zeros(m*p*p*n)
                b_idx = 0
                for b in range(m*p):
                    j2, k1 = divmod(b, p)
                    for c in range(p*n):
                        k2, i2 = divmod(c, n)
                        if i1 == i2 and j1 == j2 and k1 == k2:
                            rhs[b_idx] = 1.0
                        b_idx += 1
                
                # Solve least squares
                U[a, :], _, _, _ = np.linalg.lstsq(A_sys, rhs, rcond=None)
            
            # Similarly update V and W... (similar but different indexing)
            for b in range(m*p):
                j1, k1 = divmod(b, p)
                A_sys = np.zeros((n*m*p*n, target_rank))
                idx = 0
                for a in range(n*m):
                    for c in range(p*n):
                        A_sys[idx, :] = U[a, :] * W[c, :]
                        idx += 1
                rhs = np.zeros(n*m*p*n)
                idx = 0
                for a in range(n*m):
                    i1, j2 = divmod(a, m)
                    for c in range(p*n):
                        k2, i2 = divmod(c, n)
                        if i1 == i2 and j2 == j1 and k1 == k2:
                            rhs[idx] = 1.0
                        idx += 1
                V[b, :], _, _, _ = np.linalg.lstsq(A_sys, rhs, rcond=None)
            
            for c in range(p*n):
                k1, i1 = divmod(c, n)
                A_sys = np.zeros((n*m*m*p, target_rank))
                idx = 0
                for a in range(n*m):
                    for b in range(m*p):
                        A_sys[idx, :] = U[a, :] * V[b, :]
                        idx += 1
                rhs = np.zeros(n*m*m*p)
                idx = 0
                for a in range(n*m):
                    i2, j1 = divmod(a, m)
                    for b in range(m*p):
                        j2, k2 = divmod(b, p)
                        if i1 == i2 and j1 == j2 and k1 == k2:
                            rhs[idx] = 1.0
                        idx += 1
                W[c, :], _, _, _ = np.linalg.lstsq(A_sys, rhs, rcond=None)
        
        # Check loss
        loss = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b = j*p+k
                    c = k*n+i
                    val = np.dot(U[a,:], V[b,:]*W[c,:])
                    loss += (val-1)**2
        # Check zeros
        for _ in range(100):
            a = rng.randint(0, n*m)
            b = rng.randint(0, m*p)
            c = rng.randint(0, p*n)
            i1,j1 = divmod(a,m)
            j2,k1 = divmod(b,p)
            k2,i2 = divmod(c,n)
            if i1==i2 and j1==j2 and k1==k2:
                continue
            val = np.dot(U[a,:], V[b,:]*W[c,:])
            loss += val**2
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-10:
            break
    
    if best_loss < 1e-8:
        # Round to half-integers
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        # Verify rounded version
        if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Try rank 21
    result = _try_rank(21, 60)
    if result is not None:
        return result
    
    if time.time() > deadline:
        return _schoolbook()
    
    # Try rank 27
    result = _try_rank(27, 40)
    if result is not None:
        return result
    
    if time.time() > deadline:
        return _schoolbook()
    
    # Fallback
    return _schoolbook()
# EVOLVE-BLOCK-END
