"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    deadline = time.time() + 110  # leave safety margin before 120s kill

    def verify_np(U, V, W, n, m, p, rank, tol=1e-10):
        """Vectorized verification using tensor contraction."""
        # Build the multiplication tensor T[i,j,k] = delta (should be 1 for all)
        # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # We need T[i,j,k] = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute dot product
                    u_row = U[i*m+j, :]
                    v_row = V[j*p+k, :]
                    w_row = W[k*n+i, :]
                    s = np.sum(u_row * v_row * w_row)
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def verify_fast(U, V, W, n, m, p, rank, tol=1e-10):
        """Faster verification by reshaping."""
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        # For each (i,j,k): sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
        # = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
        # We can compute this as: for each t, outer product
        # Actually, let's just loop since n,m,p are small
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U3[i,j,:] * V3[j,k,:] * W3[k,i,:])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def try_construct_direct(rank, seed=42):
        """Try to directly construct a decomposition using structured initialization + ALS."""
        rng = np.random.RandomState(seed)
        
        # Strategy: Use the fact that matrix multiplication tensor has rank at most n*m*p
        # and try to find a lower rank decomposition.
        
        # Initialize with a structured pattern
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Try different initialization strategies
        strategies = ['schoolbook_partial', 'strassen_based', 'random']
        
        for strat in strategies:
            if time.time() > deadline - 5:
                break
                
            if strat == 'schoolbook_partial':
                # Start with schoolbook and try to merge terms
                for t in range(min(rank, n*m*p)):
                    i = t % n
                    j = (t // n) % m
                    k = (t // (n*m)) % p
                    U[i*m+j, t] = 1.0
                    V[j*p+k, t] = 1.0
                    W[k*n+i, t] = 1.0
                # Add small random perturbation
                U += rng.randn(n * m, rank) * 0.05
                V += rng.randn(m * p, rank) * 0.05
                W += rng.randn(p * n, rank) * 0.05
            elif strat == 'strassen_based':
                # Use Strassen for 2x2x2 core + additional terms
                # Strassen 7-term for the j=0,1; k=0,1 block
                U_strassen = np.array([
                    [1, 0, -1, 0, 1, 1, 1],
                    [1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 1, 0, 1],
                    [0, 0, 0, 1, 1, -1, 0]
                ], dtype=float)
                V_strassen = np.array([
                    [0, 0, 1, -1, 1, 0, -1],
                    [-1, 0, 0, 0, 1, 0, 1],
                    [1, 1, 0, 0, 1, 1, 0],
                    [-1, 1, 0, 1, 1, 1, 0]
                ], dtype=float)
                W_strassen = np.array([
                    [1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 1, 0, 1],
                    [0, 0, 0, 0, 1, 1, 1]
                ], dtype=float)
                
                for t in range(7):
                    for i in range(2):
                        for j in range(2):
                            U[i*m+j, t] = U_strassen[i, j]
                    for j in range(2):
                        for k in range(2):
                            V[j*p+k, t] = V_strassen[j, k]
                    for k in range(2):
                        for i in range(2):
                            W[k*n+i, t] = W_strassen[k, i]
                
                # Add terms for j=2 (third column of A)
                col = 7
                for k in range(p):
                    if col < rank:
                        U[0*m+2, col] = 1.0
                        U[1*m+2, col] = 1.0
                        V[2*p+k, col] = 1.0
                        W[k*n+0, col] = 1.0
                        W[k*n+1, col] = 1.0
                        col += 1
                
                # Add terms for k=2,3 with j=0,1
                for j in range(2):
                    for k in range(2, p):
                        if col < rank:
                            U[0*m+j, col] = 1.0
                            U[1*m+j, col] = 1.0
                            V[j*p+k, col] = 1.0
                            W[k*n+0, col] = 1.0
                            W[k*n+1, col] = 1.0
                            col += 1
                
                # Perturb
                U += rng.randn(n * m, rank) * 0.03
                V += rng.randn(m * p, rank) * 0.03
                W += rng.randn(p * n, rank) * 0.03
            else:  # random
                U = rng.randn(n * m, rank) * 0.3
                V = rng.randn(m * p, rank) * 0.3
                W = rng.randn(p * n, rank) * 0.3
            
            # Run ALS
            result = _als_solve(U.copy(), V.copy(), W.copy(), rank, max_iter=300, seed=seed)
            if result is not None:
                return result
        
        return None

    def _als_solve(U, V, W, rank, max_iter=300, seed=42):
        """Alternating least squares solver."""
        rng = np.random.RandomState(seed)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 3:
                return None
            
            # Solve for W
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    w = k * n + i
                    A_mat = np.zeros((m, rank))
                    for j in range(m):
                        A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W_new[w, :] = np.linalg.lstsq(A_mat, b, rcond=None)[0]
                    except:
                        W_new[w, :] = rng.randn(rank) * 0.1
            
            # Solve for U
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    u = i * m + j
                    A_mat = np.zeros((p, rank))
                    for k in range(p):
                        A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        U_new[u, :] = np.linalg.lstsq(A_mat, b, rcond=None)[0]
                    except:
                        U_new[u, :] = rng.randn(rank) * 0.1
            
            # Solve for V
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    v = j * p + k
                    A_mat = np.zeros((n, rank))
                    for i in range(n):
                        A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V_new[v, :] = np.linalg.lstsq(A_mat, b, rcond=None)[0]
                    except:
                        V_new[v, :] = rng.randn(rank) * 0.1
            
            diff = np.max(np.abs(W_new - W)) + np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
            W, U, V = W_new, U_new, V_new
            
            if diff < 1e-12:
                if verify_np(U, V, W, n, m, p, rank, tol=1e-9):
                    return U, V, W
        
        if verify_np(U, V, W, n, m, p, rank, tol=1e-9):
            return U, V, W
        
        return None

    def rational_polish(U, V, W, rank, max_iters=50, seed=0):
        """Polish to half-integers."""
        rng = np.random.RandomState(seed)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = np.inf
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if verify_np(U_r, V_r, W_r, n, m, p, rank, tol=1e-8):
                return U_r, V_r, W_r
            
            # Compute error
            U3 = U_r.reshape(n, m, rank)
            V3 = V_r.reshape(m, p, rank)
            W3 = W_r.reshape(p, n, rank)
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U3[i,j,:] * V3[j,k,:] * W3[k,i,:])
                        err += (s - 1.0)**2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
            
            scale = max(0.05, 1.0 - it / max_iters)
            U = U_r + rng.randn(n * m, rank) * 0.01 * scale
            V = V_r + rng.randn(m * p, rank) * 0.01 * scale
            W = W_r + rng.randn(p * n, rank) * 0.01 * scale
        
        return best_U, best_V, best_W

    best_result = None
    best_rank = n * m * p  # 24

    # Try ranks from lower bound up
    for rank in range(14, 25):
        if time.time() > deadline - 10:
            break
        
        found = False
        for seed in range(15):
            if time.time() > deadline - 5:
                break
            
            result = try_construct_direct(rank, seed=seed)
            if result is not None:
                U, V, W = result
                
                # Try rational polish
                U_r, V_r, W_r = rational_polish(U, V, W, rank, max_iters=30, seed=seed)
                
                if verify_np(U_r, V_r, W_r, n, m, p, rank, tol=1e-8):
                    best_result = {
                        "U": [[float(x) for x in row] for row in U_r],
                        "V": [[float(x) for x in row] for row in V_r],
                        "W": [[float(x) for x in row] for row in W_r]
                    }
                    best_rank = rank
                    found = True
                    break
                
                # Also try without polish
                if verify_np(U, V, W, n, m, p, rank, tol=1e-9):
                    best_result = {
                        "U": [[round(float(x), 12) for x in row] for row in U],
                        "V": [[round(float(x), 12) for x in row] for row in V],
                        "W": [[round(float(x), 12) for x in row] for row in W]
                    }
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
    
    if best_result is not None:
        return best_result

    # Fallback: schoolbook rank 24
    rank = n * m * p  # 24
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
# EVOLVE-BLOCK-END
