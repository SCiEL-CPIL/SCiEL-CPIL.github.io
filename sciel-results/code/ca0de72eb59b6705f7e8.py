"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    deadline = time.time() + 110  # leave safety margin before 120s kill

    def verify_np(U, V, W, n, m, p, rank, tol=1e-10):
        """Vectorized verification."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def rational_round(arr, denom=2):
        """Round to multiples of 1/denom."""
        return np.round(arr * denom) / denom

    def try_als(rank, max_iter=800, tol=1e-14, seed=42, init_method='random'):
        """Alternating least squares to find rank decomposition."""
        rng = np.random.RandomState(seed)
        
        if init_method == 'random':
            U = rng.randn(n * m, rank) * 0.5
            V = rng.randn(m * p, rank) * 0.5
            W = rng.randn(p * n, rank) * 0.5
        elif init_method == 'strassen_embed':
            # Border rank lifting: embed in 3x3x3 and project
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Strassen's 7-term decomposition for 2x2x2 core
            # M1 = (A11)(B12 - B22), M2 = (A11 + A21)(B22), 
            # M3 = (A21 - A11)(B11), M4 = (A22)(B21 - B11),
            # M5 = (A11 + A22)(B11 + B22), M6 = (A12 - A22)(B21 + B22),
            # M7 = (A21 + A11)(B12 - B11)
            strassen_U = [[1,0,0,0,1,0,0],
                          [0,1,-1,0,1,0,0],
                          [0,0,0,1,0,-1,0],
                          [0,0,0,0,0,0,1]]
            # Actually let me use the standard Strassen matrices
            # For C = A*B where A is 2x2, B is 2x2:
            # M1 = A11*(B12-B22), M2 = (A11+A21)*B22, M3 = (A21-A11)*B11,
            # M4 = A22*(B21-B11), M5 = (A11+A22)*(B11+B22), M6 = (A12-A22)*(B21+B22),
            # M7 = (A21+A11)*(B12-B11)
            # U factors (for A): row = i*m+j
            U_strassen = np.array([
                [1, 0, -1, 0, 1, 1, 1],   # A11: M1, M3, M5, M6, M7
                [1, 1, 0, 0, 0, 0, 0],    # A12: M1
                [0, 1, 1, 0, 1, 0, 1],    # A21: M2, M3, M5, M7
                [0, 0, 0, 1, 1, -1, 0]    # A22: M4, M5, M6
            ], dtype=float)
            V_strassen = np.array([
                [0, 0, 1, -1, 1, 0, -1],  # B11: M3, M4, M5, M7
                [-1, 0, 0, 0, 1, 0, 1],   # B12: M1, M5, M7
                [1, 1, 0, 0, 1, 1, 0],    # B21: M2, M4, M5, M6
                [-1, 1, 0, 1, 1, 1, 0]    # B22: M1, M2, M4, M5, M6
            ], dtype=float)
            W_strassen = np.array([
                [1, 0, 0, 0, 1, 0, 0],    # C11: M1, M5
                [0, 0, 0, 0, 1, 1, 0],    # C12: M5, M6
                [0, 0, 0, 0, 1, 0, 1],    # C21: M5, M7
                [0, 0, 0, 0, 1, 1, 1]     # C22: M5, M6, M7
            ], dtype=float)
            
            # Map 2x2x2 Strassen to 2x3x4 by embedding
            # A is 2x3: rows i=0,1; cols j=0,1,2
            # B is 3x4: rows j=0,1,2; cols k=0,1,2,3
            # C is 2x4: rows i=0,1; cols k=0,1,2,3
            
            # Use Strassen for the 2x2x2 sub-block (j=0,1; k=0,1)
            for t in range(7):
                # U: map Strassen U[i,j] to U[i*m+j, t] for j=0,1
                for i in range(2):
                    for j in range(2):
                        U[i*m+j, t] = U_strassen[i, j]
                # V: map Strassen V[j,k] to V[j*p+k, t] for j=0,1; k=0,1
                for j in range(2):
                    for k in range(2):
                        V[j*p+k, t] = V_strassen[j, k]
                # W: map Strassen W[k,i] to W[k*n+i, t] for k=0,1; i=0,1
                for k in range(2):
                    for i in range(2):
                        W[k*n+i, t] = W_strassen[k, i]
            
            # Add schoolbook terms for remaining dimensions
            # j=2 column of A, all k
            col = 7
            for k in range(p):
                if col >= rank:
                    break
                U[1*m+2, col] = 1.0  # A[1,2] (only row 1 has j=2 effectively)
                # Actually A is 2x3, so both rows have j=2
                U[0*m+2, col] = 1.0
                U[1*m+2, col] = 1.0
                V[2*p+k, col] = 1.0
                W[k*n+0, col] += 1.0  # C[0,k]
                W[k*n+1, col] += 1.0  # C[1,k]
                col += 1
            
            # k=2,3 columns of B for j=0,1 (already partially covered by Strassen for k=0,1)
            # Need to add terms for k=2,3 with j=0,1
            for j in range(2):
                for k in range(2, p):
                    if col >= rank:
                        break
                    for i in range(2):
                        U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    for i in range(2):
                        W[k*n+i, col] = 1.0
                    col += 1
            
            # Add randomization
            U += rng.randn(n * m, rank) * 0.05
            V += rng.randn(m * p, rank) * 0.05
            W += rng.randn(p * n, rank) * 0.05
        else:  # 'structured'
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            for t in range(min(rank, n*m*p)):
                i = t % n
                j = (t // n) % m
                k = (t // (n*m)) % p
                U[i*m+j, t] = 1.0
                V[j*p+k, t] = 1.0
                W[k*n+i, t] = 1.0
            U += rng.randn(n * m, rank) * 0.1
            V += rng.randn(m * p, rank) * 0.1
            W += rng.randn(p * n, rank) * 0.1

        for iteration in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Solve for W: fix U, V
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    w = k * n + i
                    A_mat = np.zeros((m, rank))
                    for j in range(m):
                        A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W_new[w, :], _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                    except np.linalg.LinAlgError:
                        W_new[w, :] = rng.randn(rank) * 0.1
            
            # Solve for U: fix V, W
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    u = i * m + j
                    A_mat = np.zeros((p, rank))
                    for k in range(p):
                        A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        U_new[u, :], _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                    except np.linalg.LinAlgError:
                        U_new[u, :] = rng.randn(rank) * 0.1
            
            # Solve for V: fix U, W
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    v = j * p + k
                    A_mat = np.zeros((n, rank))
                    for i in range(n):
                        A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V_new[v, :], _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                    except np.linalg.LinAlgError:
                        V_new[v, :] = rng.randn(rank) * 0.1
            
            diff = np.max(np.abs(W_new - W)) + np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
            W, U, V = W_new, U_new, V_new
            
            if diff < tol:
                if verify_np(U, V, W, n, m, p, rank):
                    return U, V, W
        
        if verify_np(U, V, W, n, m, p, rank):
            return U, V, W
        
        return None

    def rational_polish(U, V, W, rank, max_iters=200, seed=0):
        """Try to find a rational (half-integer) decomposition near the given solution."""
        rng = np.random.RandomState(seed)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = np.inf
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Round to half-integers
            U_r = rational_round(U, 2)
            V_r = rational_round(V, 2)
            W_r = rational_round(W, 2)
            
            if verify_np(U_r, V_r, W_r, n, m, p, rank, tol=1e-8):
                return U_r, V_r, W_r
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                        err += (s - 1.0)**2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
            
            # Perturb and retry
            scale = max(0.1, 1.0 - it / max_iters)
            U = U_r + rng.randn(n * m, rank) * 0.01 * scale
            V = V_r + rng.randn(m * p, rank) * 0.01 * scale
            W = W_r + rng.randn(p * n, rank) * 0.01 * scale
        
        return best_U, best_V, best_W

    best_result = None
    best_rank = n * m * p  # 24

    init_methods = ['strassen_embed', 'random', 'structured']
    
    for rank in range(14, 24):
        if time.time() > deadline - 10:
            break
        
        found = False
        for init_method in init_methods:
            if time.time() > deadline - 5:
                break
            
            for seed in range(20):
                if time.time() > deadline - 3:
                    break
                
                result = try_als(rank, max_iter=500, tol=1e-14, seed=seed, init_method=init_method)
                if result is not None:
                    U, V, W = result
                    
                    # Try rational polish
                    U_r, V_r, W_r = rational_polish(U, V, W, rank, max_iters=100, seed=seed)
                    
                    if verify_np(U_r, V_r, W_r, n, m, p, rank, tol=1e-8):
                        U_list = [[float(x) for x in row] for row in U_r]
                        V_list = [[float(x) for x in row] for row in V_r]
                        W_list = [[float(x) for x in row] for row in W_r]
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        found = True
                        break
                    
                    # Also try without polish
                    U_list = [[round(float(x), 12) for x in row] for row in U]
                    V_list = [[round(float(x), 12) for x in row] for row in V]
                    W_list = [[round(float(x), 12) for x in row] for row in W]
                    U_check = np.array(U_list, dtype=float)
                    V_check = np.array(V_list, dtype=float)
                    W_check = np.array(W_list, dtype=float)
                    if verify_np(U_check, V_check, W_check, n, m, p, rank, tol=1e-9):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        found = True
                        break
            
            if found:
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
