"""Matrix multiplication 2x3x9 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54

    def build_schoolbook():
        """Return (U, V, W) for the trivial rank-54 decomposition."""
        r = schoolbook_rank
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    def tensor_error(U, V, W, n, m, p):
        """Compute sum over (i,j,k) of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2."""
        r = U.shape[1]
        total = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U[i * m + j]
                for k in range(p):
                    v_row = V[j * p + k]
                    w_row = W[k * n + i]
                    val = np.dot(u_row, v_row * w_row)
                    total += (val - 1.0) ** 2
        return total

    def verify_exact(U, V, W, n, m, p, tol=1e-10):
        """Check if decomposition is exact within tolerance."""
        for i in range(n):
            for j in range(m):
                u_row = U[i * m + j]
                for k in range(p):
                    v_row = V[j * p + k]
                    w_row = W[k * n + i]
                    val = np.dot(u_row, v_row * w_row)
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def als_step(U, V, W, n, m, p, max_iter=200, tol=1e-12, time_budget=None):
        """
        Alternating Least Squares for tensor decomposition.
        U: (n*m, r), V: (m*p, r), W: (p*n, r)
        Minimize sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
        """
        r = U.shape[1]
        U = U.copy()
        V = V.copy()
        W = W.copy()

        prev_err = tensor_error(U, V, W, n, m, p)

        for it in range(max_iter):
            if time_budget is not None and time.time() > time_budget:
                break

            # Solve for U: fix V, W
            # For each (i,j): find u such that sum_c u[c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all k
            # A_ij has shape (p, r), A_ij[k,c] = V[j*p+k,c]*W[k*n+i,c]
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A_ij = np.empty((p, r))
                    for k in range(p):
                        A_ij[k] = V[j * p + k] * W[k * n + i]
                    b = np.ones(p)
                    # Solve least squares
                    u_sol, _, _, _ = np.linalg.lstsq(A_ij, b, rcond=None)
                    U_new[i * m + j] = u_sol
            U = U_new

            # Solve for V: fix U, W
            # For each (j,k): find v such that sum_c U[i*m+j,c]*v[c]*W[k*n+i,c] = 1 for all i
            # A_jk has shape (n, r), A_jk[i,c] = U[i*m+j,c]*W[k*n+i,c]
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A_jk = np.empty((n, r))
                    for i in range(n):
                        A_jk[i] = U[i * m + j] * W[k * n + i]
                    b = np.ones(n)
                    v_sol, _, _, _ = np.linalg.lstsq(A_jk, b, rcond=None)
                    V_new[j * p + k] = v_sol
            V = V_new

            # Solve for W: fix U, V
            # For each (k,i): find w such that sum_c U[i*m+j,c]*V[j*p+k,c]*w[c] = 1 for all j
            # A_ki has shape (m, r), A_ki[j,c] = U[i*m+j,c]*V[j*p+k,c]
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A_ki = np.empty((m, r))
                    for j in range(m):
                        A_ki[j] = U[i * m + j] * V[j * p + k]
                    b = np.ones(m)
                    w_sol, _, _, _ = np.linalg.lstsq(A_ki, b, rcond=None)
                    W_new[k * n + i] = w_sol
            W = W_new

            # Check convergence
            err = tensor_error(U, V, W, n, m, p)
            if abs(prev_err - err) < tol * max(1.0, err):
                break
            prev_err = err

        return U, V, W

    def polish(U, V, W, n, m, p, max_iter=500, time_budget=None):
        """
        High-precision polish: run ALS with tighter tolerance to drive error to ~0.
        """
        return als_step(U, V, W, n, m, p, max_iter=max_iter, tol=1e-14, time_budget=time_budget)

    # Build schoolbook as fallback
    U_sb, V_sb, W_sb = build_schoolbook()
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Search for lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # But tensor rank can be higher. Let's search from 27 upward.
    
    # Time budget per rank: we have ~140s total
    # Allocate time based on remaining ranks
    ranks_to_try = list(range(27, schoolbook_rank))
    
    for r in ranks_to_try:
        remaining_time = deadline - time.time()
        if remaining_time < 5.0:
            break
        
        # Allocate time for this rank: proportional to remaining ranks
        ranks_remaining = len(ranks_to_try) - ranks_to_try.index(r)
        time_per_rank = max(3.0, (remaining_time - 5.0) / max(1, ranks_remaining))
        
        # Number of random restarts based on time available
        n_restarts = max(1, int(time_per_rank / 2.0))
        n_restarts = min(n_restarts, 10)
        
        found = False
        for restart in range(n_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Random initialization
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # Run ALS
            U, V, W = als_step(U, V, W, n, m, p, max_iter=100, 
                               time_budget=deadline - 10.0)
            
            # Check if we got a good solution
            err = tensor_error(U, V, W, n, m, p)
            
            if err < 1e-8:
                # Polish to high precision
                U, V, W = polish(U, V, W, n, m, p, max_iter=200,
                                time_budget=deadline - 5.0)
                err = tensor_error(U, V, W, n, m, p)
                
                if verify_exact(U, V, W, n, m, p, tol=1e-10):
                    best_rank = r
                    best_U, best_V, best_W = U, V, W
                    found = True
                    break
        
        if found:
            break
    
    # If no better rank found, try a few more ranks with more restarts if time permits
    if best_rank == schoolbook_rank:
        remaining_time = deadline - time.time()
        if remaining_time > 10.0:
            # Try ranks 27-35 with more restarts
            for r in range(27, min(36, schoolbook_rank)):
                if time.time() > deadline - 5.0:
                    break
                
                n_restarts = 5
                found = False
                for restart in range(n_restarts):
                    if time.time() > deadline - 5.0:
                        break
                    
                    U = np.random.randn(n * m, r) * 0.5
                    V = np.random.randn(m * p, r) * 0.5
                    W = np.random.randn(p * n, r) * 0.5
                    
                    U, V, W = als_step(U, V, W, n, m, p, max_iter=150,
                                       time_budget=deadline - 10.0)
                    
                    err = tensor_error(U, V, W, n, m, p)
                    
                    if err < 1e-8:
                        U, V, W = polish(U, V, W, n, m, p, max_iter=300,
                                        time_budget=deadline - 5.0)
                        err = tensor_error(U, V, W, n, m, p)
                        
                        if verify_exact(U, V, W, n, m, p, tol=1e-10):
                            best_rank = r
                            best_U, best_V, best_W = U, V, W
                            found = True
                            break
                
                if found:
                    break
    
    # Convert to list of lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
