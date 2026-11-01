"""Matrix multiplication 2x3 x 3x3 rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (9, rank)  
    # W: (p*n, rank) = (6, rank)
    
    def tensor_residual(U, V, W):
        """Compute residual efficiently using matrix operations."""
        rank = U.shape[1]
        total = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    total += (s - 1.0) ** 2
        return total
    
    def is_half_integer(arr, tol=1e-8):
        """Check if all entries are integer or half-integer."""
        flat = arr.flatten()
        v = flat * 2
        return np.all(np.abs(v - np.round(v)) < tol)
    
    def round_to_half(arr):
        return np.round(arr * 2) / 2
    
    def verify_solution(U, V, W, tol=1e-6):
        """Verify that the decomposition is exact and has half-integer entries."""
        if not (is_half_integer(U, 1e-8) and is_half_integer(V, 1e-8) and is_half_integer(W, 1e-8)):
            return False
        res = tensor_residual(U, V, W)
        return res < tol
    
    # Schoolbook as fallback
    def schoolbook():
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
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    def als_step(U, V, W, target_rank, n, m, p, max_iter=200, tol=1e-10):
        """Alternating Least Squares for tensor decomposition."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        prev_res = tensor_residual(U, V, W)
        
        for it in range(max_iter):
            if time.time() > deadline - 20:
                break
            
            # Update U: for each row i*m+j, solve least squares
            for i in range(n):
                for j in range(m):
                    A = np.zeros((m*p, target_rank))
                    b = np.zeros(target_rank)
                    for k in range(p):
                        A[j*p+k, :] = V[j*p+k, :] * W[k*n+i, :]
                        b += W[k*n+i, :]
                    # U[i*m+j, :] should satisfy A @ U[i*m+j, :] = b
                    # i.e., sum_r U[i*m+j,r] * A[r, :] = b
                    # This is: U[i*m+j, :] @ A.T = b.T
                    # So A.T @ U[i*m+j, :].T = b
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(A.T, b, rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each row j*p+k, solve least squares
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n*m, target_rank))
                    b = np.zeros(target_rank)
                    for i in range(n):
                        A[i*m+j, :] = U[i*m+j, :] * W[k*n+i, :]
                        b += U[i*m+j, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(A.T, b, rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each row k*n+i, solve least squares
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m*p, target_rank))
                    b = np.zeros(target_rank)
                    for j in range(m):
                        A[j*p+k, :] = U[i*m+j, :] * V[j*p+k, :]
                        b += V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(A.T, b, rcond=None)[0]
                    except:
                        pass
            
            res = tensor_residual(U, V, W)
            if abs(prev_res - res) < tol:
                break
            prev_res = res
        
        return U, V, W, prev_res
    
    def try_rank(target_rank, max_attempts=6):
        """Try to find a rank-minimal decomposition for given target_rank."""
        nonlocal best, best_rank
        if target_rank >= best_rank:
            return False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 25:
                return False
            
            # Different initialization strategies
            seed = 42 + target_rank * 1000 + attempt * 7
            np.random.seed(seed)
            
            if attempt % 3 == 0:
                # Sparse random initialization with small values
                U = np.zeros((n*m, target_rank))
                V = np.zeros((m*p, target_rank))
                W = np.zeros((p*n, target_rank))
                for r in range(target_rank):
                    n_u = np.random.randint(2, n*m+1)
                    n_v = np.random.randint(2, m*p+1)
                    n_w = np.random.randint(2, p*n+1)
                    U[np.random.choice(n*m, n_u, replace=False), r] = np.random.choice([-1, 1], n_u) / 2
                    V[np.random.choice(m*p, n_v, replace=False), r] = np.random.choice([-1, 1], n_v) / 2
                    W[np.random.choice(p*n, n_w, replace=False), r] = np.random.choice([-1, 1], n_w) / 2
            elif attempt % 3 == 1:
                # Dense random half-integer
                U = np.random.randint(-2, 3, size=(n*m, target_rank)) / 2.0
                V = np.random.randint(-2, 3, size=(m*p, target_rank)) / 2.0
                W = np.random.randint(-1, 2, size=(p*n, target_rank)) / 2.0
            else:
                # Structured: exploit symmetry, start from scaled schoolbook
                U = np.zeros((n*m, target_rank))
                V = np.zeros((m*p, target_rank))
                W = np.zeros((p*n, target_rank))
                # Initialize first few columns from schoolbook pattern
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col < target_rank:
                                U[i*m+j, col] = 1
                                V[j*p+k, col] = 1
                                W[k*n+i, col] = 1
                                col += 1
                # Add small random perturbation to break symmetry
                U += np.random.randn(n*m, target_rank) * 0.1
                V += np.random.randn(m*p, target_rank) * 0.1
                W += np.random.randn(p*n, target_rank) * 0.1
            
            # Quick check before expensive optimization
            res = tensor_residual(U, V, W)
            if res < 1e-4:
                if verify_solution(U, V, W, 1e-6):
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = target_rank
                    return True
                U_r = round_to_half(U)
                V_r = round_to_half(V)
                W_r = round_to_half(W)
                if verify_solution(U_r, V_r, W_r, 1e-6):
                    best = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    best_rank = target_rank
                    return True
            
            # Use ALS optimization
            U_opt, V_opt, W_opt, res_opt = als_step(U, V, W, target_rank, n, m, p, max_iter=150)
            
            if res_opt < 1e-8:
                if verify_solution(U_opt, V_opt, W_opt, 1e-6):
                    best = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
                    best_rank = target_rank
                    return True
                else:
                    U_r = round_to_half(U_opt)
                    V_r = round_to_half(V_opt)
                    W_r = round_to_half(W_opt)
                    if verify_solution(U_r, V_r, W_r, 1e-6):
                        best = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        best_rank = target_rank
                        return True
        
        return False
    
    # Try ranks from low to high
    for target_rank in range(12, rank_sb):
        if time.time() > deadline - 30:
            break
        found = try_rank(target_rank, max_attempts=6)
        if found:
            # Found a solution, try to go lower if time permits
            if time.time() > deadline - 35:
                break
    
    return best
# EVOLVE-BLOCK-END
