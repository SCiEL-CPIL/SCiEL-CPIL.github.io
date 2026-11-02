"""Optimized program for 2x3 * 3x7 matrix multiplication tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START

def _to_int(x, tol=1e-9):
    """Round to integer if close enough, else return None."""
    ix = int(round(x))
    if abs(x - ix) < tol:
        return ix
    return None

def _to_half_int(x, tol=1e-9):
    """Round to half-integer (k/2) if close enough, else return None."""
    ix = int(round(2 * x))
    if abs(2 * x - ix) < tol:
        return ix
    return None

def _check_solution(U, V, W):
    """Check if U, V, W (as lists of lists) give a valid decomposition."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i,j,k), we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # This is a 3-way contraction. Let's check a few random entries.
    # Actually, we need to check ALL entries.
    rank = len(U[0])
    
    # Construct the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We can do this efficiently.
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    
    # Let's reshape and use einsum or explicit loops for small sizes.
    # n=2, m=3, p=7. Rank is at most 42.
    # Explicit loop is fine.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for c in range(rank):
                    s += U_arr[u_idx, c] * V_arr[v_idx, c] * W_arr[w_idx, c]
                if abs(s - 1.0) > 1e-7:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # 15s safety margin
    
    # 1. Start with Schoolbook (Rank 42)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Verify schoolbook
    if not _check_solution(U_sb, V_sb, W_sb):
        # Should not happen, but just in case
        pass
        
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # 2. Try to find a better decomposition using coordinate descent
    # We initialize with the schoolbook solution but zero out some terms
    # to encourage rank reduction, or just random initialization.
    # Actually, a common trick is to start with rank 42, but randomly set some
    # entries to 0 and see if we can optimize the rest to fill the gaps.
    # Or, better, use a smaller rank and initialize randomly or via projection.
    
    # Let's try ranks from 41 down to 32.
    # If we find a valid integer/half-integer solution, return it.
    
    # To speed up search, we can use numpy for the optimization steps.
    
    # Heuristic: 
    # The tensor is T = sum_c u_c \otimes v_c \otimes w_c.
    # We want to solve for U, V, W given a fixed rank.
    # This is a non-convex problem.
    # Coordinate descent: Fix V, W, update U. Then fix U, W, update V. Etc.
    # The update for U is a least squares problem:
    # T[i,j,k] = sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c])
    # This is not linear in U if V and W are fixed? No, it IS linear in U!
    # Let A[i*m+j, c] = U[i*m+j, c].
    # For a fixed (i,j,k), T[i,j,k] = sum_c A[i*m+j, c] * B_c where B_c = V[j*p+k, c]*W[k*n+i, c].
    # This couples different (i,j,k) terms for the same A row index?
    # A row index is (i,j). The equation for T[i,j,k] involves A[i*m+j, :].
    # So for a fixed (i,j), we have equations for all k:
    # T[i,j,k] = sum_c A[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c])
    # Let M_{i,j} be a (p x rank) matrix where M_{i,j}[k, c] = V[j*p+k, c] * W[k*n+i, c].
    # Then T_{i,j} (vector of length p) = M_{i,j} @ A[i*m+j, :].
    # So A[i*m+j, :] = pinv(M_{i,j}) @ T_{i,j}.
    # This gives the optimal U given V and W.
    # Similarly for V and W.
    
    def optimize(U, V, W, max_iters=200, tol=1e-8):
        """Run coordinate descent to minimize residual."""
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()
        
        n, m, p = DIMENSIONS
        rank = len(U[0])
        
        prev_res = float('inf')
        
        for it in range(max_iters):
            # Check time
            if time.time() > deadline:
                break
                
            # Update U
            for i in range(n):
                for j in range(m):
                    # Build M_{i,j}
                    M = np.zeros((p, rank))
                    for k in range(p):
                        v_row = V[j * p + k, :]
                        w_row = W[k * n + i, :]
                        M[k, :] = v_row * w_row
                    
                    # Target vector t
                    t = np.ones(p) # T[i,j,k] = 1 for all k
                    
                    # Solve M @ u = t
                    # Using least squares
                    try:
                        # If M is full column rank, use lstsq
                        u_sol = np.linalg.lstsq(M, t, rcond=None)[0]
                        U[i * m + j, :] = u_sol
                    except:
                        pass
                        
            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, rank))
                    for i in range(n):
                        u_row = U[i * m + j, :]
                        w_row = W[k * n + i, :]
                        M[i, :] = u_row * w_row
                    t = np.ones(n)
                    try:
                        v_sol = np.linalg.lstsq(M, t, rcond=None)[0]
                        V[j * p + k, :] = v_sol
                    except:
                        pass
                        
            # Update W
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        u_row = U[i * m + j, :]
                        v_row = V[j * p + k, :]
                        M[j, :] = u_row * v_row
                    t = np.ones(m)
                    try:
                        w_sol = np.linalg.lstsq(M, t, rcond=None)[0]
                        W[k * n + i, :] = w_sol
                    except:
                        pass
                        
            # Check residual
            res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        for c in range(rank):
                            s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                        res += (s - 1.0) ** 2
                        
            if abs(prev_res - res) < tol:
                break
            prev_res = res
            
        # Convert back to lists and check exactness
        U_list = [[_to_half_int(x) if _to_half_int(x) is not None else x for x in row] for row in U]
        V_list = [[_to_half_int(x) if _to_half_int(x) is not None else x for x in row] for row in V]
        W_list = [[_to_half_int(x) if _to_half_int(x) is not None else x for x in row] for row in W]
        
        # Check if integer/half-integer and valid
        valid = True
        for matrix in [U_list, V_list, W_list]:
            for row in matrix:
                for val in row:
                    if not isinstance(val, int):
                        # It's a float, check if it's close to half-integer
                        if _to_half_int(val) is None:
                            valid = False
                            break
                if not valid:
                    break
            if not valid:
                break
                
        if valid and _check_solution(U_list, V_list, W_list):
            return U_list, V_list, W_list, res
        else:
            return None, None, None, res

    # Try ranks 41 down to 32
    # We need a good initialization.
    # Random initialization often fails.
    # Let's try initializing from the schoolbook solution but "compressing" it.
    # Or just use random initialization with multiple restarts.
    
    # Given the time limit, we can afford a few hundred iterations of CD per rank.
    # Let's try a few random restarts for each rank.
    
    num_restarts = 5
    rng = np.random.default_rng(42)
    
    for rank in range(41, 31, -1):
        if time.time() > deadline:
            break
            
        for r in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize
            U = rng.random((n * m, rank))
            V = rng.random((m * p, rank))
            W = rng.random((p * n, rank))
            
            # Optional: Scale initialization?
            # The tensor entries are 1.
            # Product of three random numbers is small.
            # Let's scale by some factor to make products closer to 1.
            # E[U*V*W] = 1/8 * scale^3. We want this to be ~1. So scale ~ 2.
            scale = 2.0
            U = U * scale
            V = V * scale
            W = W * scale
            
            U_opt, V_opt, W_opt, res = optimize(U, V, W, max_iters=100)
            
            if U_opt is not None:
                return {"U": U_opt, "V": V_opt, "W": W_opt}
                
    # If nothing found, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
