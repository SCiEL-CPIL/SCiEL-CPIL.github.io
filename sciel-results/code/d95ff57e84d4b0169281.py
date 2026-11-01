"""Matrix multiplication 2x3x4 rank-minimal exact tensor decomposition."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3x4 matrix multiplication.
    
    Returns dict with 'U', 'V', 'W' matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    The decomposition satisfies: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    and for all other index combinations, the sum equals 0.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 6, 12, 8
    
    # The tensor T[i,j,k] = 1 if (i,j,k) is a valid multiplication index, else 0
    # Actually, the matrix multiplication tensor is:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait, that's not right. Let me reconsider.
    
    # For matrix multiplication C = A * B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 for all valid (i,j,k)
    # A rank-r decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # (using the trace convention W[k*n+i, r])
    
    # So we need: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is the schoolbook decomposition with rank n*m*p = 24.
    
    # We want to find a lower rank decomposition.
    # The known lower bound for 2x3x4 is not trivially known to be less than 24.
    # Let's try to find if rank < 24 is possible.
    
    # Approach: Use integer programming / search for small rank decompositions.
    # For small ranks, we can enumerate or use optimization.
    
    # Let's start with the schoolbook solution as a baseline.
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank decompositions using optimization
    # Strategy: For a given rank r, we have r columns in U, V, W.
    # The constraint is that the tensor product sum equals the target tensor.
    # This is a system of nonlinear equations.
    
    # For small ranks, we can try random initialization + gradient descent
    # or use the fact that entries can be half-integers to simplify.
    
    # Let's try ranks from 23 down to some lower bound.
    # Lower bound: max(nm, mp, pn) = max(6, 12, 8) = 12, but this is very loose.
    # A better lower bound comes from the slice rank or other bounds.
    
    # For now, let's try to verify if rank 23 or lower works using a search.
    
    # Actually, let's use a different approach: 
    # The matrix multiplication tensor for (2,3,4) has a known structure.
    # Let's try to use the "border rank" or approximate methods, but we need exact.
    
    # Let me try a direct search for rank r where r < 24.
    # For each rank r, we need to solve:
    # For all (i,j,k): sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p = 24 equations with 3*r variables (if we fix one scale).
    # Actually, we have nm*r + mp*r + pn*r = (6+12+8)*r = 26*r variables.
    # And 24 equations. For r >= 2, we have more variables than equations, so solutions may exist.
    
    # But we need EXACT solutions with integer/half-integer entries.
    
    # Let's try a greedy/structured approach:
    # Start with the schoolbook and try to merge terms.
    
    # Alternative: Use the fact that for 2x2x2, rank is 7 (Strassen).
    # For 2x3x4, we can embed the 2x2x2 problem.
    
    # Let's try a numerical optimization approach for small ranks.
    
    def try_rank(r, max_iter=10000, tol=1e-10):
        """Try to find a rank-r decomposition using least squares."""
        if time.time() > deadline - 5:
            return None
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(nm, r) * 0.5
        V = np.random.randn(mp, r) * 0.5
        W = np.random.randn(pn, r) * 0.5
        
        # Target tensor values
        target = np.ones((n, m, p))
        
        # Define the residual
        def residual(params):
            U_f = params[:nm*r].reshape(nm, r)
            V_f = params[nm*r:nm*r+mp*r].reshape(mp, r)
            W_f = params[nm*r+mp*r:nm*r+mp*r+pn*r].reshape(pn, r)
            # Compute tensor product sum
            # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            T = np.zeros((n, m, p))
            for c in range(r):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i,j,k] += U_f[i*m+j, c] * V_f[j*p+k, c] * W_f[k*n+i, c]
            return (T - target).flatten()
        
        # Use scipy optimize
        from scipy.optimize import least_squares
        
        params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = least_squares(residual, params0, method='lm', max_nfev=5000, xtol=1e-12)
        except Exception:
            return None
        
        if result.cost < tol:
            # Check if entries are close to integers or half-integers
            U_f = result.x[:nm*r].reshape(nm, r)
            V_f = result.x[nm*r:nm*r+mp*r].reshape(mp, r)
            W_f = result.x[nm*r+mp*r:nm*r+mp*r+pn*r].reshape(pn, r)
            
            # Round to nearest half-integer
            U_round = np.round(U_f * 2) / 2
            V_round = np.round(V_f * 2) / 2
            W_round = np.round(W_f * 2) / 2
            
            # Verify the rounded solution
            T = np.zeros((n, m, p))
            for c in range(r):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i,j,k] += U_round[i*m+j, c] * V_round[j*p+k, c] * W_round[k*n+i, c]
            
            if np.allclose(T, target, atol=1e-10):
                return (U_round, V_round, W_round)
        
        return None
    
    # Try ranks from 23 down to 12
    for r in range(rank_schoolbook - 1, 11, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r)
        if result is not None:
            U_f, V_f, W_f = result
            best_rank = r
            best_U, best_V, best_W = U_f, V_f, W_f
            break  # We found the lowest rank, stop
    
    # If no lower rank found, use schoolbook
    if best_rank == rank_schoolbook:
        # Return schoolbook as list of lists
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
    else:
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
