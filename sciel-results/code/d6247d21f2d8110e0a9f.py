import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Build the tensor T
    # T[a,b,c] where a in [0, n*m), b in [0, m*p), c in [0, p*n)
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    target_rank = 29
    rank = target_rank
    
    def verify_decomposition(U, V, W):
        """
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        Returns True if decomposition is exact (within float precision, assuming half-integers)
        """
        T_hat = np.einsum('ar,br,cr->abc', U, V, W)
        diff = np.abs(T_hat - T)
        if np.max(diff) < 1e-9:
            return True
        return False

    def round_to_half_int(M):
        """Round matrix entries to nearest 0.5"""
        return np.round(M * 2) / 2

    def check_half_integers(U, V, W, tol=1e-5):
        """Check if all entries are close to half-integers"""
        for M in [U, V, W]:
            # Distance to nearest half-integer
            dist = np.abs(M - np.round(M * 2) / 2)
            if np.max(dist) > tol:
                return False
        return True

    def solve_compensation(removed_idx, seed=0):
        """
        Try to find a rank 29 decomposition by initializing with schoolbook terms 
        excluding the one at 'removed_idx' and optimizing to match T.
        """
        np.random.seed(seed)
        
        I, J, K = T.shape
        # Initialize U, V, W with schoolbook terms except removed_idx
        # Schoolbook terms are indexed by (i, j, k)
        # We need to map removed_idx (0..29) to (i, j, k)
        
        # List of all (i, j, k) tuples
        indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    indices.append((i, j, k))
        
        # The removed index corresponds to indices[removed_idx]
        # We use all other indices for initialization
        
        U_init = np.zeros((I, rank))
        V_init = np.zeros((J, rank))
        W_init = np.zeros((K, rank))
        
        col = 0
        for idx, (i, j, k) in enumerate(indices):
            if idx == removed_idx:
                continue
            a = i * m + j
            b = j * p + k
            c = k * n + i
            U_init[a, col] = 1.0
            V_init[b, col] = 1.0
            W_init[c, col] = 1.0
            col += 1
            
        # Flatten parameters
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def objective(x):
            U = x[:I*rank].reshape(I, rank)
            V = x[I*rank:I*rank+J*rank].reshape(J, rank)
            W = x[I*rank+J*rank:].reshape(K, rank)
            
            # Compute T_hat
            # T_hat[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            T_hat = np.einsum('ar,br,cr->abc', U, V, W)
            # Residual
            res = (T_hat - T).flatten()
            return res
        
        def jac(x):
            # Numerical Jacobian is too slow. Let's use finite differences or analytic?
            # Analytic Jacobian for einsum is complex but doable.
            # However, least_squares with method='trf' or 'dogbox' can use finite differences.
            # Given 899 vars, finite differences is 899 evaluations.
            # Each evaluation is einsum.
            # This might be slow but feasible within 140s for a few tries.
            pass
            
        try:
            # Use least_squares
            # 'trf' (Trust Region Reflective) is good for large problems
            result = scipy.optimize.least_squares(objective, x0, method='trf', max_nfev=1000)
            
            if result.success or result.cost < 1e-8:
                U_sol = result.x[:I*rank].reshape(I, rank)
                V_sol = result.x[I*rank:I*rank+J*rank].reshape(J, rank)
                W_sol = result.x[I*rank+J*rank:].reshape(K, rank)
                
                # Check if close to half-integers
                if check_half_integers(U_sol, V_sol, W_sol, tol=1e-4):
                    U_rounded = round_to_half_int(U_sol)
                    V_rounded = round_to_half_int(V_sol)
                    W_rounded = round_to_half_int(W_sol)
                    
                    if verify_decomposition(U_rounded, V_rounded, W_rounded):
                        return U_rounded, V_rounded, W_rounded
                else:
                    # Maybe it's close to integers?
                    U_int = np.round(U_sol)
                    V_int = np.round(V_sol)
                    W_int = np.round(W_sol)
                    if verify_decomposition(U_int, V_int, W_int):
                        return U_int, V_int, W_int
        except Exception as e:
            pass
            
        return None

    import scipy.optimize

    best_result = None
    
    # Try removing each of the 30 terms
    for removed_idx in range(30):
        if time.time() > deadline - 5:
            break
            
        sol = solve_compensation(removed_idx, seed=removed_idx)
        if sol is not None:
            U, V, W = sol
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
            
    if best_result is not None:
        return best_result
        
    # If compensation fails, try random ALS with better initialization
    # Or just fallback to schoolbook
    
    # Fall back to schoolbook rank-30
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
# EVOLVE-BLOCK-END
