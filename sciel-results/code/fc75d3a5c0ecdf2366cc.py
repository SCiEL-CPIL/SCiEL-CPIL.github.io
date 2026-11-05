import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # We need to decompose the tensor T[i,j,k] = 1 for all i,j,k
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Strategy: Use the slice rank method / partitioning approach
    # The naive rank is n*m*p = 3*3*7 = 63
    # We can try to find a lower rank decomposition.
    
    # Let's first establish a valid baseline with rank 63
    # Then try to reduce it.
    
    # Actually, let's try a constructive approach for small dimensions.
    # For 3x3x3, the Strassen rank is 7 (vs naive 27).
    # For 3x3x7, we can think of it as 7 slices of 3x3.
    # But the tensor is dense: T[i,j,k]=1 for all i,j,k.
    
    # Wait, the tensor for matrix multiplication C = A*B where A is n x m and B is m x p
    # has entries T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
    # Actually, the standard matrix multiplication tensor T_{ijk}^{(n,m,p)} has T_{ijk}=1 
    # for all valid i,j,k indices because C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor is indeed all-ones over the index set {0..n-1} x {0..m-1} x {0..p-1}.
    
    # The rank of the all-ones tensor J_{n,m,p} is a known problem.
    # For J_{3,3,7}, what is the rank?
    
    # A simple upper bound: rank <= n*m*p = 63.
    # We can also use: rank <= n*m + n*p + m*p - 2? No, that's not a general formula.
    # 
    # Let's try to construct a decomposition.
    # 
    # Approach 1: Use the fact that the all-ones tensor can be decomposed.
    # For J_{n,m,p}, one can use the "border rank" or "tensor rank" results.
    # 
    # Let's try a greedy/optimization approach to find a low-rank decomposition.
    # We'll use random initialization and alternating least squares (ALS) 
    # or a similar nonlinear optimization method.
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # First, establish the trivial decomposition
    U_triv = np.zeros((n*m, best_rank))
    V_triv = np.zeros((m*p, best_rank))
    W_triv = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i*m + j, col] = 1.0
                V_triv[j*p + k, col] = 1.0
                W_triv[k*n + i, col] = 1.0
                col += 1
    best_U = U_triv
    best_V = V_triv
    best_W = W_triv
    
    # Now try to find a lower rank decomposition
    # We'll try ranks from 62 down to some minimum
    # For each rank, try multiple random initializations
    
    def validate(U, V, W, rank):
        """Check if the decomposition is valid."""
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need this to equal 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True
    
    def try_rank(rank, num_attempts=50):
        """Try to find a valid decomposition of given rank."""
        for attempt in range(num_attempts):
            np.random.seed(42 + attempt * 1000 + rank)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Alternating minimization
            converged = False
            for it in range(200):
                # Update U: for each (i,j), solve linear system
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # This is: U[i*m+j, :] * M_k = 1 where M_k[r] = V[j*p+k, r] * W[k*n+i, r]
                # For fixed (i,j), we have p equations (one for each k) and rank unknowns
                # If rank > p, we can find a minimum-norm solution
                # If rank <= p, we might have an underdetermined system
                
                new_U = U.copy()
                for i in range(n):
                    for j in range(m):
                        # Build matrix A where A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        A = np.zeros((p, rank))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve A @ new_U[i*m+j, :] = ones(p)
                        # Minimum norm solution: x = A^T (A A^T)^{-1} ones
                        # Or use lstsq
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            new_U[i*m+j, :] = sol
                        except:
                            pass
                
                # Similarly for V
                new_V = V.copy()
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, rank))
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            new_V[j*p+k, :] = sol
                        except:
                            pass
                
                # Similarly for W
                new_W = W.copy()
                for i in range(n):
                    for k in range(p):
                        A = np.zeros((m, rank))
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            new_W[k*n+i, :] = sol
                        except:
                            pass
                
                # Check convergence
                if np.max(np.abs(new_U - U)) < 1e-10 and \
                   np.max(np.abs(new_V - V)) < 1e-10 and \
                   np.max(np.abs(new_W - W)) < 1e-10:
                    U, V, W = new_U, new_V, new_W
                    break
                
                U, V, W = new_U, new_V, new_W
                
                # Check if we have a valid solution
                if validate(U, W, W, rank):  # Note: this is wrong, should be U,V,W
                    pass
                # Let's properly check
                if _is_valid(U, V, W, rank):
                    return U, V, W
            else:
                # Try final validation
                if _is_valid(U, V, W, rank):
                    return U, V, W
        
        return None, None, None
    
    def _is_valid(U, V, W, rank):
        """Check if the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True
    
    # Try to find lower rank decompositions
    for rank in range(best_rank - 1, 20, -1):
        result = try_rank(rank, num_attempts=20)
        if result[0] is not None:
            U, V, W = result
            best_rank = rank
            best_U = U
            best_V = V
            best_W = W
            break
    
    # Convert to lists for the output format
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    # Round to clean values if they're close to integers or half-integers
    def round_clean(matrix):
        result = []
        for row in matrix:
            clean_row = []
            for val in row:
                # Check if close to integer
                if abs(val - round(val)) < 1e-6:
                    clean_row.append(float(round(val)))
                # Check if close to half-integer
                elif abs(val - (round(val*2)/2)) < 1e-6:
                    clean_row.append(round(val*2)/2)
                else:
                    clean_row.append(val)
            result.append(clean_row)
        return result
    
    U_out = round_clean(U_out)
    V_out = round_clean(V_out)
    W_out = round_clean(W_out)
    
    return {"U": U_out, "V": V_out, "W": W_out}
