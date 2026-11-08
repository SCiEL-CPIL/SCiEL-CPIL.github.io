"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 20s safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 50
    
    # The bilinear tensor T[i,j,k] = delta(i,j,k) for matrix multiplication
    # We need U, V, W such that for all i,j,k:
    #   T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # 
    # For 2x5x5 matrix multiplication, we search for rank-minimal decomposition.
    # Known lower bound: rank >= max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # Upper bound: schoolbook = 50
    # 
    # Strategy: Use a randomized algebraic approach.
    # We'll try to find a decomposition of rank r by solving a system of equations.
    # 
    # Actually, let me think about this more carefully.
    # The matrix multiplication tensor M_{n,p,m} has entries indexed by (i,j,k) 
    # where i in [0,n), j in [0,m), k in [0,p).
    # T[i,j,k] = 1 if the (i,j) entry of the product involves A[i,j]*B[j,k],
    # which is always the case, so T[i,j,k] = 1 for all valid (i,j,k).
    # Wait, that's not right. Let me reconsider.
    #
    # For matrix multiplication C = A * B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The bilinear form is: C[i,k] = sum_j (A[i,j] * B[j,k])
    # So the tensor has T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in 
    # the bilinear sense), but actually the tensor is defined by:
    # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    # 
    # Actually, the standard formulation: the multiplication tensor M_{n,p,m} 
    # is in (F^n x F^p)^* x (F^m)^* x (F^m)^* or similar.
    # 
    # Let me use the convention from the problem:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition means:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   delta(i,j,k) = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where delta(i,j,k) = 1 (since every (i,j,k) triple contributes to the product)
    #
    # Wait, that's the all-ones tensor. That seems too simple. Let me re-examine.
    #
    # Actually no. The matrix multiplication tensor is NOT the all-ones tensor.
    # The correct formulation:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a bilinear map from (A, B) to C.
    # In tensor form, if we flatten A to index (i,j) and B to index (j,k),
    # then the output C[i,k] depends on the sum over j.
    #
    # The tensor T is defined such that:
    # <T, A \otimes B> = C (flattened)
    # Or more precisely, the bilinear form is:
    # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # where T[i,j,k] = 1 for all i,j,k.
    #
    # So yes, the tensor IS the all-ones tensor in this indexing!
    # The constraint is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  for all i,j,k
    #
    # This is a system of n*m*p = 2*5*5 = 50 equations.
    # We want to minimize rank (number of columns).
    
    # Let's verify: for the schoolbook decomposition with rank = n*m*p = 50,
    # each term c corresponds to a unique (i,j,k) triple, and
    # U[i*m+j, c] = 1 if c = (i,j,k) else 0
    # V[j*p+k, c] = 1 if c = (i,j,k) else 0  
    # W[k*n+i, c] = 1 if c = (i,j,k) else 0
    # Then sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 (only c=(i,j,k) contributes)
    
    # So we need to find rank-minimal U, V, W with integer/half-integer entries
    # satisfying: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is a hard problem. Let's try a structured approach.
    # 
    # One idea: use the fact that this is related to matrix multiplication complexity.
    # For 2x5x5, the known rank is not trivially determined.
    # 
    # Let's try a different approach: assume a low-rank structure.
    # For example, try rank = 25 (the lower bound) and see if we can find a solution.
    #
    # Actually, let me try a constructive approach.
    # 
    # Consider: if we set W[k*n+i, c] = 1 for all k,i,c (constant), then
    # sum_c U[i*m+j,c] * V[j*p+k,c] = 1 for all i,j,k
    # This means the matrix product of U^T (n*m x rank) and V (rank x m*p) 
    # should give something specific... but this doesn't quite work because
    # the sum is over c and we need it to equal 1 for each (i,j,k).
    #
    # Let me try yet another approach. Since the problem is small (50 equations),
    # let's try to find a solution with rank r by random initialization and 
    # gradient descent or Newton's method, then round to half-integers.
    
    best_rank = rank_schoolbook
    best_solution = None
    
    # First, let's just return the schoolbook solution as a fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    best_solution = (U_sb, V_sb, W_sb)
    
    # Try to find lower rank solutions
    # Use random initialization and try to solve the system
    # The system is: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    def check_solution(U, V, W, tol=1e-6):
        """Check if the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_rank(r, max_iters=1000, seed=42):
        """Try to find a rank-r decomposition using random init + gradient descent."""
        if time.time() > deadline - 10:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        U = rng.randn(n*m, r) * 0.1
        V = rng.randn(m*p, r) * 0.1
        W = rng.randn(p*n, r) * 0.1
        
        # Use a simple gradient descent on the squared error
        lr = 0.01
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = U[i*m+j, :]
                        v = V[j*p+k, :]
                        w = W[k*n+i, :]
                        val = np.sum(u * v * w)
                        err = val - 1.0
                        # d/dU[i*m+j,c] = err * V[j*p+k,c] * W[k*n+i,c]
                        dU[i*m+j, :] += err * v * w
                        dV[j*p+k, :] += err * u * w
                        dW[k*n+i, :] += err * u * v
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if it % 100 == 0:
                # Check convergence
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            max_err = max(max_err, abs(val - 1.0))
                if max_err < 1e-8:
                    break
        
        # Check if we found a valid solution
        if check_solution(U, V, W, tol=1e-4):
            # Try to round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            if check_solution(U_rounded, V_rounded, W_rounded, tol=1e-6):
                return (U_rounded, V_rounded, W_rounded)
            else:
                # Try to find a nearby half-integer solution
                # Use the continuous solution as a starting point
                # and try local search
                for _ in range(50):
                    if time.time() > deadline - 10:
                        break
                    U_try = U_rounded + rng.randn(n*m, r) * 0.01
                    V_try = V_rounded + rng.randn(m*p, r) * 0.01
                    W_try = W_rounded + rng.randn(p*n, r) * 0.01
                    U_try = np.round(U_try * 2) / 2
                    V_try = np.round(V_try * 2) / 2
                    W_try = np.round(W_try * 2) / 2
                    if check_solution(U_try, V_try, W_try, tol=1e-6):
                        return (U_try, V_try, W_try)
        
        return None
    
    # Try various ranks from the lower bound upward
    for r in range(25, rank_schoolbook + 1):
        if time.time() > deadline - 15:
            break
        
        found = False
        for seed in range(20):
            if time.time() > deadline - 15:
                break
            sol = try_rank(r, max_iters=500, seed=seed)
            if sol is not None:
                found = True
                if r < best_rank:
                    best_rank = r
                    best_solution = sol
                break
        
        if found:
            # Found a solution at rank r, no need to try higher ranks
            break
    
    U, V, W = best_solution
    
    # Convert to lists of lists as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Ensure entries are exactly representable (integers or half-integers)
    # Round to nearest half-integer to be safe
    U_list = [[round(x * 2) / 2 for x in row] for row in U_list]
    V_list = [[round(x * 2) / 2 for x in row] for row in V_list]
    W_list = [[round(x * 2) / 2 for x in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
