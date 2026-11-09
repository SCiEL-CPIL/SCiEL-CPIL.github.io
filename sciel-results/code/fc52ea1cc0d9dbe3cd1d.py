"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 63
    
    # Build the matrix multiplication tensor T[i, j, k] = 1 for all i,j,k
    # We want to find rank-r decomposition: T = sum_r U[:,r] * V[:,r] * W[:,r]
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # and T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For a 3x3x3 matrix multiplication, the tensor is just the identity-like tensor
    # T[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T[i,j,k] has entries: T[i,j,k] = 1 for all i,j,k in {0,1,2}
    # Wait, no. The matrix multiplication tensor is T[i,j,k] where the multiplication is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor for the "multiplication" structure)
    # Actually, the matrix multiplication tensor M_{n,p,q} has entries:
    # M_{i,j,k} = 1 if we're multiplying an (n x m) matrix by an (m x p) matrix to get (n x p)
    # The indices are: i in [0,n), j in [0,m), k in [0,p)
    # And the tensor entry T[i,j,k] = 1 for all valid i,j,k
    # The decomposition says: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # For 3x3x3, this is the standard matrix multiplication tensor
    # Let's try to use known low-rank decompositions
    
    # First, let's verify the schoolbook works
    def make_schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank
    
    # Check if time allows for search
    elapsed = time.time() - start_time
    if elapsed > 10:
        U, V, W, rank = make_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    # Try to find a better decomposition using randomized/local search
    # The key insight: for 3x3x7, we can view this as 7 separate 3x3x1 multiplications
    # or as a 3x3 matrix times a 3x7 matrix
    
    # Let's try a block approach:
    # C = A * B where A is 3x3, B is 3x7
    # We can write B = [B1, B2, B3, B4, B5, B6, B7] where each Bi is 3x1
    # C = [A*B1, A*B2, ..., A*B7]
    # Each A*Bi is a 3x3 * 3x1 = 3x1 multiplication
    # The tensor for 3x3*3x1 has entries T[i,j,k] for i in [0,3), j in [0,3), k in [0,1)
    # This is just a 3x3 matrix times a vector, which has rank 3 (one per output)
    # Actually for 3x3*3x1, the "tensor" is simpler
    
    # Let me think differently. The full tensor for 3x3*3x7:
    # T[i, j, k] where i in [0,3) (row of C), j in [0,3) (column of A / row of B), k in [0,7) (column of C)
    # T[i,j,k] = 1 for all i,j,k
    # We need: sum_r U[i*3+j, r] * V[j*7+k, r] * W[k*3+i, r] = 1 for all i,j,k
    
    # This is a 3x3x7 all-ones tensor problem in the matrix multiplication context
    
    # Let's try to use a structured approach with fewer terms
    # One idea: use the fact that we can share computations across the 7 columns
    
    # Try rank = 49 (7 * 7, using rank-7 for each 3x3 block if we had them, but 3x3 needs rank >= 19 or so)
    # Actually, let's just try random search with small ranks
    
    import numpy as np
    from scipy.optimize import minimize
    
    def tensor_residual(U, V, W, n, m, p):
        """Compute max absolute deviation from all-ones tensor."""
        r = len(U[0])
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    max_err = max(max_err, abs(s - 1.0))
        return max_err
    
    # Try a greedy/additive approach
    # Start with schoolbook and try to reduce
    
    # Actually, let's try a different strategy: use known results
    # For M(3,3,3), best known is rank 19 (or 23 depending on what's proven)
    # But we need integer/half-integer entries
    
    # Let me try a simple local search from schoolbook
    # Replace pairs of terms with a single term if possible
    
    U, V, W, rank = make_schoolbook(n, m, p)
    
    # Try to find a lower-rank decomposition via random initialization + gradient-like search
    # This is tricky with integer/half-integer constraints
    
    # Alternative: try specific structured decompositions
    # For 3x3x7, one approach is to use Kronecker/product structures
    
    # Let's try: can we do it in rank 42? (6 * 7)
    # Or rank 35? (5 * 7)
    
    # Given time constraints, let's try a few random rank values and see if we can find valid ones
    # using a simple iterative method
    
    def try_rank(r, n, m, p, max_iters=10000, seed=42):
        """Try to find a rank-r decomposition."""
        np.random.seed(seed)
        
        # Random initialization
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        for it in range(max_iters):
            if time.time() - start_time > deadline:
                return None
            
            # Compute gradient with respect to U, V, W
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # d/dU[i*m+j, c] of (sum_c' U[i*m+j,c']*V[j*p+k,c']*W[k*n+i,c'] - 1)^2
                        # = 2 * (sum_c' U[i*m+j,c']*V[j*p+k,c']*W[k*n+i,c'] - 1) * V[j*p+k,c] * W[k*n+i,c]
                        prod = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        err = 2.0 * (prod - 1.0)
                        dU[i*m+j] += err * V[j*p+k] * W[k*n+i]
                        dV[j*p+k] += err * U[i*m+j] * W[k*n+i]
                        dW[k*n+i] += err * U[i*m+j] * V[j*p+k]
            
            # Gradient step
            lr = 0.01
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Check convergence
            max_err = tensor_residual(U.tolist(), V.tolist(), W.tolist(), n, m, p)
            if max_err < 1e-10:
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                if tensor_residual(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p) < 1e-10:
                    return U_rounded, V_rounded, W_rounded
        
        return None
    
    # Try ranks from 35 to 62
    best_result = None
    for r in range(35, 63):
        if time.time() - start_time > deadline - 10:
            break
        
        result = try_rank(r, n, m, p, max_iters=5000)
        if result is not None:
            U_r, V_r, W_r = result
            best_result = (U_r.tolist(), V_r.tolist(), W_r.tolist(), r)
            break  # Found a valid one, use the smallest rank found
    
    if best_result is not None:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    # Fall back to schoolbook
    U, V, W, rank = make_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
