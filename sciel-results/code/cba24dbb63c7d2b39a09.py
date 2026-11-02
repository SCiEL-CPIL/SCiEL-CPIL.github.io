"""Search for a low-rank exact tensor decomposition of 3x4 * 4x6 matrix multiplication."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Build the multiplication tensor T[i,j,k] for (i in n, j in m, k in p)
    # T is rank r if T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We check by building both sides for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 1.0  # The multiplication tensor has T[i,j,k]=1 for all valid (i,j,k)
                rhs = 0.0
                for c in range(rank):
                    rhs += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(lhs - rhs) > 1e-9:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, rank):
    """Faster verification using numpy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # Compute sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] for each (i,j,k)
    # This is a tensor contraction
    # result[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # We can do this by: for each c, outer product contribution
    # Or use einsum
    # result[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # This is tricky with einsum because of the index cycling
    # Let's do it directly
    result = np.zeros((n, m, p))
    for c in range(rank):
        # U3[:, :, c] is (n, m), V3[:, :, c] is (m, p), W3[:, :, c] is (p, n)
        # We need sum over i,j,k of... no, we need for each (i,j,k):
        # U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        # This is like a cyclic product
        # For fixed c, define A[i,j] = U3[i,j,c], B[j,k] = V3[j,k,c], C[k,i] = W3[k,i,c]
        # Then contribution to (i,j,k) is A[i,j]*B[j,k]*C[k,i]
        # We can compute this as: for each (i,j), A[i,j] * (B * C^T)[j, ?]... 
        # Actually, let's just do it with a loop over one index for speed
        # Or use the fact that for fixed i: sum_k B[j,k]*C[k,i] = (B @ C)[j,i]
        # So contribution[i,j,k] = A[i,j] * B[j,k] * C[k,i]
        # Hmm, this doesn't simplify to a simple matrix product for all three at once.
        # Let's just loop over k for each c
        BC = V3[:, :, c] @ W3[:, :, c].T  # (m, p) @ (n, p).T -> (m, n)... no
        # W3[:, :, c] is (p, n), so W3.T is (n, p)
        # V3[:, :, c] is (m, p)
        # V3 @ W3.T gives (m, n) which is sum_k V[j,k]*W[k,i]... but we need W[k,i] where i is the first index of W
        # W3[k,i,c] so W3[:, :, c] has shape (p, n), indexing [k, i]
        # So W3[:, :, c].T has shape (n, p), indexing [i, k]
        # V3[:, :, c] has shape (m, p), indexing [j, k]
        # (V3 @ W3.T)[j, i] = sum_k V3[j,k,c] * W3[i,k,c]... but we want W3[k,i,c]
        # So this is not quite right. Let me just do it element-wise for clarity
        for i in range(n):
            for j in range(m):
                u_val = U3[i, j, c]
                if u_val == 0:
                    continue
                for k in range(p):
                    result[i, j, k] += u_val * V3[j, k, c] * W3[k, i, c]
    return np.allclose(result, np.ones((n, m, p)), atol=1e-9)

def construct():
    start_time = time.time()
    deadline = 130.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find lower rank decompositions
    # The theoretical lower bound for matrix multiplication rank is complex.
    # For 3x4 * 4x6, let's try ranks below schoolbook.
    
    # Strategy: Use random initialization and gradient-free optimization (e.g., coordinate descent)
    # to find a valid decomposition at a given rank.
    
    def try_rank(r, max_iterations=5000, seed=42):
        """Try to find a valid decomposition of rank r."""
        np.random.seed(seed)
        
        # Initialize with random small values
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        # We want sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
        # This is a system of n*m*p equations with (n*m + m*p + p*n)*r unknowns.
        
        # Use a simple iterative approach: for each (i,j,k), adjust the variables to reduce error.
        # Or use least squares / Newton-like methods.
        
        # Let's try a simple approach: treat it as a constraint satisfaction problem.
        # For each c, we have a "term" that contributes to the tensor.
        # The tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # We can use the fact that this is equivalent to:
        # For each c, define a matrix product contribution.
        # Actually, let's use a different approach: random search with refinement.
        
        # Initialize U, V, W randomly
        # Then for each (i,j,k), compute the current value and adjust one of the three factors.
        
        # Better approach: Use the structure. For a fixed c, the term is:
        # U_c[i,j] * V_c[j,k] * W_c[k,i] where U_c = U[:,c], V_c = V[:,c], W_c = W[:,c]
        # reshaped appropriately.
        
        # Let's use a coordinate descent approach.
        # Variables: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # Objective: minimize sum_{i,j,k} (1 - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])^2
        
        # For efficiency, let's precompute partial sums.
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                return None
            
            # Compute current errors
            # For each (i,j,k), error = 1 - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            
            # To speed up, we can update one variable at a time.
            # Let's cycle through all variables.
            
            # Update U[i*m+j, c] for random (i,j,c)
            idx = np.random.randint(0, n*m)
            i, j = divmod(idx, m)
            c = np.random.randint(0, r)
            
            # Current contribution of U[i*m+j, c] to the objective
            # We want to find the optimal value of U[i*m+j, c] given all other variables fixed.
            # Let x = U[i*m+j, c]
            # The terms involving x are for all k: (1 - x*V[j*p+k,c]*W[k*n+i,c] - S_other[i,j,k])^2
            # where S_other is the sum over c' != c
            # This is quadratic in x, so we can solve analytically.
            
            # Let a_k = V[j*p+k, c] * W[k*n+i, c]
            # Let b_k = 1 - sum_{c' != c} U[i*m+j, c']*V[j*p+k, c']*W[k*n+i, c']
            # We want to minimize sum_k (b_k - x*a_k)^2
            # Derivative: -2 * sum_k a_k * (b_k - x*a_k) = 0
            # x * sum_k a_k^2 = sum_k a_k * b_k
            # x = sum_k a_k * b_k / sum_k a_k^2
            
            a = np.array([V[j*p+k, c] * W[k*n+i, c] for k in range(p)])
            # b_k = 1 - (sum_{c' != c} U[i*m+j, c']*V[j*p+k, c']*W[k*n+i, c'])
            # = 1 - (sum_{c'} U[i*m+j, c']*V[j*p+k, c']*W[k*n+i, c'] - U[i*m+j, c]*V[j*p+k, c]*W[k*n+i, c])
            # = 1 - (current_total[i,j,k] - x_current * a_k)
            # where current_total[i,j,k] = sum_{c'} U[i*m+j, c']*V[j*p+k, c']*W[k*n+i, c']
            
            x_current = U[idx, c]
            # Compute current_total for this (i,j) and all k
            # current_total[k] = sum_{c'} U[i*m+j, c'] * V[j*p+k, c'] * W[k*n+i, c']
            current_total = np.zeros(p)
            for c_prime in range(r):
                if c_prime == c:
                    continue
                for k in range(p):
                    current_total[k] += U[i*m+j, c_prime] * V[j*p+k, c_prime] * W[k*n+i, c_prime]
            
            b = 1.0 - (current_total - x_current * a)
            
            denom = np.dot(a, a)
            if abs(denom) > 1e-15:
                U[idx, c] = np.dot(a, b) / denom
            else:
                # If a is zero, this variable doesn't matter, skip
                pass
        
        # Check if we've converged
        if time.time() > deadline:
            return None
        
        # Verify
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        result = np.zeros((n, m, p))
        for c in range(r):
            for i in range(n):
                for j in range(m):
                    u_val = U3[i, j, c]
                    if abs(u_val) < 1e-12:
                        continue
                    for k in range(p):
                        result[i, j, k] += u_val * V3[j, k, c] * W3[k, i, c]
        
        if np.allclose(result, np.ones((n, m, p)), atol=1e-8):
            return (U, V, W)
        
        return None
    
    # Try ranks from schoolbook-1 downward
    # But this is computationally expensive. Let's be smarter.
    
    # Known results: The rank of matrix multiplication for small sizes is known.
    # For 3x4 * 4x6, the exact rank is not immediately known to me, but it's likely less than 72.
    # Let's try a few specific ranks.
    
    # Actually, let's use a different approach: start from schoolbook and try to merge terms.
    # Or use the fact that we can try rank = n*m*p - 1, n*m*p - 2, etc.
    
    # Given the time constraint, let's try a limited search.
    
    # Try rank 71, 70, ... down to some minimum
    for r in range(rank_schoolbook - 1, max(rank_schoolbook - 10, 0), -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, max_iterations=2000, seed=r)
        if result is not None:
            U, V, W = result
            # Convert to list of lists with integer/half-integer entries if possible
            # For now, just return the numpy arrays converted to lists
            best_rank = r
            best_result = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            break
    
    # If we didn't find anything better, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
