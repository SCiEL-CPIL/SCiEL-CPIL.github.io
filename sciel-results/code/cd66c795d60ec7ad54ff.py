"""Find rank-minimal exact tensor decomposition for 2x5x6 matrix multiplication."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Build the matrix multiplication tensor T[i,j,k]
    # T is a 3D array where T[i,j,k] = 1 if this term contributes, 0 otherwise
    # Actually, the tensor is: T_{i,j,k} contributes to C[i,k] = sum_j A[i,j] B[j,k]
    # In the decomposition: C[i,k] = sum_c (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k]) * W[k*n+i, c]
    # Wait, that's not right. Let me reconsider.
    
    # The tensor for matrix multiplication: we want C = A*B where A is n×m, B is m×p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The bilinear form: for each (i,j,k), we have a term A[i,j]*B[j,k] that contributes to C[i,k]
    # In the decomposition with rank r:
    # C[i,k] = sum_{c=1}^{r} (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    # Wait, this doesn't look right either. Let me re-examine.
    
    # Actually, the standard decomposition is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Each term A[i,j] B[j,k] is a rank-1 contribution.
    # In the tensor decomposition framework:
    # The tensor T has entries T[i,j,k] which is a vector in R^{n*m*p}
    # For the (i,j,k) slice, T[i,j,k] has a 1 at position (i*m+j) in U, (j*p+k) in V, (k*n+i) in W
    
    # Let me use the convention from the task:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition: for each c, U[:,c], V[:,c], W[:,c]
    # The product: sum_c (U^T A_flat) * (V^T B_flat) * W^T C_flat = 0 for the tensor to be exact
    # Actually, the condition is:
    # For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,j,k} (the tensor entry)
    # where delta_{i,j,k} = 1 if this is a valid term in the multiplication, 0 otherwise
    # But in matrix multiplication, every (i,j,k) term is valid, so delta_{i,j,k} = 1 for all i,j,k
    # Wait, that's not right. The tensor T_{i,j,k} is defined such that the bilinear form equals the matrix product.
    
    # Let me re-read the task. The tensor for matrix multiplication n×m×p:
    # T(i,j,k) = e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)} in the appropriate spaces
    # The condition for a rank-r decomposition is:
    # sum_{c=1}^{r} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # No wait, that's not right either. The tensor entries are not all 1.
    
    # Let me think again. The matrix multiplication tensor is:
    # T = sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}
    # where e_{(i,j)} is the basis vector for A[i,j], e_{(j,k)} for B[j,k], e_{(k,i)} for C[k,i]
    # Wait, the output is C[i,k], not C[k,i]. Let me use the convention W[k*n+i, r] as stated.
    
    # So the tensor is: T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}
    # A rank-r decomposition: T = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
    # where u_c is in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
    # The condition: for each (i,j,k):
    # sum_{c=1}^{r} u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1
    # And for other indices (i',j',k') that are not the same, the sum should be 0? No.
    
    # Actually, the tensor T has T[i,j,k] as a scalar (0 or 1), and the decomposition must satisfy:
    # sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = T[i,j,k]
    # where T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k], which is always true.
    # So T[i,j,k] = 1 for all i,j,k.
    # Wait, that can't be right because then the rank would be at least n*m*p.
    
    # I think I'm confusing myself. Let me look at this differently.
    # The matrix multiplication tensor is a 3-tensor T ∈ R^{(n*m) × (m*p) × (p*n)}
    # T[a,b,c] = 1 if a corresponds to (i,j), b corresponds to (j,k), c corresponds to (k,i) for some i,j,k
    # More precisely, T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    # And T[a,b,c] = 0 otherwise.
    # The rank of this tensor is what we want to minimize.
    
    # For the schoolbook decomposition, each (i,j,k) gives a rank-1 term, so rank = n*m*p.
    # We want to find a lower rank decomposition.
    
    # Let me build the tensor and try to find a low-rank decomposition.
    
    def build_tensor(n, m, p):
        """Build the matrix multiplication tensor as a 3D array."""
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1
        return T
    
    T = build_tensor(n, m, p)
    
    # Try to find a rank-r decomposition for decreasing r
    # For a given r, we need to find U (n*m × r), V (m*p × r), W (p*n × r)
    # such that T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # i.e., T[a,b,c] = sum_{c=1}^{r} U[a,c] * V[b,c] * W[c_idx,c] (using different index names)
    
    # This is a non-convex optimization problem. Let me try a greedy approach.
    # Start with the schoolbook decomposition and try to merge terms.
    
    # Actually, let me try a different approach: use the fact that for n=2, we can use
    # the structure to find a better decomposition.
    
    # For n=2, the tensor has a special structure. Let me try to find a decomposition
    # by fixing the W factor and solving for U and V.
    
    # Alternative: use the "rank profile" approach or try to find a decomposition
    # based on known results for small matrix multiplication.
    
    # Let me try a computational search with a time limit.
    # I'll use a greedy algorithm: start with all (i,j,k) terms and try to merge them.
    
    # Actually, let me try a more direct approach: for small r, use a random initialization
    # and gradient descent to minimize the error.
    
    def try_decomposition(r, T, n, m, p, max_iter=1000, seed=42):
        """Try to find a rank-r decomposition of T using iterative methods."""
        rng = np.random.RandomState(seed)
        U = rng.randn(n*m, r)
        V = rng.randn(m*p, r)
        W = rng.randn(p*n, r)
        
        # Normalize
        for c in range(r):
            norm = max(np.linalg.norm(U[:,c]), np.linalg.norm(V[:,c]), np.linalg.norm(W[:,c]), 1e-10)
            U[:,c] /= norm
            V[:,c] /= norm
            W[:,c] /= norm
        
        best_error = np.inf
        for it in range(max_iter):
            # Compute current tensor
            T_current = np.einsum('ac,bc,dc->abd', U, V, W)
            error = np.linalg.norm(T_current - T)
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Gradient descent step
            # dT/dU[a,c] = sum_{b,d} T[a,b,d] * V[b,c] * W[d,c]
            # This is expensive, so let's use a simpler update.
            
            # Update U: U[a,c] += lr * (T[a, :, :] - T_current[a, :, :]) contracted with V and W
            # Actually, the gradient of the squared error with respect to U[a,c] is:
            # 2 * sum_{b,d} (T_current[a,b,d] - T[a,b,d]) * V[b,c] * W[d,c]
            
            lr = 0.01
            for c in range(r):
                # Gradient for U[:,c]
                grad_U = np.einsum('abd,bc,dc->a', T_current - T, V[:,c], W[:,c])
                U[:,c] -= lr * grad_U
                
                # Gradient for V[:,c]
                grad_V = np.einsum('abd,ac,dc->b', T_current - T, U[:,c], W[:,c])
                V[:,c] -= lr * grad_V
                
                # Gradient for W[:,c]
                grad_W = np.einsum('abd,ac,bc->d', T_current - T, U[:,c], V[:,c])
                W[:,c] -= lr * grad_W
            
            # Renormalize
            for c in range(r):
                norm = max(np.linalg.norm(U[:,c]), np.linalg.norm(V[:,c]), np.linalg.norm(W[:,c]), 1e-10)
                U[:,c] /= norm
                V[:,c] /= norm
                W[:,c] /= norm
        
        return best_error, best_U, best_V, best_W
    
    # Try ranks from a reasonable upper bound down to a lower bound
    # The rank must be at least max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # And at most n*m*p = 60
    
    # Let me try a few ranks and see if we can find a valid decomposition.
    # Due to time constraints, I'll try a limited search.
    
    best_rank = n * m * p  # Default to schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Try a few candidate ranks
    for r in range(35, 61):  # Try ranks from 35 to 60
        if time.time() > deadline:
            break
        if r >= best_rank:
            continue
        
        # Try a few seeds
        found = False
        for seed in range(10):
            if time.time() > deadline:
                break
            error, U, V, W = try_decomposition(r, T, n, m, p, max_iter=500, seed=seed)
            if error < 1e-6:
                best_rank = r
                best_U, best_V, best_W = U, V, W
                found = True
                break
        if found:
            # Try to go lower
            continue
        else:
            # If we can't find a decomposition at this rank, stop
            break
    
    # If we didn't find a better decomposition, use schoolbook
    if best_U is None:
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END

