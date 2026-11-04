"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x5 matrix.
    Uses HOOI (Higher-Order Orthogonal Iteration) to find low-rank decompositions,
    combined with structured initialization and local optimization.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # The tensor T[i,j,k] = 1 if we're doing standard matrix mult
    # Actually, for matrix multiplication A (n x m) * B (m x p) = C (n x p)
    # The tensor T[i,j,k] represents the coefficient of A[i,j] * B[j,k] in C[i,k]
    # So T[i,j,k] = 1 for all i,j,k (it's the identity-like tensor for matmul)
    # Wait, let me think more carefully.
    
    # The matrix multiplication tensor M has entries M[i,j,k] where:
    # i indexes rows of A (0..n-1), j indexes columns of A / rows of B (0..m-1), 
    # k indexes columns of B (0..p-1)
    # M[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # Actually, the standard matmul tensor is: T[i,j,k] = delta_{i,k} ... no.
    
    # Let me reconsider. The matmul tensor T of dimensions (n, m, p) is defined as:
    # T[i,j,k] = 1 if i is row of A, j is column of A = row of B, k is column of B
    # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, the problem says W[k*n+i, r], so W is indexed by (k, i) where k is col of B, i is row of A.
    
    # For standard matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] should satisfy: T[i,j,k] = 1 for all valid i,j,k
    # because each term A[i,j]*B[j,k] contributes exactly once to C[i,k].
    
    # Actually no. The tensor for matrix multiplication is:
    # T[i,j,k] = 1 if and only if the index j connects A's column j to B's row j
    # which is always true, so T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # But that's just an all-ones tensor, which has rank 1! That can't be right.
    
    # Let me re-read the problem. "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # So U has rows indexed by (i,j) pairs, V by (j,k) pairs, W by (k,i) pairs.
    # The decomposition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # For matrix multiplication, T[i,j,k] = 1 for all i,j,k? No, that's not right either.
    # 
    # Actually, I think the tensor is defined differently. Let me think about it as:
    # The bilinear map (A, B) -> C where C = A*B
    # This can be written as: C[i,k] = sum_{j,c} U[i,j,c] * A[i,j] * V[j,k,c] * B[j,k] * ... 
    # No, that's not standard.
    
    # Standard tensor decomposition for matmul:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We want: C[i,k] = sum_c (sum_j U[i,j,c] A[i,j]) (sum_j' V[j',k,c] B[j',k]) ... no.
    
    # Actually the correct formulation:
    # C[i,k] = sum_c (sum_j U[i,j,c] A[i,j]) * (sum_j V[j,k,c] B[j,k]) * W[k,i,c]
    # No, that doesn't work dimensionally.
    
    # Let me look at this differently. The tensor T[i,j,k] for matmul 4x4 * 4x5 is:
    # T[i,j,k] = 1 for all i in {0,1,2,3}, j in {0,1,2,3}, k in {0,1,2,3,4}
    # This is because each triple (i,j,k) corresponds to one multiplication A[i,j]*B[j,k]
    # which contributes to C[i,k].
    
    # If T is all ones, then rank is 1, which is clearly wrong for matmul.
    
    # I think I'm misunderstanding the tensor structure. Let me reconsider.
    
    # The matmul tensor is NOT all ones. The tensor T is defined such that:
    # sum_{j} A[i,j] B[j,k] = sum_{c} (sum_j U[i*m+j, c] A[i,j]) (sum_j V[j*p+k, c] B[j,k]) W[k*n+i, c]
    
    # Wait, that's not quite right either. Let me think about it as a 3-tensor.
    
    # The correct tensor for matrix multiplication M[n,m,p] has entries:
    # T[i,j,k] = 1 if and only if the indices form a valid "path" in the computation
    # For standard matmul, T[i,j,k] = 1 for all i,j,k because every A[i,j]*B[j,k] 
    # contributes to exactly one output C[i,k].
    
    # But if T is all-ones, the rank is 1. That contradicts known results.
    
    # I must be misunderstanding the tensor definition. Let me re-read the interface:
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "W[k*n+i, r]" - so W is indexed by (k, i)
    
    # The decomposition must satisfy:
    # For all i, j, k: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # What is T[i,j,k]? For matrix multiplication, I believe:
    # T[i,j,k] = 1 for all valid i, j, k
    # 
    # But wait, that gives rank 1. Unless... the tensor is NOT all ones.
    
    # Let me reconsider. Perhaps the tensor is defined as:
    # T[i,j,k] = delta_{i,k} ... no, that doesn't make sense for rectangular matrices.
    
    # Actually, I think I've been overthinking this. The standard result is that the
    # tensor for n x m * m x p matrix multiplication has entries T[i,j,k] = 1 for all
    # i, j, k, and its rank is the matmul rank. But an all-ones tensor has rank 1...
    
    # UNLESS the tensor is not simply all ones. Let me look at this from the decomposition
    # perspective. The schoolbook algorithm uses rank n*m*p, with:
    # U[i*m+j, i*m+j] = 1, V[j*p+k, i*m+j] = 1, W[k*n+i, i*m+j] = 1
    # where the column index c = i*m*p + j*p + k or similar.
    
    # From the seed code:
    # for i in range(n):
    #   for j in range(m):
    #     for k in range(p):
    #       U[i*m+j][column] = 1
    #       V[j*p+k][column] = 1
    #       W[k*n+i][column] = 1
    #       column += 1
    
    # So for each (i,j,k), there's a column where U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    # The decomposition gives: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # For the schoolbook, this sum equals 1 for each (i,j,k) because exactly one column
    # has all three factors equal to 1.
    
    # So T[i,j,k] = 1 for all i,j,k. The tensor IS all ones!
    
    # But an all-ones tensor of shape (n*m, m*p, p*n) ... wait, no. The tensor T is 
    # implicitly defined by the decomposition constraint. T is a 3-tensor with indices
    # (i*m+j, j*p+k, k*n+i) but that's not a standard 3-tensor indexing.
    
    # Actually, T is a 3-tensor with dimensions (n*m) x (m*p) x (p*n), but not all entries
    # are defined independently. The entry T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # is 1 if and only if the indices are consistent (i.e., the j in a matches the j in b,
    # and the k in b matches the k in c, and the i in c matches the i in a).
    
    # So T is NOT all ones. It's a sparse tensor where T[i*m+j, j*p+k, k*n+i] = 1 for 
    # valid (i,j,k), and 0 otherwise.
    
    # This makes much more sense! The tensor has a specific structure based on the 
    # index consistency constraints.
    
    # Let me construct this tensor properly.
    
    # Create the full tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Now we want to find a rank-r decomposition:
    # T[a,b,c] = sum_{r=0}^{rank-1} U[a,r] * V[b,r] * W[c,r]
    
    # We'll use HOOI to find low-rank approximations, then check for exactness.
    
    best_result = None
    best_rank = n * m * p  # Schoolbook rank
    
    # Try various ranks
    # Known bounds: For 4x4*4x5, the rank is at most 4*4*5=80 (schoolbook)
    # Lower bounds: max(n*m, m*p, p*n) = max(16, 20, 20) = 20
    # So rank is at least 20.
    
    # Let's try to find the minimum rank using HOOI
    # Start from lower bound and go up
    
    lower_bound = max(n * m, m * p, p * n)
    upper_bound = n * m * p
    
    # Try ranks from lower_bound to some reasonable upper limit
    # We'll use HOOI with multiple random initializations
    
    for target_rank in range(lower_bound, min(upper_bound + 1, lower_bound + 40)):
        if time.time() > deadline:
            break
        
        found_exact = False
        # Multiple random restarts
        n_restarts = 10 if target_rank < 30 else 5
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            np.random.seed(42 + restart * 1000 + target_rank)
            U_init = np.random.randn(n * m, target_rank)
            V_init = np.random.randn(m * p, target_rank)
            W_init = np.random.randn(p * n, target_rank)
            
            # HOOI iterations
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()
            
            max_iter = 100
            tol = 1e-12
            prev_error = float('inf')
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Compute residuals
                # T_approx[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                
                # Update U: U_new[a,r] = sum_{b,c} T[a,b,c] * V[b,r] * W[c,r]
                # This is equivalent to: U_new = (V^T * T * W) reshaped
                # More efficiently: for each r, U[:,r] = sum_b V[b,r] * (T * W[:,r])
                
                # Let's compute this efficiently
                # T is sparse, so we can use the explicit structure
                
                # Actually, let's just compute it directly since the tensor is small
                # T has shape (16, 20, 20) = 6400 entries, most zeros
                
                # Compute U update
                U_new = np.zeros_like(U)
                for r in range(target_rank):
                    # U[:,r] = sum_{b,c} T[:,b,c] * V[b,r] * W[c,r]
                    # = sum_b V[b,r] * sum_c T[:,b,c] * W[c,r]
                    # = sum_b V[b,r] * (T[:,b,:] @ W[:,r])
                    for b in range(m * p):
                        if np.any(T[:, b, :]):
                            U_new[:, r] += V[b, r] * (T[:, b, :] @ W[:, r])
                
                # Update V
                V_new = np.zeros_like(V)
                for r in range(target_rank):
                    for a in range(n * m):
                        if np.any(T[a, :, :]):
                            V_new[:, r] += U[a, r] * (T[a, :, :] @ W[:, r])
                
                # Update W
                W_new = np.zeros_like(W)
                for r in range(target_rank):
                    for a in range(n * m):
                        if np.any(T[a, :, :]):
                            W_new[:, r] += U[a, r] * (T[a, :, :] @ V[:, r])
                
                # Normalize columns to prevent drift
                for r in range(target_rank):
                    norm_u = np.linalg.norm(U_new[:, r])
                    norm_v = np.linalg.norm(V_new[:, r])
                    norm_w = np.linalg.norm(W_new[:, r])
                    if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                        U_new[:, r] /= norm_u
                        V_new[:, r] /= norm_v
                        W_new[:, r] /= norm_w
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                if time.time() > deadline:
                    break
            
            # Check if we have an exact decomposition
            # Compute T_approx and compare to T
            T_approx = np.zeros_like(T)
            for r in range(target_rank):
                T_approx += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            error = np.linalg.norm(T_approx - T)
            
            if error < 1e-10:
                # Found exact decomposition!
                # Round to nearest integers or half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify after rounding
                T_approx_rounded = np.zeros_like(T)
                for r in range(target_rank):
                    T_approx_rounded += np.einsum('a,b,c->abc', U_rounded[:, r], V_rounded[:, r], W_rounded[:, r])
                
                error_rounded = np.linalg.norm(T_approx_rounded - T)
                
                if error_rounded < 1e-10:
                    best_rank = target_rank
                    best_result = {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    found_exact = True
                    break
                else:
                    # Try integer rounding
                    U_rounded = np.round(U)
                    V_rounded = np.round(V)
                    W_rounded = np.round(W)
                    
                    T_approx_rounded = np.zeros_like(T)
                    for r in range(target_rank):
                        T_approx_rounded += np.einsum('a,b,c->abc', U_rounded[:, r], V_rounded[:, r], W_rounded[:, r])
                    
                    error_rounded = np.linalg.norm(T_approx_rounded - T)
                    
                    if error_rounded < 1e-10:
                        best_rank = target_rank
                        best_result = {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                        found_exact = True
                        break
            
            if found_exact:
                break
        
        if found_exact:
            break
    
    # If no exact decomposition found, fall back to schoolbook
    if best_result is None:
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
    
    return best_result
# EVOLVE-BLOCK-END
