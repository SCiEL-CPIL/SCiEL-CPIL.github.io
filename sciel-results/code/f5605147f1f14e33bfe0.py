"""Matrix multiplication 3x3 x 3x7 - rank-minimal tensor decomposition."""
DIMENSIONS = [3, 3, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """Construct rank-minimal tensor decomposition for 3x3 * 3x7 multiplication."""
    n, m, p = 3, 3, 7
    
    # The 3x3x7 tensor decomposes as direct sum of three 3x3x3 tensors
    # (for column blocks 0-2, 3-4, 5-6... actually 3x3x3, 3x3x3, 3x3x1)
    # Wait: p=7, so it's 3x3x7. Partition into 3x3x3 + 3x3x3 + 3x3x1? No.
    # The tensor T[i,j,k] for i in 0..2, j in 0..2, k in 0..6.
    # This is just the 3x3 * 3x7 multiplication.
    # We can view it as three independent 3x3 * 3x1 multiplications? No.
    # Actually: C[i,k] = sum_j A[i,j]*B[j,k], for k=0..6.
    # The tensor is T[i,j,k] = delta (1 if the (i,j,k) triple participates).
    # It's the standard matrix multiplication tensor of size 3x3x7.
    # 
    # Key insight: we can partition the 7 columns into blocks and use
    # efficient algorithms. Or we can find a global low-rank decomposition.
    #
    # Approach: Use the fact that M_{3,3,7} can be decomposed using
    # rank-7 Strassen for 2x2 blocks. 
    #
    # Partition 3x3 matrix A and 3x7 matrix B:
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B = [[B11, B12, B13], [B21, B22, B23], [B31, B32, B33]] (each B1j etc is 1x7/2x7/1x7)
    #
    # Actually simpler: just find a good decomposition via optimization.
    #
    # Let me try: rank = 3*3*7 = 63 (schoolbook). Try to beat it.
    #
    # Strategy: Use random initialization + gradient descent to minimize
    # the reconstruction error, then round to integers/half-integers.
    
    deadline = time.time() + 130  # Leave 20s safety margin
    
    # Build the target tensor
    # T[i*m + j, j*p + k, k*n + i] = 1 for the bilinear form
    # Actually the tensor is defined by: T[a, b, c] where
    # a = i*m+j (row of U), b = j*p+k (row of V), c = k*n+i (row of W)
    # T[a, b, c] = 1 if there exists i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
    # and the product A[i,j]*B[j,k] contributes to C[i,k].
    # For each (i,j,k): U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = 1
    
    # So the tensor T is a 3D array of size (n*m) x (m*p) x (p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Try rank minimization via iterative method
    # Start with various rank candidates
    
    best_rank = n * m * p  # schoolbook = 63
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook solution as fallback
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    # Use the approach: for a given rank r, find U,V,W such that
    # sum_r U[:,r] outer V[:,r] outer W[:,r] approximates T
    # Then round to integers/half-integers
    
    np.random.seed(42)
    
    for target_rank in range(20, best_rank + 1):
        if time.time() > deadline - 30:
            break
            
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 20:
                break
                
            # Initialize
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # Gradient descent to minimize ||T - sum_r U_r ⊗ V_r ⊗ W_r||^2
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 15:
                    break
                    
                # Compute reconstruction
                # R = sum_r np.einsum('ar,br,cr->abc', U, V, W)
                # This is expensive, so do it efficiently
                # R[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                
                # Compute gradient w.r.t. U
                # dL/dU[a,r] = -2 * sum_{b,c} T[a,b,c] * V[b,r] * W[c,r]
                #            + 2 * sum_{b,c} R[a,b,c] * V[b,r] * W[c,r]
                
                # Use einsum for efficiency
                # V[b,r] * W[c,r] -> for each r, outer product of V[:,r] and W[:,r]
                # Then contract with T and R over b,c
                
                # grad_U[a,r] = 2 * (R - T) contracted with V[:,r] and W[:,r] over b,c
                # = 2 * sum_{b,c} (R[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
                
                # Efficient computation:
                # For each r: temp_r[b,c] = V[b,r]*W[c,r]
                # grad_U[:,r] = 2 * einsum('abc,br,cr->ar', R-T, V, W)
                
                R = np.zeros((n*m, m*p, p*n))
                for r in range(target_rank):
                    R += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                
                err = R - T
                
                grad_U = 2.0 * np.einsum('abc,br,cr->ar', err, V, W)
                grad_V = 2.0 * np.einsum('abc,ar,cr->br', err, U, W)
                grad_W = 2.0 * np.einsum('abc,ar,br->cr', err, U, V)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Normalize to prevent divergence
                for r in range(target_rank):
                    nu = np.linalg.norm(U[:,r])
                    nv = np.linalg.norm(V[:,r])
                    nw = np.linalg.norm(W[:,r])
                    if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
                        scale = (nu * nv * nw) ** (1.0/3.0)
                        if scale > 1e-10:
                            U[:,r] /= scale
                            V[:,r] *= scale
                            W[:,r] *= 1.0
                        # Actually just normalize each to unit norm
                        U[:,r] /= nu
                        V[:,r] /= nv
                        W[:,r] /= nw
                
                if step % 50 == 0:
                    err_norm = np.linalg.norm(err)
                    if err_norm < 1e-4:
                        break
            
            # Check if we got a good solution
            R = np.zeros((n*m, m*p, p*n))
            for r in range(target_rank):
                R += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            err_norm = np.linalg.norm(R - T)
            
            if err_norm < 1e-3:
                # Try to round to integers/half-integers
                # Scale to make entries close to integers
                # Find a common scale factor
                all_vals = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                all_vals = all_vals[all_vals != 0]
                
                # Try different scales
                best_rounded = None
                for scale in [1, 2, 0.5, 4, 0.25]:
                    U_rounded = np.round(U * scale) / scale
                    V_rounded = np.round(V * scale) / scale
                    W_rounded = np.round(W * scale) / scale
                    
                    # Verify
                    R_check = np.zeros((n*m, m*p, p*n))
                    for r in range(target_rank):
                        R_check += np.einsum('a,b,c->abc', U_rounded[:,r], V_rounded[:,r], W_rounded[:,r])
                    check_err = np.linalg.norm(R_check - T)
                    
                    if check_err < 1e-6:
                        best_rounded = (U_rounded, V_rounded, W_rounded, target_rank)
                        break
                
                if best_rounded:
                    best_rank = target_rank
                    best_U, best_V, best_W = best_rounded[:3]
                    break
        
        if best_rank < best_rank + 10:  # Found a good one
            break
    
    # If no better solution found, use schoolbook
    if best_U is None:
        best_U = U_school
        best_V = V_school
        best_W = W_school
        best_rank = n * m * p
    
    # Convert to lists with integer/half-integer entries
    def to_list(mat):
        return [[float(val) for val in row] for row in mat]
    
    # Ensure entries are multiples of 0.5
    best_U = np.round(best_U * 2) / 2
    best_V = np.round(best_V * 2) / 2
    best_W = np.round(best_W * 2) / 2
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
# EVOLVE-BLOCK-END
