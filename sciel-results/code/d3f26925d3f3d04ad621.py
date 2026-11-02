"""
Matrix multiplication 2x3x7 rank-minimal exact tensor decomposition.

Strategy:
1. Start with the schoolbook decomposition of rank 42.
2. Use a structured local search (simulated annealing-like) to reduce rank.
   - Represent the decomposition as a set of "terms" (triples of vectors).
   - Operations: remove a term and re-approximate, swap terms, merge/split.
   - Use a smooth surrogate: minimize the Frobenius norm of the residual tensor.
   - Periodically round to nearest half-integers and check exactness.

Given the 120s limit, we use a time-bounded search with periodic exact checks.
"""

import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin

    n, m, p = DIMENSIONS
    # The multiplication tensor T[i,j,k] = 1 if output index (k*n+i) depends on (i*m+j) and (j*p+k)
    # Actually, the tensor for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p
    # T_{i,j,k} = 1 if C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The tensor is T in R^{(n*p) x n x m x p} ... wait, let me re-read.
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # So W has shape (p*n, rank) with W[k*n + i, r]
    # U has shape (n*m, rank) with U[i*m + j, r]
    # V has shape (m*p, rank) with V[j*p + k, r]
    # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that's not right either. Let me think again.
    # Standard tensor decomposition for matrix multiplication:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This is a bilinear form. The tensor T has indices (i,j,k) and T[i,j,k] = 1 if C[i,k] depends on A[i,j] and B[j,k].
    # Actually T[i,j,k] = delta_{i,i} delta_{k,k} ... no.
    # T[i,j,k] = 1 for all valid i,j,k because C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor is T in R^{n x m x p} with T[i,j,k] = 1.
    # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Wait, that doesn't look like a standard rank decomposition. Let me re-read the interface.
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # So U is indexed by (i*m+j), V by (j*p+k), W by (k*n+i).
    # The decomposition should satisfy: for all i,j,k: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But T[i,j,k] = 1 for all i,j,k in the matrix multiplication tensor.
    # Wait, that's not right. The matrix multiplication tensor is not all ones.
    # Let me reconsider. The tensor for matrix multiplication C = AB is:
    # T_{i,j,k} = 1 if the (i,k)-entry of C depends on A_{i,j} and B_{j,k}.
    # Since C_{i,k} = sum_j A_{i,j} B_{j,k}, T_{i,j,k} = 1 for all i,j,k.
    # So the tensor is indeed the all-ones tensor in R^{n x m x p}.
    # But that can't be right because the all-ones tensor has rank 1.
    # I think I'm misunderstanding the tensor structure.
    # Let me re-read: "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 7 matrix."
    # The multiplication tensor is T in R^{(n*p) x n x m x p}? No, that doesn't make sense.
    # Actually, the standard formulation is: the matrix multiplication tensor is a 3-tensor T in R^{n x m x p}
    # where T[i,j,k] = 1 if C[i,k] = sum_j A[i,j] B[j,k]. So T[i,j,k] = 1 for all i,j,k.
    # But then the rank is 1, which is trivially wrong.
    # I think the correct interpretation is: the tensor is T in R^{(n*m) x (m*p) x (p*n)} or similar.
    # Let me look at the schoolbook decomposition in the seed program:
    # U[i*m+j, column] = 1, V[j*p+k, column] = 1, W[k*n+i, column] = 1
    # So for each (i,j,k), there's a term that contributes 1 to U[i*m+j], V[j*p+k], W[k*n+i].
    # The decomposition is: sum_r U[a,r] V[b,r] W[c,r] where a=i*m+j, b=j*p+k, c=k*n+i.
    # For the matrix multiplication, we need: for all i,j,k, the coefficient of A[i,j] B[j,k] in C[i,k] is 1.
    # So the condition is: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k.
    # Wait, that's still the all-ones tensor condition.
    # I think the issue is that the tensor is not all ones. Let me think differently.
    # The matrix multiplication can be written as: C[i,k] = sum_j A[i,j] B[j,k].
    # This is a bilinear map. The tensor T is in R^{(n*p) x n x m x p}? No.
    # Actually, I think the correct tensor is: T[i,j,k] = 1 if the output index (k*n+i) receives input from (i*m+j) and (j*p+k).
    # So T is a 3-tensor with indices (a,b,c) where a in [0,n*m), b in [0,m*p), c in [0,p*n).
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k.
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise.
    # This makes more sense! The tensor is sparse, with exactly n*m*p non-zero entries.
    # The rank of this tensor is what we want to minimize.
    # The schoolbook decomposition has rank n*m*p = 42.
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Build the target tensor T as a sparse structure
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # Otherwise T[a,b,c] = 0
    
    # We want to find U (nm x r), V (mp x r), W (pn x r) such that:
    # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c] for all a,b,c
    
    # Start with schoolbook
    rank = nm * mp * pn  # This is wrong. Schoolbook rank is n*m*p = 42.
    rank = n * m * p  # 42
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    def compute_residual(U, V, W):
        """Compute the residual tensor R[a,b,c] = sum_r U[a,r] V[b,r] W[c,r] - T[a,b,c]"""
        # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
        # Compute sum_r U[a,r] V[b,r] W[c,r] for all a,b,c
        # This is the outer product sum
        # R = np.einsum('ar,br,cr->abc', U, V, W)
        # But that's 6*21*14 = 1764 entries, which is fine.
        R = np.einsum('ar,br,cr->abc', U, V, W)
        # Subtract T
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    R[a, b, c] -= 1
        return R
    
    def frob_norm(R):
        return np.linalg.norm(R)
    
    # Check initial residual
    R0 = compute_residual(U, V, W)
    init_norm = frob_norm(R0)
    
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_norm = init_norm
    best_rank = rank
    
    # Local search: try to reduce rank by removing columns and adjusting
    # Use a simple approach: try to find a lower rank decomposition by:
    # 1. Start with current U, V, W
    # 2. Try to remove one column at a time, adjusting the remaining columns to compensate
    # 3. Use gradient descent or a simple heuristic
    
    # Given time constraints, use a structured approach:
    # Try ranks from 42 down to some lower bound
    # Lower bound: the tensor has n*m*p = 42 non-zero entries, but the rank could be lower.
    # For matrix multiplication, known lower bounds exist.
    # For 2x3x7, let's try to find a good decomposition.
    
    # Use a simple local search: perturb columns and project to reduce residual
    random.seed(42)
    
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = rank
    current_norm = init_norm
    
    step_size = 0.1
    iterations = 0
    
    while time.time() < deadline:
        iterations += 1
        # Try to reduce rank by 1
        if current_rank > 1:
            # Try removing a random column
            col_to_remove = random.randint(0, current_rank - 1)
            
            U_new = np.delete(current_U, col_to_remove, axis=1)
            V_new = np.delete(current_V, col_to_remove, axis=1)
            W_new = np.delete(current_W, col_to_remove, axis=1)
            
            # Adjust the remaining columns to minimize residual
            # Use a few steps of gradient descent
            lr = 0.01
            for _ in range(10):
                R = compute_residual(U_new, V_new, W_new)
                # Gradient with respect to U: grad_U[a,r] = sum_{b,c} R[a,b,c] V[b,r] W[c,r]
                grad_U = np.einsum('abc,br,cr->ar', R, V_new, W_new)
                grad_V = np.einsum('abc,ar,cr->br', R, U_new, W_new)
                grad_W = np.einsum('abc,ar,br->cr', R, U_new, V_new)
                
                U_new -= lr * grad_U
                V_new -= lr * grad_V
                W_new -= lr * grad_W
            
            new_norm = frob_norm(compute_residual(U_new, V_new, W_new))
            
            if new_norm < current_norm * 0.9:
                # Accept
                current_U = U_new
                current_V = V_new
                current_W = W_new
                current_rank -= 1
                current_norm = new_norm
                
                # Check if we can round to half-integers
                U_rounded = np.round(current_U * 2) / 2
                V_rounded = np.round(current_V * 2) / 2
                W_rounded = np.round(current_W * 2) / 2
                R_rounded = compute_residual(U_rounded, V_rounded, W_rounded)
                if frob_norm(R_rounded) < 1e-6:
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    best_rank = current_rank
                    best_norm = 0.0
                    break  # Found exact decomposition
        else:
            break
        
        # Also try small perturbations
        if random.random() < 0.5:
            U_pert = current_U + step_size * np.random.randn(*current_U.shape)
            V_pert = current_V + step_size * np.random.randn(*current_V.shape)
            W_pert = current_W + step_size * np.random.randn(*current_W.shape)
            new_norm = frob_norm(compute_residual(U_pert, V_pert, W_pert))
            if new_norm < current_norm:
                current_U = U_pert
                current_V = V_pert
                current_W = W_pert
                current_norm = new_norm
    
    # Final check: try to round best solution
    U_final = np.round(best_U * 2) / 2
    V_final = np.round(best_V * 2) / 2
    W_final = np.round(best_W * 2) / 2
    R_final = compute_residual(U_final, V_final, W_final)
    
    if frob_norm(R_final) < 1e-6:
        # Convert to lists
        U_list = U_final.tolist()
        V_list = V_final.tolist()
        W_list = W_final.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no exact solution found, return schoolbook
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
