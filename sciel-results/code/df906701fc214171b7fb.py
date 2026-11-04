import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin

    # Schoolbook decomposition as fallback
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower rank decomposition using structured search
    # The strategy is to use Kronecker product structure
    
    # For 3x3x7, we can think of this as multiplying 3x3 matrices by 3x7 matrices
    # One approach: use the fact that 3x3 multiplication has rank 21 (Strassen-like)
    # and combine with the 7 dimension
    
    # Let's try a direct algebraic approach
    # We want to find tensors U, V, W of minimal rank r such that:
    # For all i,j,k: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # Start with rank 63 (schoolbook) and try to reduce
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try ranks from lower bounds up to schoolbook
    # Lower bound for 3x3x7 is at least max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But we need to verify if rank 21 is achievable
    
    for target_rank in range(21, best_rank + 1):
        if time.time() > deadline - 20:
            break
            
        # Use structured tensors based on Kronecker product
        # U = U1 ⊗ U2, V = V1 ⊗ V2, W = W1 ⊗ W2
        # where U1 is for 3x3 multiplication and U2 is for the 7 dimension
        
        # For 3x3 scalar multiplication, rank is 9 (schoolbook) or 7 (Strassen)
        # Let's use rank 7 Strassen for 3x3
        
        # Strassen for 3x3: we can use the rank-7 decomposition
        # This requires specific matrices
        
        # Alternative: try to construct using bilinear maps
        # U has shape (9, r), V has shape (21, r), W has shape (21, r)
        
        # For rank 21, we need:
        # 9*21 = 189 constraints from U
        # 21*21 = 441 constraints from V  
        # 21*21 = 441 constraints from W
        # Total: 1071 variables (3*21*17 = no, r*9 + r*21 + r*21 = r*51)
        # For r=21: 21*51 = 1071 variables
        # Constraints: 3*3*7 = 63 tensor equations
        # This is an underdetermined system, so solutions likely exist
        
        # Let's set up the system for rank 21
        if target_rank == 21:
            r = 21
            # Build the constraint matrix
            # For each (i,j,k) in [0,n)×[0,m)×[0,p):
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            
            # This is a system of 63 cubic equations in 1071 variables
            # Let's use a greedy approach: fix W first, then solve for U and V
            
            # Initialize W with identity-like structure
            W = np.zeros((p * n, r))
            for k in range(p):
                for i in range(n):
                    idx = k * n + i
                    W[idx, (i * p + k) % r] = 1.0
            
            # For each r, we have a rank-1 component
            # U[:,r] * V[:,r]^T should contribute to the tensor
            # Let's set U and V to extract the right components
            
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            
            # For each r, define which (i,j) and (j,k) it covers
            for r_idx in range(r):
                # This r_idx corresponds to a specific (i,j,k) combination
                i = r_idx % n
                j = (r_idx // n) % m
                k = (r_idx // (n * m)) % p
                if r_idx // (n * m) < p:
                    U[i * m + j, r_idx] = 1.0
                    V[j * p + k, r_idx] = 1.0
                    W[k * n + i, r_idx] = 1.0
            
            # Verify
            T_reconstructed = np.zeros((n, m, p))
            for r_idx in range(r):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_reconstructed[i, j, k] += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
            
            # Check if this is correct
            if np.allclose(T_reconstructed, 1.0):
                # Convert to list of lists
                result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                return result
        elif target_rank == 27:
            # Try rank 27 with a different structure
            r = 27
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            # Use a more distributed structure
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Assign multiple components
                        for offset in range(0, 3):
                            r_idx = (i * 3 + j) * 3 + k + offset * 7
                            if r_idx < r:
                                U[i * m + j, r_idx] += 1.0 / 3.0
                                V[j * p + k, r_idx] += 1.0 / 3.0
                                W[k * n + i, r_idx] += 1.0 / 3.0
            
            # Verify
            T_reconstructed = np.zeros((n, m, p))
            for r_idx in range(r):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_reconstructed[i, j, k] += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
            
            if np.allclose(T_reconstructed, 1.0, atol=1e-10):
                result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                return result
    
    return best_result
# EVOLVE-BLOCK-END
