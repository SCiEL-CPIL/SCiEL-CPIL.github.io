<<<<<<< SEARCH
"""
Optimal rank decomposition for 2x3 by 3x10 matrix multiplication.
The minimum rank for this tensor is 60 (schoolbook).
"""

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m) x rank
    # V: (m*p) x rank
    # W: (p*n) x rank
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (Trace convention: W[k*n+i, r])
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
"""
Optimal rank decomposition for 2x3 by 3x10 matrix multiplication.
Attempts to find rank below schoolbook 60 using random linear combinations.
"""

DIMENSIONS = [2, 3, 10]
import numpy as np
from scipy.optimize import nnls

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    # Try to find a decomposition with rank less than schoolbook
    # The tensor equation for each (i, j, k) is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # All other entries should be 0
    
    # We'll try ranks from schoolbook_rank-1 down to some lower bound
    # For each rank, we try to solve the nonlinear system
    
    # Lower bound on rank: max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    # But realistically, for 2x3x10, known lower bounds suggest rank is close to 60
    
    # Let's try a few ranks below 60 with random initialization
    for target_rank in range(schoolbook_rank - 1, 29, -1):
        if target_rank < 30:  # Can't go below max dimension product
            break
        
        success = False
        for attempt in range(100):
            np.random.seed(attempt * 1000 + target_rank)
            
            # Random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # We need to solve: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # This is a system of 60 equations with 3*target_rank variables
            
            # Use iterative refinement: fix two matrices, solve for the third
            # This is a linear system when two are fixed
            
            for iteration in range(100):
                # Fix U and V, solve for W
                # For each (i,j,k): W[k*n+i, :] = sum_r U[i*m+j,r]*V[j*p+k,r] * W[k*n+i,r]
                # Actually this isn't linear in W. Let me reconsider.
                
                # The equation is trilinear. Let's use a different approach.
                # For fixed U and V, the equation becomes:
                # sum_r (U[i*m+j,r]*V[j*p+k,r]) * W[k*n+i,r] = 1
                # This is linear in W!
                
                # Build the linear system for W
                # Each equation (i,j,k) gives: sum_r A[i,j,k,r] * W[k*n+i,r] = 1
                # where A[i,j,k,r] = U[i*m+j,r]*V[j*p+k,r]
                
                # But W has entries W[k*n+i, r] for r in range(target_rank)
                # So for each (i,j,k), we have one equation involving target_rank variables
                
                # Actually, let's set up the system properly.
                # Variables: W_flat of size (p*n*target_rank)
                # For each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W_flat[(k*n+i)*target_rank+r] = 1
                
                # This is a linear system: M * W_flat = b
                # where M has 60 rows and p*n*target_rank columns
                
                M_W = np.zeros((n*m*p, p*n*target_rank))
                b_W = np.ones(n*m*p)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx = i * m * p + j * p + k
                            for r in range(target_rank):
                                M_W[idx, (k*n+i)*target_rank + r] = U[i*m+j, r] * V[j*p+k, r]
                
                # Solve least squares
                W_flat, _ = np.linalg.lstsq(M_W, b_W, rcond=None)
                W = W_flat.reshape(p*n, target_rank)
                
                # Check if solution is valid
                error = np.max(np.abs(M_W @ W_flat - b_W))
                if error < 1e-6:
                    success = True
                    break
                
                # Similarly, we could alternate fixing different pairs, but for now
                # let's just check if this worked
                
                if success:
                    break
            
            if success:
                # Verify the full decomposition
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Verify
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(target_rank))
                            if abs(val - 1.0) > 1e-6:
                                success = False
                                break
                        if not success:
                            break
                    if not success:
                        break
                if not success:
                    continue
                
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # If no success at this rank, try lower
    
    # Fall back to schoolbook
    rank = schoolbook_rank
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
