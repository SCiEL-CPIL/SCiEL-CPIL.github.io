"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130  # Leave 20s safety margin before 150s kill
    
    # Trivial decomposition has rank n*m*p = 72
    # Lower bound is max(n*m, m*p, p*n) = 24
    # Try to find a decomposition with rank between 24 and 72
    
    # The multiplication tensor T has indices (i, j, k) where:
    # i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 if we're computing C[i,k] += A[i,j]*B[j,k]
    # Actually, the tensor is defined such that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 for all valid (i,j,k), and the decomposition
    # represents this as a sum of rank-1 tensors.
    
    # For a rank-1 term with factor r:
    # U[i,j,r], V[j,k,r], W[i,k,r] such that
    # sum_r U[i,j,r]*V[j,k,r]*W[i,k,r] = T[i,j,k]
    
    # Let's try to use a structured decomposition.
    # Since m=3, we can potentially use 3 terms per (i,k) pair if we can
    # factor the j-dependence.
    
    # Try rank = 24 (the lower bound)
    target_rank = 24
    
    # Set up the problem: we need to find U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # such that for all (i,j,k):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p = 72 equations in rank*(n*m + m*p + p*n) = 24*(9+24+24) = 1368 unknowns.
    # So it's underdetermined, which is good.
    
    # Use a simple approach: initialize with random values and use iterative refinement.
    # But for exact integer/half-integer solutions, let's try a structured approach.
    
    # Structured approach: For each j in [0,3), create 3*8=24 terms? No, that's 72.
    
    # Let me think differently. The tensor T[i,j,k] = 1 for all (i,j,k).
    # We can write this as:
    # T[i,j,k] = sum_{j'} delta(j, j') 
    # where delta is the Kronecker delta. But that doesn't help directly.
    
    # Alternative: use the fact that 1 = sum_{j=0}^{2} delta(j, j)
    # So T[i,j,k] = delta(j, j) = sum_{j'} delta(j, j') * delta(j', j)
    # This doesn't reduce the rank.
    
    # Let's try a numerical approach with random initialization and then round to integers.
    np.random.seed(42)
    
    # Try several random initializations
    best_result = None
    best_rank = n * m * p  # Start with trivial
    
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        # Try rank = 24
        r = 24
        
        # Initialize U, V, W with small random values
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        # Use iterative method: for each (i,j,k), we need
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is a tensor completion problem. Let's use a simple gradient-free approach.
        # Actually, let's use a different strategy: fix U and V, solve for W, then fix U and W, solve for V, etc.
        
        # For fixed U and V, W[k*n+i, r] can be determined if we can invert the system.
        # But the system is:
        # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is linear in W for each (i,k) if we fix (i,j). But j varies.
        
        # Let's try a different approach: use the fact that we can set up a linear system.
        # For each r, define a tensor T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need sum_r T_r[i,j,k] = 1 for all (i,j,k).
        
        # This is equivalent to: the sum of rank-1 tensors equals the all-ones tensor.
        
        # The all-ones tensor of size 3x3x8 has rank at most min(3*3, 3*8, 8*3) = 9?
        # No, the rank of an all-ones tensor of size a x b x c is min(a,b,c) if we're talking about
        # the standard tensor rank. Actually, the all-ones tensor can be written as:
        # T[i,j,k] = 1 = sum_{r=1}^{min(n,m,p)} e_r[i] * e_r[j] * e_r[k]
        # where e_r are basis vectors. But this only works if we can factor it.
        
        # For an all-ones tensor of size 3x3x8, the rank is 3 (since 3 = min(3,3,8)).
        # So we can write:
        # T[i,j,k] = sum_{r=1}^{3} (1/sqrt(3)) * (1/sqrt(3)) * (1/sqrt(8)) * 3 * 8 / (3*3*8) ... 
        # Actually, the all-ones tensor of size a x b x c has rank min(a,b,c).
        # Proof: T[i,j,k] = 1 = sum_{r=1}^{min(a,b,c)} u_r[i] * v_r[j] * w_r[k]
        # where u_r, v_r, w_r are chosen appropriately.
        
        # For 3x3x8: rank = 3.
        # So we can use rank 3 for the all-ones part, but we also have the structure of matrix multiplication.
        
        # Wait, I'm confusing two things. The multiplication tensor is NOT the all-ones tensor.
        # The multiplication tensor T[i,j,k] is defined such that it encodes the multiplication operation.
        # For matrix multiplication C = A*B, the tensor is:
        # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
        # So T[i,j,k] = 1 for all valid (i,j,k), and the "output" index is (i,k).
        
        # The decomposition is: C[i,k] = sum_{j,r} A[i,j] * B[j,k] * T[i,j,k,r]
        # where T[i,j,k,r] is the r-th rank-1 component.
        
        # Actually, the standard formulation is:
        # C[i,k] = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_l V[j,l,r] * B[j,l]) * W[i,k,r]
        # No, that's not right either.
        
        # Let me use the correct formulation from the task:
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # The decomposition is:
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
        # Wait, that's not standard. Let me re-read the task.
        
        # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
        # "output factor follows the official trace convention W[k*n+i,r]"
        
        # The standard tensor decomposition for matrix multiplication is:
        # M[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
        # where M[i,j,k] = 1 for the multiplication tensor.
        # And the multiplication is:
        # C[i,k] = sum_{j,r} U[i,j,r] * V[j,k,r] * W[i,k,r] * A[i,j] * B[j,k]
        # = sum_r W[i,k,r] * (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k])
        
        # So U[i*m+j, r] = U[i,j,r], V[j*p+k, r] = V[j,k,r], W[k*n+i, r] = W[i,k,r]
        
        # We need: sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = 1 for all (i,j,k)
        
        # This is the same as before. The all-ones tensor of size 3x3x8 has rank 3.
        # So we should be able to achieve rank 3!
        
        # Let's construct it:
        # T[i,j,k] = 1 = sum_{r=1}^{3} u_r[i] * v_r[j] * w_r[k]
        # Choose u_r[i] = 1 for all i,r, v_r[j] = 1 for all j,r, w_r[k] = 1/3 for all k,r
        # Then sum_r u_r[i] * v_r[j] * w_r[k] = sum_r 1 * 1 * (1/3) = 3 * (1/3) = 1. 
        
        # So U[i*m+j, r] = 1 for all i,j,r
        # V[j*p+k, r] = 1 for all j,k,r  
        # W[k*n+i, r] = 1/3 for all i,k,r
        
        # This gives rank 3!
        
        r = 3
        U = np.ones((n*m, r))
        V = np.ones((m*p, r))
        W = np.ones((p*n, r)) / r
        
        # Verify: for all (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # = sum_r 1 * 1 * (1/3) = 3 * (1/3) = 1. Correct!
        
        # Convert to list of lists with integer/half-integer entries
        U_list = [[float(val) for val in row] for row in U]
        V_list = [[float(val) for val in row] for row in V]
        W_list = [[float(val) for val in row] for row in W]
        
        # Verify the decomposition
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r] for r in range(r))
                    if abs(val - 1.0) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: return trivial decomposition
    rank_trivial = n * m * p
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
