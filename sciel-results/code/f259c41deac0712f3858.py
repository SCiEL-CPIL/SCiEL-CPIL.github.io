"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]
import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Try to find a better decomposition
    # Strategy: search for rank-r decompositions by solving the tensor equation
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # The tensor T has T[i,j,k] = 1 for all i,j,k (it's the multiplication tensor)
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # For 3x3x7, let's try ranks from 40 upward (below 63)
    # Use a structured approach: try to find integer/half-integer solutions
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Precompute the schoolbook as fallback
    def make_schoolbook():
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
    
    # Try to find a decomposition with rank < 63
    # Use a randomized/structured search
    # The tensor equation is: for each (i,j,k), sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Let's try a more systematic approach using the fact that we can use
    # rank-1 tensors and combine them
    
    # For small ranks, try integer entries in {-2,-1,0,1,2}
    # This is a constraint satisfaction problem
    
    # Try rank from 50 down to 40
    for target_rank in range(55, 39, -1):
        if time.time() > deadline - 10:
            break
        
        # Use a greedy/structured approach
        # Initialize with zeros and try to fill in
        # This is NP-hard in general, so use heuristics
        
        # Try a specific structured decomposition
        # For 3x3x7, one approach: use the 3x3x3 part (rank 23) + 3x3x4 part
        
        # 3x3x3 multiplication tensor rank is 23 (Strassen-like)
        # 3x3x4: try to find its rank
        
        # Let's try to construct a decomposition directly
        # Using bilinear forms
        
        # Approach: parameterize U, V, W with small integer entries and solve
        # This is a system of quadratic equations
        
        # Simpler: try known constructions
        # For 3x3x7, the rank is conjectured to be around 47-52
        
        # Let's try a direct search with small integer entries
        # Use backtracking on the tensor equations
        
        if target_rank <= 55:
            # Try to find a solution with entries in {-1, 0, 1}
            # This is very hard, so use a smarter approach
            
            # Use the fact that matrix multiplication tensor has structure
            # T[i,j,k] = delta(i,j,k) in some sense
            
            # Actually T[i,j,k] = 1 for all valid (i,j,k) in the multiplication
            # The multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
            # So the tensor is T[i,j,k] where output index is (i,k) and input indices are (j)
            # Wait, let me re-read the convention.
            
            # W[k*n+i, r] - so W is indexed by (k, i) where k is from B's rows, i is from A's cols
            # U[i*m+j, r] - U indexed by (i, j) where i is A's row, j is A's col/B's row
            # V[j*p+k, r] - V indexed by (j, k) where j is B's row, k is B's col
            
            # The tensor equation: for output (i,k), sum_j A[i,j]*B[j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            # Wait, that's not right. Let me think again.
            
            # Standard tensor decomposition of matrix mult:
            # C = A * B, C[i,k] = sum_j A[i,j] B[j,k]
            # Tensor T has indices (i, j, k) with T[i,j,k] = 1 (it's the identity in the j-sum sense)
            # Actually the multiplication tensor is: T[i,j,k] = 1 for all i,j,k
            # And the decomposition: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[i,k,c]
            # Wait no. Let me be precise.
            
            # The bilinear map: (A,B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
            # This can be written as: C[i,k] = sum_j sum_c (A[i,j] * U_c[i,j]) * (B[j,k] * V_c[j,k]) * W_c[i,k]
            # Hmm, that's not standard.
            
            # Standard: the multiplication tensor T is in U ⊗ V ⊗ W where
            # U has basis e_{i,j} (A's entries), V has basis f_{j,k} (B's entries), W has basis g_{i,k} (C's entries)
            # T = sum_{i,j,k} e_{i,j} ⊗ f_{j,k} ⊗ g_{i,k}
            # Rank-r decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
            # where u_c = sum_{i,j} U_c[i,j] e_{i,j}, etc.
            # Then: T[i,j,k] = sum_c U_c[i,j] V_c[j,k] W_c[i,k] = 1 for all i,j,k
            
            # So the constraint is: for ALL i, j, k:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            
            # This is a system of n*m*p = 63 quadratic equations in 3*rank*? variables
            
            # For rank r, we have 3*r*dim variables (U has n*m*r, V has m*p*r, W has p*n*r)
            # Total: r*(n*m + m*p + p*n) = r*(9+21+21) = 51r variables
            # 63 equations, quadratic
            
            # This is underdetermined for r >= 2 (51*2=102 > 63), so solutions should exist
            # But we need them to be "nice" (integer/half-integer)
            
            # Let's try a specific approach: use a known low-rank decomposition structure
            
            # For 3x3 matrix multiplication, rank 23 is known (Landsberg, Strassen)
            # For 3x3x7, we can try: rank(3x3x3) + rank(3x3x4)
            # But the tensors aren't simply additive in the way I described
            
            # Let me just try to find ANY valid decomposition with rank < 63
            # using a numerical approach and then round
            
            # Use least squares / optimization
            # Minimize sum of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
            
            # This is non-convex, but we can try random restarts
            
            if time.time() > deadline - 5:
                break
            
            # Try a few random initializations
            found = False
            for attempt in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Random initialization with small integers
                np.random.seed(attempt)
                U = np.random.randint(-2, 3, size=(n*m, target_rank)).astype(float)
                V = np.random.randint(-2, 3, size=(m*p, target_rank)).astype(float)
                W = np.random.randint(-2, 3, size=(p*n, target_rank)).astype(float)
                
                # Check if this satisfies the tensor equations
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(target_rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    # Convert to lists
                    U_list = [[U[i, c] for c in range(target_rank)] for i in range(n*m)]
                    V_list = [[V[i, c] for c in range(target_rank)] for i in range(m*p)]
                    W_list = [[W[i, c] for c in range(target_rank)] for i in range(p*n)]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    found = True
                    break
            
            if found:
                break  # Found a better rank, stop searching
    
    if best_result is None:
        best_result = make_schoolbook()
    
    return best_result
# EVOLVE-BLOCK-END
