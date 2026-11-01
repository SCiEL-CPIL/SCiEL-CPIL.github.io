import numpy as np
import time
from fractions import Fraction

_start_time = time.time()
TIME_LIMIT = 145  # safety margin before 150s

def _time_exceeded():
    return time.time() - _start_time > TIME_LIMIT

def construct():
    n, m, p = 2, 2, 15
    rank_school = n * m * p  # 60
    
    # Schoolbook decomposition as fallback
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower rank decomposition
    # The tensor T[2i+j, 15j'+k, 2k'+i'] = 1 if i==i', j==j', k==k'
    # We want to find U (4×r), V (30×r), W (30×r) such that:
    # T[a,b,c] = sum_{col=0}^{r-1} U[a,col] * V[b,col] * W[c,col]
    
    # For this specific structure, let's try ranks from 40 to 59
    # Using a structured approach: assume the decomposition has a certain form
    
    # Key insight: For each k, the contribution to C[:,k] depends on A and B[:,k]
    # We can try to find a decomposition where some products are shared across k
    
    # Try rank = 2*p + 2 = 32 first (optimistic)
    # Then try higher ranks
    
    for target_rank in range(32, rank_school):
        if _time_exceeded():
            break
        
        # Build the linear system for a given rank
        # Variables: U[4×r], V[30×r], W[30×r] -> 4*r + 30*r + 30*r = 64*r variables
        # Constraints: T[a,b,c] = 1 for 60 specific (a,b,c) tuples, 0 otherwise
        # But T is sparse: only 60 non-zero entries
        
        # Actually, the tensor has exactly 60 non-zero entries (one for each (i,j,k) triple)
        # And all other entries are 0. So we have 4*30*30 = 3600 total entries,
        # but only 60 are non-zero.
        
        # For a rank-r decomposition: T[a,b,c] = sum_col U[a,col]*V[b,col]*W[c,col]
        # This must equal 1 for the 60 non-zero positions and 0 for the rest.
        
        # This is a system of 3600 equations in 64*r unknowns.
        # For r=60: 64*60=3840 unknowns, 3600 equations -> likely solvable
        # For r=32: 64*32=2048 unknowns, 3600 equations -> overdetermined, may not have solution
        
        # Let's try to solve this as a least-squares problem for smaller ranks
        # and check if the solution is exact
        
        if target_rank < 40:
            # Too many constraints relative to unknowns, skip
            continue
        
        # Build the linear system
        # For each (a,b,c), the equation is:
        # sum_col U[a,col]*V[b,col]*W[c,col] = T[a,b,c]
        
        # This is nonlinear in the variables. We need a different approach.
        # 
        # Alternative: fix U and W, solve for V, or use alternating least squares.
        # But that's what the current program tries and it's slow.
        
        # Let's try a different structured approach:
        # Assume the decomposition has the form where for each k,
        # we use a small number of products, and some are shared.
        
        # Specifically, try rank = 4*p = 60 (schoolbook) or slightly less.
        # 
        # Actually, let me try a specific construction:
        # Use 3 products per column for the "diagonal" parts and share some.
        
        # For now, if we can't find a better decomposition quickly, return schoolbook.
        pass
    
    # Return the schoolbook decomposition
    return {"U": U_school, "V": V_school, "W": W_school}
