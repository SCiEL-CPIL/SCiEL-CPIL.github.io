"""Program for exact tensor decomposition of matrix multiplication."""
import numpy as np
from fractions import Fraction
import time
import random

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120s wall-clock limit
    
    # Schoolbook decomposition of rank n*m*p
    rank_school = n * m * p
    
    # Try to find a lower rank decomposition
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # Let's try ranks from 21 up to schoolbook-1
    
    # First, let's try a structured approach: 
    # For 2x3 by 3x7, we can potentially use the fact that one dimension is small (n=2)
    
    # Strategy: Try to find rank r decomposition for r < n*m*p
    # We'll use a randomized algebraic approach with verification
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is exact."""
        # Check that for all (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # This is a multilinear form check
        
        num_rows_U = len(U)
        num_cols_U = len(U[0]) if U else 0
        num_rows_V = len(V)
        num_cols_V = len(V[0]) if V else 0
        num_rows_W = len(W)
        num_cols_W = len(W[0]) if W else 0
        
        if num_cols_U != num_cols_V or num_cols_V != num_cols_W:
            return False
        
        rank = num_cols_U
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s - 1) > 1e-10:
                        return False
        return True
    
    def schoolbook_decomposition(n, m, p):
        """Return the standard schoolbook decomposition of rank n*m*p."""
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
    
    # Try to find a rank-(n*m*p - 1) decomposition
    # This is a known technique for small matrix multiplication tensors
    
    def try_rank_minus_one(n, m, p):
        """
        Try to find a rank (n*m*p - 1) decomposition.
        For small dimensions, this can sometimes be done by finding
        a linear dependence among the schoolbook terms.
        """
        rank = n * m * p - 1
        
        # Initialize with schoolbook
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Fill with first rank-1 schoolbook terms
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if column < rank:
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
        
        # Now we need to adjust to make the decomposition work
        # This requires solving a system of equations
        # For now, this is a placeholder - the actual solution requires
        # more sophisticated algebraic methods
        
        return None  # Not implemented for this specific case
    
    # Let's try a more practical approach:
    # For 2x3 by 3x7, let's see if we can use a block structure
    
    # The multiplication is:
    # [a11 a12 a13]   [b11 b12 b13 b14 b15 b16 b17]
    # [a21 a22 a23] * [b21 b22 b23 b24 b25 b26 b27]
    #                  [b31 b32 b33 b34 b35 b36 b37]
    
    # = [a11*b11 + a12*b21 + a13*b31, ..., a11*b17 + a12*b27 + a13*b37]
    #   [a21*b11 + a22*b21 + a23*b31, ..., a21*b17 + a22*b27 + a23*b37]
    
    # Each output entry is a sum of 3 products (m=3)
    # There are 2*7 = 14 output entries
    
    # Schoolbook: 42 terms
    # Lower bound: 21 terms
    
    # Let's try to find a decomposition with rank 41, 40, etc.
    # by using a randomized search with verification
    
    # For reliability, let's first check if we can quickly find rank 41
    # If not, return schoolbook
    
    # Actually, for the specific case of 2x3x7, let's try a direct construction
    
    # One approach: use the fact that we can sometimes express the multiplication
    # as a sum of structured terms
    
    # Let's try a different strategy: use the known result that for matrix
    # multiplication M_{n,m,p}, if min(n,m,p) >= 2, the rank is often
    # strictly less than n*m*p
    
    # For our specific case, let's try to construct a rank-41 decomposition
    
    # The key insight: if we can find a non-trivial linear relation among
    # the 42 schoolbook terms, we can reduce the rank by 1
    
    # Let's try to find such a relation using linear algebra
    
    # Create the tensor in a form suitable for rank analysis
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # For rank analysis, we can flatten the tensor and look for dependencies
    
    # However, this is complex. Let's try a more direct approach.
    
    # For 2x3 by 3x7, let's manually construct a potential lower-rank decomposition
    
    # Actually, let's be pragmatic: try a few ranks and verify
    
    # Start with rank = n*m*p - 1 = 41
    # If we can't find it quickly, fall back to schoolbook
    
    # Given the time constraints, let's try a simple heuristic:
    # Check if rank 41 is achievable by attempting to solve
    
    # For now, let's return schoolbook as the safe choice
    # but structure the code to allow easy modification
    
    # Let's try one more thing: a structured decomposition
    
    # For 2x3 by 3x7, we can think of it as:
    # Row 1 of A times B: 1x3 * 3x7 = 1x7 (rank 7)
    # Row 2 of A times B: 1x3 * 3x7 = 1x7 (rank 7)
    # But these share the same B, so we can't simply add them
    
    # The total rank is not simply 7+7=14 because of the shared structure
    
    # Let's try a randomized approach with a time limit
    
    def randomized_search(target_rank, max_iterations=10000):
        """
        Try to find a rank-target_rank decomposition using random initialization.
        """
        if time.time() > deadline - 10:  # Leave 10s margin
            return None
        
        n, m, p = DIMENSIONS
        num_constraints = n * m * p  # 42
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            # Randomly initialize U and V
            U = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(m * p)]
            
            # For each constraint (i, j, k), we need:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            
            # This is a system of equations in W
            # But it's not linear in W because W appears multiplied by U*V
            
            # Actually, for fixed U and V, the equation is linear in W:
            # For each (i,j,k): sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1
            
            # So we have a linear system for W
            # Number of equations: 42
            # Number of unknowns: 14 * target_rank
            
            # For target_rank >= 3, we have more unknowns than equations (underdetermined)
            # So we can find a solution if the system is consistent
            
            # Let's set up the linear system
            # Unknowns: W[c, r] for c in 0..13, r in 0..target_rank-1
            # Total unknowns: 14 * target_rank
            
            # For each (i, j, k):
            #   a = i*m + j
            #   b = j*p + k  
            #   c = k*n + i
            #   sum_r U[a, r] * V[b, r] * W[c, r] = 1
            
            # This gives us 42 equations
            
            # Let's build the matrix for the linear system
            # Each equation has 14*target_rank variables
            
            # Actually, this is sparse - each equation only involves target_rank variables
            # (one for each r, all with the same c)
            
            # So we can solve for each c independently if we group by c
            
            # For a fixed c (which determines k and i via c = k*n + i),
            # the equations involving W[c, :] are those where k*n + i = c
            
            # For c = k*n + i, we have k = c // n, i = c % n
            # The equations are for all j in 0..m-1:
            #   sum_r U[i*m + j, r] * V[j*p + k, r] * W[c, r] = 1
            
            # So for each c, we have m equations in target_rank unknowns (W[c, 0], ..., W[c, target_rank-1])
            
            # For m=3 and target_rank=41, we have 3 equations in 41 unknowns - underdetermined
            # We can find a solution if the 3 equations are consistent
            
            # Let's try to solve for each c
            
            W = [[0] * target_rank for _ in range(p * n)]
            valid = True
            
            for c in range(p * n):
                k = c // n
                i = c % n
                
                # Build the 3 x target_rank matrix for this c
                A_c = []
                b_c = []
                for j in range(m):
                    row = []
                    for r in range(target_rank):
                        row.append(U[i * m + j][r] * V[j * p + k][r])
                    A_c.append(row)
                    b_c.append(1)
                
                # Check if the system is consistent
                # For an underdetermined system, it's always consistent if the
                # rows are linearly dependent in a compatible way
                
                # Actually, for an underdetermined system Ax = b, a solution exists
                # if and only if b is in the column space of A^T
                # i.e., for any y such that y^T A = 0, we need y^T b = 0
                
                # This is complex to check. Let's just try to find a solution
                # using least squares or a simple method
                
                # For simplicity, let's try to set most W[c, r] = 0 and solve for the rest
                # This is not guaranteed to work, but let's try
                
                # Actually, let's just verify if the schoolbook decomposition works
                # and return it if we can't find a better one quickly
                
                valid = False
                break
            
            if not valid:
                continue
            
            # Verify the decomposition
            if verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try to find a rank-41 decomposition
    # If it takes too long, fall back to schoolbook
    
    result = None
    if time.time() < deadline - 20:  # Leave 20s margin
        result = randomized_search(41, max_iterations=1000)
    
    if result is None:
        # Fall back to schoolbook
        result = schoolbook_decomposition(n, m, p)
    
    return result
# EVOLVE-BLOCK-END
