"""Improved program for matrix multiplication 2x5x6 tensor decomposition."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Start with schoolbook decomposition (rank 60)
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def to_list(mat):
        return mat.tolist()
    
    def verify(U, V, W):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For matrix multiplication, T[i,j,k] = 1 if all indices valid (which they always are)
        # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
        # Let's verify: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1 for all i,j,k
        
        # Convert to numpy for speed
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def is_halfinteger(val):
        """Check if a value is an integer or half-integer."""
        v = float(val)
        return abs(v * 2 - round(v * 2)) < 1e-9
    
    def check_halfinteger_matrix(mat):
        for row in mat:
            for val in row:
                if not is_halfinteger(val):
                    return False
        return True
    
    def to_halfint_list(mat):
        """Convert numpy matrix to list of lists of half-integers."""
        result = []
        for row in mat:
            r = []
            for val in row:
                v = float(val)
                # Round to nearest half-integer
                half = round(v * 2) / 2.0
                r.append(half)
            result.append(r)
        return result
    
    # Try to find a lower-rank decomposition using structured search
    # The key insight: for small matrices, we can try rank-deficient decompositions
    # by looking for linear dependencies among the rank-1 tensors.
    
    # Strategy: Try ranks from lower bound up to schoolbook rank
    # Lower bound for matrix multiplication rank: max(n*m, m*p, p*n) but actually
    # the true lower bound is more complex. For 2x5x6, let's try systematically.
    
    # Known results: rank of 2x2x2 is 7, rank of 3x3x3 is at most 23
    # For 2x5x6, let's try to find if rank < 60 is possible.
    
    # Approach: Use the fact that we can try to express the tensor as a sum of
    # rank-1 tensors with structured coefficients.
    
    # A practical approach: try random structured decompositions with small rank
    # and check if they work.
    
    best_rank = schoolbook_rank
    best_decomp = (U_sb, V_sb, W_sb)
    
    # Try a few structured approaches:
    
    # Approach 1: Try rank = n*m + m*p + p*n - something (savings from shared terms)
    # Actually, let's try a more systematic search.
    
    # For small problems, we can try to find decompositions where the factor
    # matrices have sparse structure.
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Theoretical lower bound: the rank must be at least max(n*m, m*p, p*n) / something
    # Actually, a simple lower bound is that rank >= max(n, m, p) but that's weak.
    
    # Let's try: for each candidate rank r from some lower bound to schoolbook_rank,
    # try to find U, V, W of that rank that work.
    
    # Since exhaustive search is infeasible, use structured random search.
    
    def try_random_decomp(rank, num_trials=1000):
        """Try random decompositions of given rank."""
        n, m, p = DIMENSIONS
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                return None
            
            # Generate random sparse-ish matrices
            density = 0.3
            U = np.random.choice([-1, 0, 1], size=(n*m, rank))
            V = np.random.choice([-1, 0, 1], size=(m*p, rank))
            W = np.random.choice([-1, 0, 1], size=(p*n, rank))
            
            # Zero out some entries for sparsity
            U[np.random.rand(*U.shape) < (1-density)] = 0
            V[np.random.rand(*V.shape) < (1-density)] = 0
            W[np.random.rand(*W.shape) < (1-density)] = 0
            
            if verify(U, V, W):
                return (U, V, W)
        return None
    
    # Try structured decompositions based on known patterns
    # Pattern: For 2xn x nxm, there are known optimal decompositions
    
    # Let's try a systematic approach: build up from known good decompositions
    # For 2x2x2, optimal rank is 7 (Strassen).
    # We can try to combine Strassen-like patterns.
    
    # Actually, let's try a more targeted search:
    # The tensor for 2x5x6 can potentially be decomposed using the structure
    # of the 2x2x2 Strassen decomposition extended.
    
    # Let's try specific structured decompositions:
    
    # Try rank values that are "nice" numbers
    candidate_ranks = list(range(30, schoolbook_rank + 1))
    
    # Sort by likelihood of success (lower ranks first for the search, but we
    # should try to find the minimum)
    
    # For efficiency, let's try a few key ranks with more trials
    # and use the best we find.
    
    # First, let's try if rank 60 can be improved with a structured approach.
    
    # Key idea: The schoolbook decomposition has rank n*m*p = 60.
    # We can potentially save rank by sharing computations.
    
    # For 2x5x6: 
    # - 2 rows in first matrix, 5 columns
    # - 5 rows in second matrix, 6 columns
    # - Output is 2x6
    
    # Each output element C[i,k] = sum_j A[i,j]*B[j,k]
    # There are 2*6 = 12 output elements, each requiring 5 multiplications.
    
    # A decomposition of rank r means we can compute all 12 outputs using r
    # "intermediate" products, where each intermediate is a product of a linear
    # combination of A entries and a linear combination of B entries.
    
    # Let's try to find such a decomposition by solving a system of equations.
    
    # For a given rank r, we need:
    # U: 10 x r, V: 30 x r, W: 12 x r (since p*n = 6*2 = 12)
    # such that for all i in [0,2), j in [0,5), k in [0,6):
    #   sum_c U[i*5+j, c] * V[j*6+k, c] * W[k*2+i, c] = 1
    
    # This is a system of 2*5*6 = 60 equations.
    
    # Let's try a linear algebra approach: fix U and V, solve for W, then check.
    
    def solve_for_w(U, V, n, m, p, rank):
        """Given U and V, solve for W such that the decomposition holds."""
        # For each (i, k), the equation is:
        # sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * V[j*p+k, c]) = 1
        # Wait, that's not quite right. Let me re-derive.
        
        # The tensor equation is:
        # For all (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # For fixed (i, k): sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * V[j*p+k, c]) ... 
        # No, that's not a simple linear equation in W because U and V depend on j.
        
        # Actually, for fixed (i, k), the constraint involves all j:
        # For each j: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # This is not linear in W alone because U and V are also variables.
        
        # Let me think differently. Fix U and V. Then for each (i, k), we have
        # m equations (one for each j):
        # sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1
        
        # This is a system of m equations in r unknowns (the W[k*n+i, :] vector).
        # We can solve this if the system is consistent.
        
        W = np.zeros((p * n, rank))
        
        for i in range(n):
            for k in range(p):
                # Build the system: for each j, A_j @ W[k*n+i, :] = 1
                # where A_j[c] = U[i*m+j, c] * V[j*p+k, c]
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                
                # Solve A @ w = 1 (vector of ones)
                # This is an underdetermined system (m equations, r unknowns) if r > m
                # Use least squares
                try:
                    w, residuals, rank_sys, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    # Check if the solution is exact
                    if np.allclose(A @ w, np.ones(m), atol=1e-9):
                        W[k*n+i, :] = w
                    else:
                        return None
                except:
                    return None
        
        return W
    
    # Try structured U and V matrices
    # Key: use the structure of matrix multiplication
    
    # Approach: Try U and V with specific patterns
    # For example, U[i*m+j, c] could be 1 for certain (i,j,c) combinations
    
    # Let's try a more systematic approach:
    # For each candidate rank, try to find a decomposition using iterative methods.
    
    # Actually, let's try a simpler approach: use the known fact that for
    # small matrix multiplication, the rank can often be reduced below n*m*p.
    
    # For 2x5x6, let's try ranks from 35 to 60 and see if we can find a valid one.
    
    # Use a time-limited search
    search_start = time.time()
    search_time_limit = 130  # seconds
    
    # Try a few structured approaches first
    # Approach: Use block structure
    
    # For 2x5x6, we can think of it as two 1x5x6 multiplications.
    # Each 1x5x6 has rank 5*6 = 30 in schoolbook.
    # But we can share some computations between the two rows.
    
    # Let's try: rank = 2*5*6 - savings
    # The savings come from sharing intermediate products.
    
    # A simple structured decomposition:
    # For each j (column of A / row of B), we compute A[0,j]*B[j,:] and A[1,j]*B[j,:]
    # These are 2*6 = 12 products per j, times 5 j's = 60 products total (schoolbook).
    
    # Can we do better? If we compute linear combinations:
    # (A[0,j] + A[1,j]) * (B[j,0] + B[j,1]) etc.
    
    # This is essentially Strassen's idea. For 2x2x2, it saves 1 product.
    # For 2x5x6, the savings depend on the structure.
    
    # Let's try to find a decomposition with rank < 60 using random search
    # with structured initialization.
    
    # Initialize with schoolbook and try to "compress" it.
    
    # For now, let's just try random searches for a few ranks.
    
    for target_rank in [55, 50, 45, 40, 35]:
        if time.time() - search_start > search_time_limit:
            break
        
        # Try random decompositions
        result = try_random_decomp(target_rank, num_trials=200)
        if result is not None:
            best_rank = target_rank
            best_decomp = result
            break
    
    # If we found a better decomposition, use it
    if best_rank < schoolbook_rank:
        U, V, W = best_decomp
        return {"U": to_halfint_list(U), "V": to_halfint_list(V), "W": to_halfint_list(W)}
    
    # Otherwise, return schoolbook
    return {"U": to_halfint_list(U_sb), "V": to_halfint_list(V_sb), "W": to_halfint_list(W_sb)}
