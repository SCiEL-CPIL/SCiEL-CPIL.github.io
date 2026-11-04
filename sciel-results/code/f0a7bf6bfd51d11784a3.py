"""Find rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]
DEADLINE = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 2x3 * 3x4 matmul."""
    t_start = time.time()
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # W index: W[k*n + i, r] (trace convention)
    
    # The multiplication tensor T[i,j,k] = 1 for all i,j,k
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # (using the convention where W[k*n+i] corresponds to output index)
    
    # Known lower bound: max(n*m, m*p, p*n) = max(6,12,8) = 12
    # Known upper bound (schoolbook): n*m*p = 24
    # For (2,3,4), the rank is known to be 18 (from algebraic complexity results)
    
    # Let's try to find a decomposition by searching for integer/half-integer solutions
    # Start from rank 12 (lower bound) and go up
    
    def verify(U, V, W, rank):
        """Verify that U, V, W form a valid decomposition."""
        # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def verify_with_arrays(U_arr, V_arr, W_arr, rank):
        """Verify using numpy arrays for speed."""
        # U_arr: (n*m, rank), V_arr: (m*p, rank), W_arr: (p*n, rank)
        # Check: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_arr[i*m+j]
                    v_row = V_arr[j*p+k]
                    w_row = W_arr[k*n+i]
                    s = np.dot(u_row, v_row * w_row)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # The schoolbook decomposition always works as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Try to find lower rank decompositions
    # Strategy: use the known structure. For matrix multiplication, 
    # we can try various algebraic constructions.
    
    # One approach: try rank values from 12 upward and attempt to find solutions
    # using integer programming or structured search
    
    # For (2,3,4), let's try a direct algebraic construction.
    # The key insight: we can try to find a rank-r decomposition where
    # r < 24 by exploiting the small dimensions.
    
    # Let's use a systematic search with numpy for speed
    # We'll try to find solutions with entries in {-1, 0, 1, 0.5, -0.5}
    
    # Actually, let's try a more principled approach:
    # Use the fact that for 2x3*3x4, we can potentially achieve rank 18
    # by a Strassen-like construction.
    
    # Let me try to construct a rank-18 decomposition explicitly.
    # For 2x3 * 3x4:
    # Output is 2x4, so we have 8 output entries.
    # Each output C[i,k] = sum_j A[i,j]*B[j,k]
    
    # A systematic approach: try all possible rank values and use 
    # a constraint solver or structured search.
    
    # Given time constraints, let's try a randomized/structured search
    # for low-rank decompositions with small integer/half-integer entries.
    
    def try_find_decomposition(rank, max_time=5.0):
        """Try to find a rank-r decomposition using structured search."""
        t0 = time.time()
        
        # For small ranks, we can try to use the structure of the problem.
        # The multiplication tensor for 2x3*3x4 has a specific structure.
        
        # Let's try a different approach: use the known algebraic construction
        # For 2x3 * 3x4, one can achieve rank 18 using the following idea:
        # Split the 3x4 matrix into 2x2 blocks and use Strassen-like tricks.
        
        # Actually, let me try a direct construction for rank 18.
        # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0]
        # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1]
        # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
        # C[0,3] = A[0,0]B[0,3] + A[0,1]B[1,3] + A[0,2]B[2,3]
        # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0] + A[1,2]B[2,0]
        # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1] + A[1,2]B[2,1]
        # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]
        # C[1,3] = A[1,0]B[0,3] + A[1,1]B[1,3] + A[1,2]B[2,3]
        
        # Let's try to use a rank-18 construction based on the following:
        # Group the 3 middle indices into pairs and use Strassen-style products.
        
        # For now, let's try a search-based approach for small ranks.
        # We'll use a greedy/iterative approach.
        
        # Actually, the most reliable approach given time constraints:
        # 1. Try known constructions for specific ranks
        # 2. Fall back to schoolbook if nothing found
        
        # Let me try rank 18 with a specific construction.
        # The idea: use 18 rank-1 terms.
        
        # For 2x3 * 3x4, a known construction achieves rank 18.
        # Let me try to construct it explicitly.
        
        # Approach: Think of it as 2 independent 1x3 * 3x4 multiplications,
        # but share some computations.
        
        # 1x3 * 3x4 requires rank 4 (schoolbook: 3*4=12, but actually rank 4
        # since it's just a vector-matrix product: C[k] = sum_j v[j]*B[j,k])
        # Wait, 1x3 * 3x4: C is 1x4, C[k] = sum_{j=0}^{2} v[j]*B[j,k]
        # This can be done in rank 4: for each k, C[k] = v[0]*B[0,k] + v[1]*B[1,k] + v[2]*B[2,k]
        # But we can do better: rank 3 is the lower bound (since we need at least 3 
        # to represent the 3 terms), and rank 3 works:
        # C[k] = sum_j v[j]*B[j,k], use 3 rank-1 terms: for each j, v[j]*B[j,:]
        # So 1x3*3x4 has rank 3.
        
        # For 2x3*3x4, if independent, rank would be 2*3=6? No, that's not right.
        # The tensor rank of 2x3*3x4 is not simply 2*rank(1x3*3x4).
        
        # Let me just try to find a valid decomposition by search for ranks 12-24.
        
        # Given the complexity, let me use a more practical approach:
        # Try to verify if a rank-18 decomposition exists by construction.
        
        # Known result: rank(2,3,4) = 18
        # Let me try to construct it.
        
        # Construction idea:
        # Let A be 2x3, B be 3x4.
        # Write B = [B1 | B2] where B1 is 3x2, B2 is 3x2.
        # C = A*B = [A*B1 | A*B2]
        # A*B1 is 2x2, A*B2 is 2x2.
        # 2x2*2x2 has rank 7 (Strassen).
        # So total rank would be 7+7=14? But we need to be careful about sharing.
        
        # Actually, let me try a different split.
        # B is 3x4. Split into 3x2 + 3x2.
        # A*B1: 2x3 * 3x2 -> 2x2, rank at most 2*3*2=12, but actually rank 6 (schoolbook for 2x3*3x2)
        # Hmm, this is getting complicated.
        
        # Let me just do a brute-force search for small candidate ranks.
        # For rank r, we need to find U (6xr), V (12xr), W (8xr) with entries in 
        # {-1, -0.5, 0, 0.5, 1} such that the tensor equation holds.
        
        # This is a huge search space, so let me use a smarter approach.
        
        # Use the fact that we can fix some structure and solve linear equations.
        
        # For a given rank r, the condition is:
        # For each (i,j,k): sum_{s=1}^{r} U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] = 1
        
        # This is a system of n*m*p = 24 equations in r*(n*m + m*p + p*n) = r*26 unknowns.
        # For r=12: 12*26=312 unknowns, 24 equations. Underdetermined, but nonlinear.
        
        # Let me try a practical approach: use known constructions or fall back.
        
        # Given time constraints, let me try to construct a rank-18 solution.
        
        # Here's a construction for 2x3 * 3x4 with rank 18:
        # 
        # We can use the following idea:
        # Let the 18 rank-1 terms be indexed by (a, b, c) where a in {0,1,2}, b in {0,1,2}, c in {0,1,2}
        # That gives 9 terms, not 18. Let me think differently.
        
        # Actually, let me just return the schoolbook decomposition for now,
        # but try a few specific constructions first.
        
        # Try rank 18 with a specific pattern:
        # Use 18 = 3*6 terms, where for each of the 3 middle indices j,
        # we use 6 terms for the 2*4=8 outputs... that doesn't quite work.
        
        # Let me try a different approach: use numpy to solve for a decomposition
        # by fixing some entries and solving linear systems.
        
        # For rank r, fix U and W, then V is determined by linear equations.
        # Or fix U and V, then W is determined.
        
        # Let's try: fix U to be a random (or structured) 6xr matrix,
        # fix W to be a random 8xr matrix, then solve for V.
        # The equations are: for each (i,j,k), sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] = 1
        # For fixed U and W, this is linear in V!
        # Let's write it as: for each (i,j,k), sum_s (U[i*m+j,s]*W[k*n+i,s]) * V[j*p+k,s] = 1
        # So for each (j,k), we have a linear equation in V[j*p+k, :].
        # But each V[j*p+k,s] appears in multiple equations (one for each i).
        
        # Specifically, V[j*p+k, s] appears in equations for all i (i=0,1).
        # So we have 2*4=8 equations for V[*, *] involving a given (j,k) pair?
        # No: for fixed (j,k), the variable is V[j*p+k, :], which is a vector of length r.
        # It appears in equations for i=0 and i=1.
        # Equation for (i,j,k): sum_s U[i*m+j,s]*W[k*n+i,s]*V[j*p+k,s] = 1
        # So for fixed (j,k), we have 2 equations (for i=0,1) in r unknowns V[j*p+k,0..r-1].
        # That's underdetermined for r>2.
        
        # But we also have constraints across different (j,k) pairs through U and W.
        
        # Let me try a specific construction:
        # Set U[i*m+j, s] and W[k*n+i, s] to have a specific structure.
        
        # Actually, let me just try to find ANY valid decomposition for ranks 12-24
        # using a randomized approach with linear solves.
        
        import random
        random.seed(42)
        
        candidate_ranks = [18, 17, 16, 15, 14, 13, 12]
        
        for rank in candidate_ranks:
            if time.time() - t_start > max_time:
                break
            
            # Try multiple random initializations
            for attempt in range(50):
                if time.time() - t_start > max_time:
                    break
                
                # Initialize U and W with small integer/half-integer values
                U = np.zeros((n*m, rank))
                W = np.zeros((p*n, rank))
                
                # Try a structured initialization
                # For each rank index s, assign values to U and W
                for s in range(rank):
                    # Randomly assign non-zero entries
                    for idx in range(n*m):
                        if random.random() < 0.5:
                            U[idx, s] = random.choice([-1, -0.5, 0.5, 1])
                    for idx in range(p*n):
                        if random.random() < 0.5:
                            W[idx, s] = random.choice([-1, -0.5, 0.5, 1])
                
                # Now solve for V
                # For each (j,k) pair, we have a system:
                # For i in range(n): sum_s U[i*m+j,s]*W[k*n+i,s]*V[j*p+k,s] = 1
                # This is: M[j,k] @ V[j*p+k, :] = 1
                # where M[j,k] is an n x rank matrix with M[j,k][i,s] = U[i*m+j,s]*W[k*n+i,s]
                
                V = np.zeros((m*p, rank))
                valid = True
                
                for j in range(m):
                    for k in range(p):
                        # Build the n x rank matrix
                        M = np.zeros((n, rank))
                        for i in range(n):
                            for s in range(rank):
                                M[i, s] = U[i*m+j, s] * W[k*n+i, s]
                        
                        # Solve M @ V[j*p+k, :] = ones(n)
                        # This is underdetermined if rank > n
                        # Use least squares
                        try:
                            sol, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                            # Check if it's an exact solution
                            if np.allclose(M @ sol, np.ones(n), atol=1e-10):
                                V[j*p+k, :] = sol
                            else:
                                valid = False
                                break
                        except:
                            valid = False
                            break
                    
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Snap to half-integers
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                
                # Verify
                if verify_with_arrays(U, V, W, rank):
                    return U, V, W, rank
        
        return None
    
    # Try to find a low-rank decomposition
    result = try_find_decomposition(24)
    
    if result is not None:
        U, V, W, rank = result
    else:
        # Fall back to schoolbook
        U, V, W, rank = schoolbook()
    
    # Convert to lists for the required interface
    U_list = [[U[i, j] for j in range(rank)] for i in range(n*m)]
    V_list = [[V[i, j] for j in range(rank)] for i in range(m*p)]
    W_list = [[W[i, j] for j in range(rank)] for i in range(p*n)]
    
    # Ensure entries are exactly integer or half-integer
    for mat in [U_list, V_list, W_list]:
        for row in mat:
            for idx in range(len(row)):
                val = row[idx]
                # Snap to nearest half-integer
                row[idx] = round(val * 2) / 2
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
