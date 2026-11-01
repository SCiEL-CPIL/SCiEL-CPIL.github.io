"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # Tensor T[i, j, k] = 1 for all i,j,k
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # The seed program gives rank = n*m*p = 2*4*6 = 48
    
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
        return {"U": U, "V": V, "W": W}, rank

    best_result, best_rank = schoolbook()
    
    if time.time() > deadline - 10:
        return best_result

    # New Strategy: Fix U and V randomly with integer entries, solve linear system for W.
    # If the solution W has half-integer entries, we have a valid decomposition.
    
    # The linear system:
    # For each (i, j, k), we have:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # Let w be the vector of all W entries. w has size (p*n*rank).
    # The equation is linear in w.
    # We can build the matrix A and vector b.
    # A has (n*m*p) rows and (p*n*rank) columns.
    # b is a vector of ones.
    
    # To increase the chance of finding a sparse/integer solution, 
    # we can try to make U and V sparse or use specific structures.
    # However, random integers might lead to a unique rational solution.
    # If the solution is rational, we can check if it's a half-integer.
    
    # Note: If we use random integers for U and V, the solution W will be rational.
    # We need W entries to be in {0, 0.5, 1, 1.5, ...} i.e., multiples of 0.5.
    # This is a strong constraint.
    
    # Alternative: Use U and V with entries in {-1, 0, 1}.
    # The determinant of the submatrix might be small, leading to small denominators.
    
    # Let's try ranks from 47 down to 30.
    # For each rank, do a few random restarts.
    
    def try_rank(r, max_restarts=50):
        """
        Tries to find a decomposition of rank r by fixing U, V randomly 
        and solving for W.
        Returns (U, V, W) if found, else None.
        """
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                return None
            
            # Generate random U and V with entries in {-1, 0, 1}
            # We want to avoid singular matrices for the linear system.
            U = np.random.randint(-1, 2, size=(n*m, r))
            V = np.random.randint(-1, 2, size=(m*p, r))
            
            # Build the linear system A * w = b
            # A has shape (n*m*p, p*n*r)
            # Indexing:
            # Row index: idx = (i * m + j) * p + k
            # Col index: w_idx = (k * n + i) * r + rr
            # Coefficient: U[i*m+j, rr] * V[j*p+k, rr]
            
            num_eqs = n * m * p
            num_unk = p * n * r
            
            # Instead of building the full sparse matrix explicitly which might be slow,
            # we can use numpy's linear algebra if the matrix is dense.
            # 24 equations, 24*r unknowns. For r=47, 1128 unknowns.
            # This is an underdetermined system.
            # We want a solution w such that A*w = 1.
            # And w entries should be half-integers.
            
            # Since it's underdetermined, there are infinitely many solutions.
            # We can find the minimum norm solution using pseudo-inverse.
            # Or, we can add a regularization to encourage sparsity/half-integers?
            # No, we just need ANY solution that is half-integer.
            
            # Let's try to solve A * w = b using lstsq.
            # A is (24, 24*r).
            
            A = np.zeros((num_eqs, num_unk))
            b = np.ones(num_eqs)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = (i * m + j) * p + k
                        for rr in range(r):
                            # w index for W[k*n+i, rr]
                            col = (k * n + i) * r + rr
                            A[row, col] = U[i*m+j, rr] * V[j*p+k, rr]
            
            # Solve A * w = b
            # Use least squares
            try:
                w, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            # Check if w is close to half-integers
            # w is a flat array of size p*n*r
            # We need to check if all entries are close to 0.5 * integer
            
            # Convert to list for verification
            W_list = [[0]*r for _ in range(p*n)]
            valid = True
            for k in range(p):
                for i in range(n):
                    for rr in range(r):
                        val = w[(k * n + i) * r + rr]
                        # Check if val is close to a half-integer
                        # round(val * 2) / 2
                        nearest = round(val * 2) / 2.0
                        if abs(val - nearest) > 1e-6:
                            valid = False
                            break
                        W_list[k*n+i][rr] = nearest
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # We found a candidate.
                # We must verify the exact decomposition with the rounded W.
                # Also, we should ensure U and V are integers (they are).
                # And W is half-integers.
                
                # Convert U, V to lists of lists for return format
                U_list = U.tolist()
                V_list = V.tolist()
                
                # Verify
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for rr in range(r):
                                s += U_list[i*m+j][rr] * V_list[j*p+k][rr] * W_list[k*n+i][rr]
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try ranks from 47 down to 30
    # We start from 47 because it's most likely to have a solution if 48 does.
    # Actually, if 48 works, 47 might not.
    # Let's try a few ranks.
    
    # Note: The linear system approach works best when the matrix A has full row rank.
    # With random U, V, this is likely.
    
    # Let's prioritize lower ranks for better score, but start with higher to ensure we find something?
    # No, we want the MINIMAL rank.
    # But finding a low rank solution is harder.
    # Let's try ranks in descending order first to see if we can beat 48 easily.
    # If we find 47, we return 47.
    # If not, we try 46, etc.
    
    # However, if we find 40, that's better than 47.
    # So we should search for the lowest rank possible.
    # But the probability of finding a solution decreases as rank decreases?
    # Actually, for very low ranks, the system might be inconsistent or the solution might not be half-integer.
    
    # Let's try ranks from 30 up to 47? No, that's too many.
    # Let's try a few specific ranks: 47, 46, 45, 40, 36, 30.
    
    target_ranks = [47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30]
    
    for r in target_ranks:
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, max_restarts=20)
        if result is not None:
            best_result = result
            best_rank = r
            # We found a solution with rank r. 
            # Can we go lower? 
            # Given time constraints, we might stop here if r is significantly lower.
            # Or we can continue to search for even lower ranks if time permits.
            # Let's continue to the next lower rank to see if we can improve further.
            # But we need to be careful with time.
            # Let's break if we find a rank < 40, as that's a significant improvement.
            if r < 40:
                break

    return best_result

# EVOLVE-BLOCK-END
