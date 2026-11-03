"""Matrix multiplication 4,4,9 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
from itertools import product
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

def _verify(U, V, W, n=4, m=4, p=9):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    n, m, p = 4, 4, 9
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    rank = U.shape[1]
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For each (i,j,k), the sum should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if s != 1:
                    return False
    return True

def _rank_of_matrix(M):
    """Compute rank of a matrix with rational/integer entries using sympy-like approach."""
    # Use numpy with float for initial check, then verify
    Mf = np.array(M, dtype=float)
    r = np.linalg.matrix_rank(Mf, tol=1e-10)
    return int(r)

def construct():
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s limit
    
    n, m, p = 4, 4, 9
    schoolbook_rank = n * m * p  # 144
    
    # Start with schoolbook as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Strategy: Use a structured decomposition approach.
    # For matmul (n,m,p), we can try rank r decompositions where
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with entries in {0, 1, -1} or small integers.
    
    # Key insight: The tensor for matmul has a special structure.
    # T[i,j,k] = delta(i,j,k) where the "bilinear" form is A[i,j]*B[j,k].
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all valid (i,j,k)
    
    # Approach: Try to find low-rank decompositions using the fact that
    # the matmul tensor can be decomposed via various algebraic techniques.
    
    # For 4x4 x 4x9, let's try a systematic approach:
    # 1. First try rank values from a lower bound upward
    # 2. Use the structure: fix U and V patterns, solve for W
    
    # Lower bound: The rank must be at least max(n*m, m*p, p*n) / something
    # Actually, a better lower bound: rank >= n*m*p / (something)
    # For matmul, known lower bounds exist but let's try practical values.
    
    # Let's try a different approach: use the "Winograd" style for small matrices
    # or try to find decompositions with entries in {-1, 0, 1}
    
    # Given the time constraint, let's try a focused search:
    # Try ranks from a reasonable lower bound up
    
    # For 4x4x4 matmul, the best known is rank 21 (Strassen-like)
    # For 4x4x9, it's less standard. Let's try to find good decompositions.
    
    # Approach: Use a greedy/iterative method
    # 1. Start with a set of candidate "patterns" for U columns
    # 2. For each pattern, determine what V and W should be
    
    # Actually, let me try a more direct approach:
    # The matmul tensor T has T[i,j,k] = 1 if the indices form a valid chain
    # We want to write T as sum of rank-1 tensors.
    
    # A rank-1 tensor is: u_i * v_j * w_k where u depends on (i,j), v on (j,k), w on (k,i)
    # More precisely: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's try to use the fact that for fixed j, the sub-tensor T[i,j,k] for all i,k
    # is an n x p matrix (all ones in this case since it's standard matmul).
    # Wait no - T[i,j,k] = 1 for ALL valid (i,j,k) in standard matmul? No!
    # T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k].
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for ALL i,j,k
    
    # This is a system of n*m*p = 144 equations.
    
    # Let's try a structured approach:
    # Assume U, V, W have entries in {-1, 0, 1} (or small integers)
    # and search for valid decompositions at various ranks.
    
    # Given time constraints, let's try a randomized search with verification.
    
    import random
    random.seed(42)
    
    # Try ranks from a lower estimate
    # For 4x4x4, rank 21 is best known. For 4x4x9, let's estimate.
    # The "size" is 144. A rough lower bound might be around 50-70.
    # Let's try from 60 upward, but that's too many to search blindly.
    
    # Better approach: Use the algebraic structure.
    # For matmul, we can use block decompositions.
    
    # Let me try a specific structured decomposition:
    # Split the 4x4 into 2x2 blocks and use Strassen-like ideas,
    # but adapted for the 4x9 output.
    
    # Actually, let's try a more systematic approach:
    # Use the fact that we can compose smaller decompositions.
    
    # For now, let's try a randomized search for small-entry decompositions
    # at various rank levels, checking validity.
    
    # Given the 130s budget, let's be strategic.
    
    # Try a few specific rank values with structured search
    candidate_ranks = [144, 140, 135, 130, 125, 120, 115, 110, 105, 100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50]
    
    for rank in candidate_ranks:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Try to find a decomposition of this rank
        # Use a simple approach: try random assignments and verify
        
        # Actually, for a more effective search, let's use a structured approach:
        # Fix the form of U, V, W and solve.
        
        # For a rank-r decomposition, we have:
        # n*m*r + m*p*r + p*n*r variables
        # n*m*p = 144 constraints (each must equal 1)
        
        # This is underdetermined for small r, so solutions likely exist
        # but finding them with small integer entries is the challenge.
        
        # Let's try a different tactic: use known good decompositions
        # for smaller cases and extend.
        
        # For 4x4x4, rank 21 exists. Can we use that as a building block?
        # 4x4 x 4x9 = (4x4 x 4x4) * (4x4) + (4x4 x 4x4) * (4x4) + (4x4 x 4x1)
        # This doesn't directly help.
        
        # Let me try a focused random search for small ranks
        if rank < 100:
            # Try random search with small entries
            max_iter = 2000
            found = False
            for it in range(max_iter):
                if time.time() - start_time > DEADLINE - 20:
                    break
                # Generate random U, V with small entries
                U = np.random.randint(-2, 3, size=(n*m, rank))
                V = np.random.randint(-2, 3, size=(m*p, rank))
                # For each (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                # This is a linear system in W for fixed U, V
                # For each (k,i), the equation involves W[k*n+i, c] for all c
                # But W[k*n+i, c] appears in multiple equations (for different j)
                
                # Let's set up the linear system for W
                # Unknowns: W[k*n+i, c] for k in range(p), i in range(n), c in range(rank)
                # Total unknowns: p*n*rank
                # Equations: for each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # Each equation involves only W[k*n+i, :] (the row of W for fixed k,i)
                # So the system decouples by (k,i)!
                
                # For each (k,i), we have m equations (one for each j):
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1, for j=0..m-1
                # This is a system of m equations in rank unknowns W[k*n+i, 0..rank-1]
                
                # If m <= rank, this may have solutions.
                # Let's check if the system is consistent.
                
                valid = True
                W = np.zeros((p*n, rank))
                for k in range(p):
                    for i in range(n):
                        # Build the m x rank matrix A where A[j,c] = U[i*m+j,c] * V[j*p+k,c]
                        A = np.zeros((m, rank))
                        for j in range(m):
                            for c in range(rank):
                                A[j,c] = U[i*m+j,c] * V[j*p+k,c]
                        # Solve A * W[k*n+i,:] = 1
                        # Use least squares or check if 1 is in the column space
                        try:
                            # Check if system is consistent
                            # Use numpy lstsq
                            sol, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=1e-10)
                            # Check if the solution is exact
                            if np.allclose(A @ sol, np.ones(m), atol=1e-8):
                                W[k*n+i,:] = sol
                            else:
                                valid = False
                                break
                        except:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Check that entries are "nice" (integers or half-integers)
                    # For now, accept any real solution and round if close
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    
                    # Verify exactly
                    if _verify(U_list, V_list, W_list):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        found = True
                        break
                    else:
                        # Try rounding
                        U_r = np.round(U).astype(int)
                        V_r = np.round(V).astype(int)
                        W_r = np.round(W).astype(int)
                        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                            best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                            best_rank = rank
                            found = True
                            break
            
            if found:
                break
        else:
            # For larger ranks, the schoolbook or near-schoolbook should work
            # Let's just use schoolbook if we haven't found better
            pass
    
    # Check time and return best result
    if time.time() - start_time > DEADLINE - 5:
        return best_result
    
    return best_result
