"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x5 matrix by a 5x5 matrix.
    
    The matrix multiplication tensor T[i,j,k] corresponds to the coefficient
    of A[i,j]*B[j,k] in the product C[i,k].
    
    We need U, V, W such that:
      T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    where n=2, m=5, p=5.
    
    The schoolbook rank is n*m*p = 50. We try to find a lower rank.
    
    For small dimensions, we can use randomized search / rank minimization.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    def build_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = build_schoolbook()
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition using random search
    # The key insight: for matrix multiplication, the minimum rank is known
    # to be at most n*m*p, but can be lower.
    # For 2x5x5, let's try various ranks from a lower bound up.
    
    # Lower bound: max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # But actually the tensor rank lower bound is more complex.
    
    # Let's try a structured approach:
    # The matrix multiplication tensor for (n,m,p) has a natural structure.
    # We can try to exploit the fact that n=2 is small.
    
    # One approach: use the identity that for fixed i, the slice over (j,k)
    # is an m x p matrix, and we need to represent it as a sum of rank-1 terms
    # weighted by W[k*n+i, r].
    
    # Let's try rank r from some lower bound up to 50
    # For each rank, try to solve the system
    
    # Actually, for this small problem, let's try a more direct approach.
    # We know that the bilinear complexity of 2x5x5 matrix multiplication
    # is at most 50 (schoolbook). Let's see if we can do better.
    
    # A known technique: use the fact that we can share some computations.
    # For example, if we group terms, we might reduce the rank.
    
    # Let's try a random search approach with verification.
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        # (The tensor T has 1s at all positions since it's the standard MM tensor)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    # Try to find a lower rank by random search
    # Start from rank 25 (theoretical lower bound) and go up
    for target_rank in range(25, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Try to optimize using a simple gradient-free approach
            # or just check if this random init works (unlikely)
            # Better: use a structured approach
            
            # Actually, let's use a more systematic approach:
            # Fix U and V randomly, then solve for W, or vice versa.
            
            # For each (i,j,k), we need:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is a system of equations in the W variables.
            
            # Let's fix U and V, and solve for W.
            # For each k, i: we have m equations (one for each j)
            # Unknowns: W[k*n+i, r] for r=0..target_rank-1
            
            # Actually, for fixed (k,i), the equations over j are:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
            
            # This is a linear system in W[k*n+i, :]:
            # Let A_ji_r = U[i*m+j, r] * V[j*p+k, r]
            # Then sum_r A_ji_r * W[k*n+i, r] = 1 for each j
            
            # This is m equations in target_rank unknowns.
            # If target_rank >= m, we might be able to solve.
            
            # But we need consistency across all (i,j,k).
            
            # Let's try: fix U, V randomly, then for each (k,i), solve the
            # linear system for W[k*n+i, :].
            
            U_rand = np.random.randn(n * m, target_rank)
            V_rand = np.random.randn(m * p, target_rank)
            
            # For each (k, i), solve for W_row
            W_sol = np.zeros((p * n, target_rank))
            valid = True
            for k in range(p):
                for i in range(n):
                    # Build the m x target_rank system
                    # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    A = np.zeros((m, target_rank))
                    for j in range(m):
                        A[j, :] = U_rand[i * m + j, :] * V_rand[j * p + k, :]
                    # Solve A * w = ones(m)
                    # Use least squares
                    try:
                        w, residuals, rank_, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        # Check if the solution is good
                        if np.max(np.abs(A @ w - np.ones(m))) > 1e-6:
                            valid = False
                            break
                        W_sol[k * n + i, :] = w
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Verify the full decomposition
                if verify_decomposition(U_rand, V_rand, W_sol, target_rank):
                    best_result = {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_sol.tolist()}
                    best_rank = target_rank
                    break
        
        if best_rank == target_rank:
            break
    
    # If we didn't find anything better, return the schoolbook
    return best_result
# EVOLVE-BLOCK-END
