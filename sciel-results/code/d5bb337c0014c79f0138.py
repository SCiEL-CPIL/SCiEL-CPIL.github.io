"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]
import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Leave 5 seconds margin
    
    def verify(U, V, W):
        """Verify the tensor decomposition exactly using integer arithmetic."""
        # Convert to integers (multiply by 2 to handle half-integers)
        U2 = [[int(round(2 * u)) for u in row] for row in U]
        V2 = [[int(round(2 * v)) for v in row] for row in V]
        W2 = [[int(round(2 * w)) for w in row] for row in W]
        
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U2[i*m+j][r] * V2[j*p+k][r] * W2[k*n+i][r]
                    # The product should be 1, so in doubled space it should be 8
                    if s != 8:
                        return False
        return True
    
    def schoolbook():
        """Return the schoolbook decomposition with rank 24."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def try_rank_random(rank, max_trials=5000, seed=42):
        """Try random search for a decomposition of given rank."""
        random.seed(seed)
        
        # Use a small set of candidate values for half-integer entries
        # Focus on values that are likely to work: -2, -1, -0.5, 0, 0.5, 1, 2
        candidates = [-2, -1, -0.5, 0, 0.5, 1, 2]
        
        for trial in range(max_trials):
            if time.time() > deadline:
                return None
            
            # Generate random U, V, W
            U = [[random.choice(candidates) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.choice(candidates) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.choice(candidates) for _ in range(rank)] for _ in range(p*n)]
            
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    def try_rank_structured(rank, max_trials=2000, seed=123):
        """
        Structured search: fix U and V with simple patterns, solve for W.
        This exploits the bilinear structure.
        """
        random.seed(seed)
        
        # Strategy: Try various patterns for U and V, then solve linear system for W
        # For each (i,k) pair, we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
        
        # Let's try a different approach: use Kronecker product structure
        # For 2x3 * 3x4, we can try to exploit the fact that n=2 is small
        
        # Try: U has entries in {-1, 0, 1}, V has entries in {-1, 0, 1}, solve for W
        for trial in range(max_trials):
            if time.time() > deadline:
                return None
            
            # Generate U with entries in {-1, 0, 1}
            U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n*m)]
            # Generate V with entries in {-1, 0, 1}
            V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m*p)]
            
            # Now solve for W
            W = [[0.0] * rank for _ in range(p*n)]
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # For this (i,k), we need: for all j in range(m):
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    # This is a linear system in W[k*n+i, :]
                    
                    # Build the matrix A where A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    A = np.array([[U[i*m+j][r] * V[j*p+k][r] for r in range(rank)] for j in range(m)])
                    b = np.ones(m)
                    
                    # Solve A @ W[k*n+i, :] = b
                    if rank >= m:
                        try:
                            # Use least squares
                            sol, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                            # Check if solution is exact
                            if np.allclose(A @ sol, b, atol=1e-9):
                                W[k*n+i] = sol.tolist()
                            else:
                                valid = False
                                break
                        except np.linalg.LinAlgError:
                            valid = False
                            break
                    else:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Check if W entries are half-integers
            all_half_int = True
            for i in range(p*n):
                for r in range(rank):
                    w_val = W[i][r]
                    if abs(2 * w_val - round(2 * w_val)) > 1e-9:
                        all_half_int = False
                        break
                if not all_half_int:
                    break
            
            if all_half_int:
                # Round to nearest half-integer
                W_rounded = [[round(2 * w) / 2 for w in row] for row in W]
                if verify(U, V, W_rounded):
                    return {"U": U, "V": V, "W": W_rounded}
        
        return None
    
    # Try ranks from lower bound up to 24
    # Lower bound is max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # But we can start from a lower bound based on known results
    
    best_result = schoolbook()
    
    # Try ranks from 12 to 23
    for rank in range(12, 24):
        if time.time() > deadline:
            break
        
        # Try structured approach first (more efficient)
        result = try_rank_structured(rank, max_trials=1000)
        if result is not None:
            return result
        
        # Try random approach
        result = try_rank_random(rank, max_trials=2000)
        if result is not None:
            return result
    
    # If nothing better found, return schoolbook
    return best_result

# EVOLVE-BLOCK-END
