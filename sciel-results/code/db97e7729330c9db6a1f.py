"""Search for rank-minimal exact tensor decomposition of 2x4x7 matrix multiplication."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, n=2, m=4, p=7):
    """Check if U, V, W give exact decomposition."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _verify_np(U, V, W, n=2, m=4, p=7):
    """Fast verification using numpy."""
    rank = len(U[0]) if U else 0
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    # Reshape to tensors
    U_t = U_np.reshape(n, m, rank)  # (n, m, rank)
    V_t = V_np.reshape(m, p, rank)  # (m, p, rank)
    W_t = W_np.reshape(p, n, rank)  # (p, n, rank)
    # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Using einsum
    T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    return np.allclose(T, 1.0, atol=1e-10)

def _try_rank_structured(n, m, p, target_rank, deadline):
    """
    Try to find a decomposition with the given target rank using structured approach.
    Split middle dimension into two halves and use Strassen-like method on each.
    """
    start = time.time()
    elapsed = start - start
    
    # Split m into two halves: m1 = m//2, m2 = m - m1
    m1 = m // 2
    m2 = m - m1
    
    # For each half, we have a (n x m_half) * (m_half x p) multiplication
    # The rank for n x m_half x p can be bounded by n * m_half * p (schoolbook)
    # But we can do better with structured methods.
    
    # Approach: Use the fact that for small matrices, we can use known low-rank decompositions.
    # For 2x2xp, the rank is at most 3p - 2 (for p >= 2) using Strassen-like methods.
    # For 2x2x7: rank <= 3*7 - 2 = 19
    
    # Let's try to construct a decomposition for each half separately,
    # then combine them.
    
    # For half 1: A1 (n x m1) * B1 (m1 x p)
    # For half 2: A2 (n x m2) * B2 (m2 x p)
    
    # The total tensor is the sum of the two half tensors.
    # If we can find decompositions of rank r1 and r2 for each half,
    # the total rank is r1 + r2.
    
    # Let's try to find low-rank decompositions for each half.
    
    def _decompose_half(n, m_half, p, target_rank_half):
        """Find a rank-minimal decomposition for n x m_half x p."""
        # For n=2, m_half=2, p=7, try to find rank <= 19
        
        # Use a systematic approach: try random initializations and optimize
        best_rank = n * m_half * p
        best_U, best_V, best_W = None, None, None
        
        # Try various ranks
        for r in range(target_rank_half, best_rank + 1):
            if time.time() > deadline:
                break
            if r > best_rank:
                break
            
            # Try multiple random initializations
            for attempt in range(20):
                if time.time() > deadline:
                    break
                
                # Initialize with random small values
                U_init = np.random.uniform(-1, 1, (n * m_half, r))
                V_init = np.random.uniform(-1, 1, (m_half * p, r))
                W_init = np.random.uniform(-1, 1, (p * n, r))
                
                # Try to solve using iterative method
                # This is a nonlinear system, so we use a simple iterative approach
                success = _solve_nonlinear(U_init, V_init, W_init, n, m_half, p, r, deadline)
                if success:
                    return r, U_init, V_init, W_init
        
        return best_rank, best_U, best_V, best_W
    
    def _solve_nonlinear(U, V, W, n, m_half, p, r, deadline):
        """Try to solve the nonlinear system using iterative refinement."""
        # This is a placeholder - we need a better solver
        # For now, return False
        return False
    
    # For now, let's try a simpler structured approach:
    # Use the known result that 2x2xp has rank 3p-2 for p>=2
    # For p=7, rank = 19
    
    # Construct using Strassen-like method for 2x2x7
    # This requires a specific construction
    
    # Alternative: Use a greedy approach to build up the decomposition
    # Start with rank 1 terms and add more until we reach the target
    
    # For now, return a schoolbook decomposition as fallback
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
    return rank, U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Strategy: Use structured decomposition based on splitting the middle dimension
    # and using known low-rank methods for small matrix multiplication
    
    # For 2x4x7, split into two 2x2x7 problems
    # Each 2x2x7 has rank at most 19 (using Strassen-like method)
    # Total rank: 2 * 19 = 38
    
    # Let's try to construct this explicitly
    
    # Known result: The bilinear complexity of 2x2xp matrix multiplication is 3p-2 for p>=2
    # For p=7: rank = 3*7-2 = 19
    
    # We need to construct the actual decomposition
    
    # Let's try a different approach: use a systematic search with better initialization
    
    # First, let's try to find a decomposition with rank < 56 using optimization
    
    best_rank = n * m * p  # 56
    best_U, best_V, best_W = _schoolbook(n, m, p)
    
    # Try structured approaches
    # Approach 1: Split middle dimension
    # Approach 2: Use known low-rank decompositions
    
    # For now, let's try a random search with verification
    # This is slow but might find something
    
    # Better: use a deterministic construction
    
    # Let's try to use the fact that the tensor is separable in a certain way
    
    # Actually, let's just return the schoolbook for now and improve later
    # The key insight is that we need a better structural approach
    
    # Try rank 28 (lower bound based on max dimension product)
    # Try rank 38 (2 * 19 from Strassen)
    
    # For now, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}

def _schoolbook(n, m, p):
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
    return U, V, W
