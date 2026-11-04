"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the trivial rank-125 schoolbook decomposition."""
    n, m, p = DIMENSIONS
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


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == delta_{i,k} for all i,j,k.
    Returns True if exact (within tol)."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Build tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # We can do this via einsum: T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # This is tricky with einsum because indices are coupled.
    # Let's just loop over i,j,k for 5x5x5=125 entries.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ua[i * m + j, r] * Va[j * p + k, r] * Wa[k * n + i, r]
                expected = 1.0 if i == k else 0.0
                if abs(s - expected) > tol:
                    return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions using random search
    # The matrix multiplication tensor for 5x5x5 has known lower bounds.
    # The Strassen algorithm gives rank 49 for 4x4x4. For 5x5x5, 
    # we can potentially use known constructions.
    
    # One approach: use the fact that 5x5x5 can be embedded in larger structures
    # or use random tensor decomposition.
    
    # Let's try a simple random search for lower rank.
    # For each candidate rank r, try to find U, V, W of that rank.
    
    # Known: the minimum rank for 5x5x5 matrix multiplication is unknown exactly,
    # but is at most 125 (schoolbook) and at least some lower bound.
    # Let's try ranks from 125 down to see if we can find anything better.
    
    # Actually, let's try a different approach: use the Strassen-like construction
    # or known low-rank decompositions.
    
    # For now, let's just return the schoolbook decomposition.
    # A proper search would require more sophisticated methods.
    
    # Try a few random initializations for lower ranks
    random.seed(42)
    np.random.seed(42)
    
    # Try rank 120, 115, 110, etc.
    for target_rank in range(best_rank - 1, max(best_rank - 20, 50), -1):
        if time.time() > deadline:
            break
        
        # Random initialization
        U_rand = np.random.randn(n * m, target_rank)
        V_rand = np.random.randn(m * p, target_rank)
        W_rand = np.random.randn(p * n, target_rank)
        
        # Convert to lists for verification
        U_list = U_rand.tolist()
        V_list = V_rand.tolist()
        W_list = W_rand.tolist()
        
        if _verify(U_list, V_list, W_list, n, m, p):
            best = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = target_rank
            break
    
    return best
# EVOLVE-BLOCK-END
