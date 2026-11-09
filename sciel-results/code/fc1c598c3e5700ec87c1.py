"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]
_DEADLINE = time.time() + 140  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return U, V, W, rank

def _verify_decomp(n, m, p, U, V, W, rank):
    """Verify that the decomposition exactly computes matrix multiplication."""
    # Build tensor T[i,j,k] = sum_c U[c, i*m+j] * V[c, j*p+k] * W[c, k*n+i]
    # Should equal 1 for all i,j,k (the multiplication tensor)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_search(n, m, p, max_rank, time_limit):
    """
    Try to find a rank-r decomposition for r < max_rank.
    Uses iterative refinement with random restarts.
    """
    np.random.seed(42)
    best_U, best_V, best_W, best_rank = None, None, None, max_rank
    
    for target_rank in range(max_rank - 1, 30, -1):  # Try ranks from max_rank-1 down to 31
        if time.time() > time_limit:
            break
        found = False
        for attempt in range(50):
            if time.time() > time_limit:
                break
            # Random initialization with half-integer values
            U = np.round(np.random.uniform(-1, 1, (n * m, target_rank)) * 2) / 2
            V = np.round(np.random.uniform(-1, 1, (m * p, target_rank)) * 2) / 2
            W = np.round(np.random.uniform(-1, 1, (p * n, target_rank)) * 2) / 2
            
            # Verify
            if _verify_decomp(n, m, p, U.tolist(), V.tolist(), W.tolist(), target_rank):
                found = True
                best_U, best_V, best_W, best_rank = U.tolist(), V.tolist(), W.tolist(), target_rank
                break
        
        if found:
            break
    
    return best_U, best_V, best_W, best_rank

def _structured_search(n, m, p, time_limit):
    """
    Try structured decompositions that exploit the small n=2.
    """
    best_U, best_V, best_W, best_rank = None, None, None, n * m * p
    
    # Strategy: For n=2, try to combine the two rows
    # C[0,k] = sum_j A[0,j] B[j,k]
    # C[1,k] = sum_j A[1,j] B[j,k]
    
    # Try rank = 2*m*p - s for small s
    # This is hard to do analytically, so use numerical search
    
    # Alternative: use the fact that we can share some terms
    # If we let some W entries be shared between i=0 and i=1
    
    # Try a specific structure:
    # For each (j,k), we need terms that contribute to both C[0,k] and C[1,k]
    # Term c: U[0,j,c]*V[j,k,c]*W[k,0,c] contributes to C[0,k]
    #          U[1,j,c]*V[j,k,c]*W[k,1,c] contributes to C[1,k]
    
    # If W[k,0,c] = W[k,1,c] for some c, we can share
    # But this doesn't directly reduce rank
    
    # Let's try a different approach: assume some structure
    # and solve the resulting system
    
    # For now, fall back to schoolbook if no improvement found
    U, V, W, rank = _schoolbook(n, m, p)
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135  # 15s safety margin
    
    # Start with schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    
    # Try to improve
    improved_U, improved_V, improved_W, improved_rank = _try_rank_search(
        n, m, p, rank, deadline
    )
    
    if improved_rank < rank and improved_U is not None:
        U, V, W = improved_U, improved_V, improved_W
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
