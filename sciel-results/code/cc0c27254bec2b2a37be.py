"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _schoolbook(n, m, p):
    """Return schoolbook rank n*m*p decomposition for n×m × m×p."""
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
    """Verify that the decomposition is correct."""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += Ua[i * m + j, r] * Va[j * p + k, r] * Wa[k * n + i, r]
                expected = 1.0
                if abs(val - expected) > tol:
                    return False
    # Check that off-diagonal terms are zero
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if (i, j, k) == (i2, j2, k2):
                                continue
                            val = 0.0
                            for r in range(rank):
                                val += Ua[i * m + j, r] * Va[j2 * p + k2, r] * Wa[k2 * n + i2, r]
                            if abs(val) > tol:
                                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    n, m, p = 2, 3, 8
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = len(best["U"][0])
    
    # Try structural decomposition: split 3×8 into two 3×4 blocks
    # If we have a rank-r decomposition for 2×3 × 3×4, we can use it twice
    # to get rank 2r for 2×3 × 3×8
    # Schoolbook for 2×3 × 3×4 is 24, so 2*24=48 (same as overall schoolbook)
    # We need to find a rank < 24 decomposition for 2×3 × 3×4 to improve
    
    # Try a small search for 2×3 × 3×4 with rank < 24
    n2, m2, p2 = 2, 3, 4
    schoolbook_sub_rank = n2 * m2 * p2  # 24
    
    # Try ranks from 20 up to 23 for the sub-problem
    found_sub = None
    for target_rank in range(20, schoolbook_sub_rank):
        if time.time() > deadline - 20:
            break
        
        # Use random initialization and iterative refinement
        # This is a heuristic search
        np.random.seed(42)
        for attempt in range(50):
            if time.time() > deadline - 20:
                break
            
            U_sub = np.random.randn(n2 * m2, target_rank)
            V_sub = np.random.randn(m2 * p2, target_rank)
            W_sub = np.random.randn(p2 * n2, target_rank)
            
            # Simple verification: check if this random decomposition works
            # (unlikely, but let's check)
            valid = True
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        val = 0.0
                        for r in range(target_rank):
                            val += U_sub[i * m2 + j, r] * V_sub[j * p2 + k, r] * W_sub[k * n2 + i, r]
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                found_sub = (U_sub, V_sub, W_sub, target_rank)
                break
        if found_sub:
            break
    
    if found_sub:
        U_sub, V_sub, W_sub, sub_rank = found_sub
        # Compose two copies for 3×8
        total_rank = 2 * sub_rank
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        for r in range(sub_rank):
            # First 3×4 block (k=0..3)
            for i in range(n):
                for j in range(m):
                    U[i * m + j][r] = U_sub[i * m2 + j, r]
                    W[0 * n + i][r] = W_sub[0 * n2 + i, r]  # k=0
                for k in range(4):
                    for j in range(m):
                        V[j * p + k][r] = V_sub[j * m2 + k, r]
            # Second 3×4 block (k=4..7)
            for i in range(n):
                for j in range(m):
                    U[i * m + j][sub_rank + r] = U_sub[i * m2 + j, r]
                    W[4 * n + i][sub_rank + r] = W_sub[0 * n2 + i, r]  # k=4 -> local k=0
                for k in range(4):
                    for j in range(m):
                        V[j * p + 4 + k][sub_rank + r] = V_sub[j * m2 + k, r]
        
        candidate = {"U": U, "V": V, "W": W}
        # Verify
        if _verify(U, V, W, n, m, p):
            if total_rank < best_rank:
                best = candidate
                best_rank = total_rank
    
    # Return the best found
    return best
# EVOLVE-BLOCK-END
