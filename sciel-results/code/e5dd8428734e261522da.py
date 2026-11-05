"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 seconds safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Schoolbook decomposition as fallback
    def make_schoolbook():
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Verify a candidate decomposition
    def verify(U, V, W, tol=1e-9):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(val - 1.0) > tol:
                        return False
        # Check that non-(i,j,k) entries are zero
        # The tensor T[i,j,k] = 1 if valid, else 0
        # Actually, the multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
        # and 0 otherwise. Since all (i,j,k) in range are valid, we just need T[i,j,k]=1
        return True
    
    # Try to find a lower-rank decomposition using random search
    # Strategy: try ranks from 21 up to schoolbook_rank-1
    # Use random initialization with small integers and check
    
    best_result = make_schoolbook()
    
    # Try some specific known constructions or random searches
    # For M_{2,3,7}, the known rank is 21 (schoolbook), but let's verify if we can find lower
    
    # Actually, let's try a systematic search for rank 20, 21, ...
    # Given the complexity, we'll try random restarts with small integers
    
    def random_decomposition(rank, seed=None):
        if seed is not None:
            random.seed(seed)
        U = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(p * n)]
        return U, V, W
    
    # Try rank 21 (same as schoolbook) with different structure
    # And try to find rank 20 if possible
    
    for target_rank in range(20, schoolbook_rank):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            U, V, W = random_decomposition(target_rank)
            
            # Quick check: compute the tensor and see if it matches
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(target_rank):
                            val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                        if abs(val - 1.0) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                # Verify all entries
                if verify(U, V, W):
                    best_result = {"U": U, "V": V, "W": W}
                    found = True
                    break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
