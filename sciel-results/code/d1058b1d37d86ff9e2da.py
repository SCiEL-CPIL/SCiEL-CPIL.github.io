"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    Uses a structured search over low-rank tensor decompositions with rational entries.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=4
    
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    def validate(U, V, W, rank):
        """Check if U, V, W with given rank exactly decompose the multiplication tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check valid entries: T[i*m+j, j*p+k, k*n+i] should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                    if abs(val - 1) > 1e-9:
                        return False
        
        # Check that T[a,b,c] = 0 when (a,b,c) doesn't correspond to valid multiplication
        for a in range(n * m):
            i, j = divmod(a, m)
            for b in range(m * p):
                j2, k = divmod(b, p)
                if j2 != j:
                    for c in range(p * n):
                        val = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                        if abs(val) > 1e-9:
                            return False
                else:
                    valid_c = k * n + i
                    for c in range(p * n):
                        val = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                        if c == valid_c:
                            if abs(val - 1) > 1e-9:
                                return False
                        else:
                            if abs(val) > 1e-9:
                                return False
        return True
    
    # Create schoolbook solution of rank 24
    rank_schoolbook = n * m * p  # 24
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions using structured approaches
    # Key insight: we can combine schoolbook terms if they share structure
    
    # Strategy 1: Try to find a decomposition where each rank component
    # handles multiple (i,j,k) triples by exploiting shared indices
    
    # For 2x3x4, let's try to construct a decomposition manually
    # by grouping terms that share the same i, j, or k
    
    # The 24 schoolbook terms are indexed by (i,j,k) where i in {0,1}, j in {0,1,2}, k in {0,1,2,3}
    
    # Try rank 20: can we combine 4 pairs of terms?
    # Try rank 18: can we combine 6 pairs or 3 triples?
    
    # Let's try a more systematic approach: use the fact that the tensor is
    # a sum of rank-1 tensors, and try to find a minimal set
    
    # For now, try random search with structured initialization
    random.seed(42)
    
    # Candidate values for entries (small integers and half-integers)
    candidates = [0, 0.5, -0.5, 1, -1, 1.5, -1.5, 2, -2]
    
    # Try ranks from 12 to 23
    for target_rank in range(12, 24):
        if time.time() > deadline - 10:
            break
        
        # Number of attempts based on rank (lower ranks need more attempts)
        num_attempts = 30 if target_rank <= 16 else 15
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize U, V, W with sparse random values
            U = [[0.0] * target_rank for _ in range(n * m)]
            V = [[0.0] * target_rank for _ in range(m * p)]
            W = [[0.0] * target_rank for _ in range(p * n)]
            
            # Structured initialization: each rank component should be "active"
            # on a subset of indices that makes sense for matrix multiplication
            for r in range(target_rank):
                # For U: pick 1-3 random (i,j) positions
                num_u = random.choice([1, 2, 3])
                u_indices = random.sample(range(n * m), min(num_u, n * m))
                for idx in u_indices:
                    U[idx][r] = random.choice(candidates[1:])
                
                # For V: pick 1-3 random (j,k) positions
                num_v = random.choice([1, 2, 3])
                v_indices = random.sample(range(m * p), min(num_v, m * p))
                for idx in v_indices:
                    V[idx][r] = random.choice(candidates[1:])
                
                # For W: pick 1-3 random (i,k) positions
                num_w = random.choice([1, 2, 3])
                w_indices = random.sample(range(p * n), min(num_w, p * n))
                for idx in w_indices:
                    W[idx][r] = random.choice(candidates[1:])
            
            # Quick validation
            if validate(U, V, W, target_rank):
                best_solution = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
    
    # If we found a better solution, return it; otherwise return schoolbook
    if best_rank < rank_schoolbook:
        return best_solution
    else:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
