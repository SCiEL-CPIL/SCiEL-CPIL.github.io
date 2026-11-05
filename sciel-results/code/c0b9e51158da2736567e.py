import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 120
    
    # Build schoolbook decomposition
    U_school = np.zeros((n * m, rank_schoolbook), dtype=np.float64)
    V_school = np.zeros((m * p, rank_schoolbook), dtype=np.float64)
    W_school = np.zeros((p * n, rank_schoolbook), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook
    
    # Verification function using einsum
    def verify(U, V, W, rank):
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.allclose(T_recon, np.ones((n, m, p)), atol=1e-9)
    
    # Strategy: Try to find linear dependencies in the schoolbook decomposition
    # by looking for subsets of terms that can be combined
    
    # First, try a structured approach: use the fact that we can sometimes
    # reduce rank by finding non-trivial linear relations
    
    # For 4x5x6, try to see if we can merge some (i,j,k) terms
    # by using linear combinations
    
    target_ranks = [119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 10:
            break
        
        found = False
        
        # Try random searches with various entry ranges
        for seed in range(500):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(seed * 7919 + target_rank * 1009 + 42)
            
            # Use small integer entries: -1, 0, 1
            def random_small_integers(shape, max_val=1):
                ints = np.random.randint(-max_val, max_val + 1, size=shape)
                return ints.astype(np.float64)
            
            U = random_small_integers((n * m, target_rank))
            V = random_small_integers((m * p, target_rank))
            W = random_small_integers((p * n, target_rank))
            
            if verify(U, V, W, target_rank):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    # If we haven't found anything better with pure random search,
    # try a more structured approach: start from schoolbook and try to
    # eliminate one term at a time by finding linear combinations
    
    if best_rank == rank_schoolbook:
        if time.time() < deadline - 5:
            # Try to reduce by 1: find a linear combination of 2 schoolbook terms
            # that can replace them with 1 term (unlikely but worth checking)
            # This is computationally expensive, so skip for now
            
            # Instead, try a different random strategy with half-integers
            for target_rank in [119, 118, 117]:
                if time.time() > deadline - 5:
                    break
                
                for seed in range(200):
                    if time.time() > deadline - 5:
                        break
                    
                    np.random.seed(seed * 31337 + target_rank * 99991)
                    
                    # Use half-integer entries: -0.5, 0, 0.5
                    U = np.random.choice([-0.5, 0, 0.5], size=(n * m, target_rank))
                    V = np.random.choice([-0.5, 0, 0.5], size=(m * p, target_rank))
                    W = np.random.choice([-0.5, 0, 0.5], size=(p * n, target_rank))
                    
                    if verify(U, V, W, target_rank):
                        best_U, best_V, best_W = U, V, W
                        best_rank = target_rank
                        break
                
                if best_rank < rank_schoolbook:
                    break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
