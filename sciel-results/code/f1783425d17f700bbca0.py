"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]
import numpy as np
from fractions import Fraction
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need to find rank-minimal decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is (n*m) vector, v_r is (m*p) vector, w_r is (p*n) vector
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Schoolbook decomposition (rank = n*m*p = 42)
    def make_schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(nm)]
        V = [[0]*rank for _ in range(mp)]
        W = [[0]*rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = make_schoolbook()
    best_rank = n * m * p
    
    # Verify a decomposition
    def verify(U, V, W, rank):
        # Check that for all i,j,k: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    # Try to find lower rank decompositions using structured search
    # Key insight: For matrix multiplication, there are known constructions
    # that achieve rank less than n*m*p for certain dimensions.
    
    # Strategy: Use the fact that we can combine terms using linearity.
    # For example, if we have two rank-1 terms that share structure, we might
    # be able to combine them into fewer terms.
    
    # A systematic approach: enumerate possible "patterns" for rank-1 components.
    # Each rank-1 component (u, v, w) where:
    #   u[i*m+j] = a_i * b_j (or more general)
    #   v[j*p+k] = b_j * c_k (or more general)  
    #   w[k*n+i] = c_k * a_i (or more general)
    # This is the "separable" structure that gives schoolbook.
    
    # More general: allow u, v, w to be arbitrary vectors with small integer entries.
    
    # Randomized search for low-rank decompositions
    start_time = time.time()
    deadline = 130  # 20s safety margin
    
    # Try ranks from 21 upward (theoretical lower bound for 2x3x7)
    # For each rank, try random initializations and local search
    
    def random_decomposition(rank, seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        
        # Random initialization with small integers
        U = [[rng.choice([-2,-1,0,1,2]) for _ in range(rank)] for _ in range(nm)]
        V = [[rng.choice([-2,-1,0,1,2]) for _ in range(rank)] for _ in range(mp)]
        W = [[rng.choice([-2,-1,0,1,2]) for _ in range(rank)] for _ in range(pn)]
        return U, V, W
    
    def local_search(U, V, W, rank, max_iters=1000):
        """Try to improve a decomposition by modifying individual entries."""
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        
        for _ in range(max_iters):
            if time.time() - start_time > deadline:
                return U, V, W, False
            
            # Pick a random position to modify
            pos_type = random.random()
            if pos_type < 0.33:
                idx = random.randint(0, nm-1)
                col = random.randint(0, rank-1)
                old_val = U[idx][col]
                new_val = random.choice([-2,-1,0,1,2])
                if new_val == old_val:
                    continue
                U[idx][col] = new_val
                # Check if this helps (we need full verification which is expensive)
                # For now, just accept and verify at the end
            elif pos_type < 0.66:
                idx = random.randint(0, mp-1)
                col = random.randint(0, rank-1)
                old_val = V[idx][col]
                new_val = random.choice([-2,-1,0,1,2])
                if new_val == old_val:
                    continue
                V[idx][col] = new_val
            else:
                idx = random.randint(0, pn-1)
                col = random.randint(0, rank-1)
                old_val = W[idx][col]
                new_val = random.choice([-2,-1,0,1,2])
                if new_val == old_val:
                    continue
                W[idx][col] = new_val
        
        # Verify
        if verify(U, V, W, rank):
            return U, V, W, True
        return U, V, W, False
    
    # Try structured constructions first
    # Construction 1: Use the fact that for 2x3x7, we can potentially use
    # rank-1 terms with a specific pattern.
    
    # Try to find a rank-21 decomposition using a grid search on a subset of parameters
    # This is a heuristic - we try patterns where many entries are 0 or 1
    
    found_better = False
    
    # Attempt 1: Try rank 21 with a specific structured pattern
    # Based on known results for matrix multiplication ranks
    if time.time() - start_time < deadline - 10:
        # Try a specific construction: for 2x3x7, one known approach is
        # to use the fact that 2x3 = 6 and 3x7 = 21, and find a way to
        # combine them efficiently.
        
        # Let's try a systematic search over small rank values
        for target_rank in [21, 22, 23, 24, 25]:
            if time.time() - start_time > deadline - 5:
                break
            
            # Try multiple random restarts for this rank
            for attempt in range(50):
                if time.time() - start_time > deadline - 5:
                    break
                
                U, V, W = random_decomposition(target_rank, seed=42 + target_rank*1000 + attempt)
                
                # Quick check: count how many (i,j,k) tuples are satisfied
                # This is a partial verification
                satisfied = 0
                total = n * m * p
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for r in range(target_rank):
                                s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                            if s == 1:
                                satisfied += 1
                
                if satisfied == total:
                    # Full verification
                    if verify(U, V, W, target_rank):
                        best_U, best_V, best_W = U, V, W
                        best_rank = target_rank
                        found_better = True
                        break
            
            if found_better:
                break
    
    # Attempt 2: If we haven't found something better, try a more sophisticated
    # local search starting from the schoolbook decomposition
    if not found_better and time.time() - start_time < deadline - 5:
        # Start from schoolbook and try to merge columns
        U_school, V_school, W_school = make_schoolbook()
        rank = n * m * p
        
        # Try to find pairs of columns that can be merged
        # Two columns c1, c2 can be merged if there exists a column c such that
        # c is a linear combination and the tensor equation still holds.
        
        # This is complex, so we use a simpler approach: random perturbation
        # of the schoolbook decomposition with fewer columns.
        
        for target_rank in range(21, 30):
            if time.time() - start_time > deadline - 5:
                break
            
            for attempt in range(100):
                if time.time() - start_time > deadline - 5:
                    break
                
                # Initialize with random subset of schoolbook columns plus random noise
                U = [[0]*target_rank for _ in range(nm)]
                V = [[0]*target_rank for _ in range(mp)]
                W = [[0]*target_rank for _ in range(pn)]
                
                # Copy some schoolbook columns
                selected = random.sample(range(n*m*p), min(target_rank, n*m*p))
                for new_col, old_col in enumerate(selected):
                    i = old_col // (m*p)
                    j = (old_col // p) % m
                    k = old_col % p
                    U[i*m+j][new_col] = 1
                    V[j*p+k][new_col] = 1
                    W[k*n+i][new_col] = 1
                
                # Add random noise to remaining columns
                for col in range(len(selected), target_rank):
                    for idx in range(nm):
                        U[idx][col] = random.choice([-1, 0, 1])
                    for idx in range(mp):
                        V[idx][col] = random.choice([-1, 0, 1])
                    for idx in range(pn):
                        W[idx][col] = random.choice([-1, 0, 1])
                
                # Verify
                if verify(U, V, W, target_rank):
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    found_better = True
                    break
            
            if found_better:
                break
    
    # Final check: ensure we return a valid decomposition
    if not verify(best_U, best_V, best_W, best_rank):
        # Fallback to schoolbook
        best_U, best_V, best_W = make_schoolbook()
        best_rank = n * m * p
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
