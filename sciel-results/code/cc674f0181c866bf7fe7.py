"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    # Schoolbook decomposition as baseline (rank 42)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    best_rank = n * m * p
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    # Strategy: Algebraic/Coding Theory inspired search.
    # We look for a rank-r decomposition where r < 42.
    # The tensor is T[i,j,k] = 1.
    # We want sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k.
    # This is a system of 42 equations.
    # Variables: U (nm*r), V (mp*r), W (pn*r).
    # We restrict entries to small integers/half-integers to facilitate exactness.
    
    # Known lower bound for 2x3x7 is 21 (from max(nm, mp, pn) / 2? No, max is 21).
    # Actually, max(nm, mp, pn) = 21.
    # So rank >= 21.
    # We try ranks from 21 upwards.
    
    # To improve the search, we use a "structured" random initialization.
    # Instead of pure random, we try to align with the Kronecker product structure 
    # or use a "parity-check" like constraint satisfaction approach.
    
    # For each target rank, we try to solve the system using a greedy/local search 
    # or randomized restarts with small integer values.
    
    min_rank = 21
    max_search_rank = best_rank - 1  # Try to improve
    
    # We will attempt to find a solution for each rank in [min_rank, max_search_rank]
    # We prioritize lower ranks.
    
    # Improved generation strategy:
    # 1. Start with sparse matrices (mostly 0, -1, 1)
    # 2. Use local search to flip entries to satisfy equations
    # 3. Allow half-integers if integers fail
    
    def gen_sparse_matrix(rows, cols):
        """Generate a sparse matrix with entries in {-1, 0, 1}."""
        mat = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if random.random() < 0.4:  # 40% density
                    mat[r][c] = random.choice([-1, 1])
        return mat

    def local_search(U, V, W, rank, max_steps=1000):
        """
        Try to fix the decomposition by flipping individual entries.
        Returns True if a valid decomposition is found.
        """
        # Compute residual for each (i,j,k)
        # Residual[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] - 1
        
        # We'll maintain the current sum for each (i,j,k)
        # sum_ijk = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        
        # Initialize sums
        sums = [[[0 for _ in range(p)] for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    sums[i][j][k] = s
        
        # Count how many are correct
        def count_correct():
            cnt = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if sums[i][j][k] == 1:
                            cnt += 1
            return cnt
        
        correct_count = count_correct()
        if correct_count == n * m * p:
            return True
        
        # List of all (i,j,k) that are incorrect
        def get_incorrect():
            inc = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if sums[i][j][k] != 1:
                            inc.append((i, j, k))
            return inc
        
        # Local search: randomly flip an entry in U, V, or W
        # and update the sums accordingly
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            # Pick a random matrix and a random entry
            mat_choice = random.choice([0, 1, 2])  # 0=U, 1=V, 2=W
            
            if mat_choice == 0:
                r = random.randint(0, nm - 1)
                c = random.randint(0, rank - 1)
                old_val = U[r][c]
                new_val = -old_val if old_val != 0 else random.choice([-1, 1])
                if new_val == old_val:
                    continue
                
                # Update sums: U[i*m+j][c] changes
                i = r // m
                j = r % m
                delta = new_val - old_val
                
                # For each k, sums[i][j][k] += delta * V[j*p+k][c] * W[k*n+i][c]
                for k in range(p):
                    sums[i][j][k] += delta * V[j * p + k][c] * W[k * n + i][c]
                
                U[r][c] = new_val
                
            elif mat_choice == 1:
                r = random.randint(0, mp - 1)
                c = random.randint(0, rank - 1)
                old_val = V[r][c]
                new_val = -old_val if old_val != 0 else random.choice([-1, 1])
                if new_val == old_val:
                    continue
                
                # Update sums: V[j*p+k][c] changes
                j = r // p
                k = r % p
                delta = new_val - old_val
                
                # For each i, sums[i][j][k] += U[i*m+j][c] * delta * W[k*n+i][c]
                for i in range(n):
                    sums[i][j][k] += U[i * m + j][c] * delta * W[k * n + i][c]
                
                V[r][c] = new_val
                
            else:
                r = random.randint(0, pn - 1)
                c = random.randint(0, rank - 1)
                old_val = W[r][c]
                new_val = -old_val if old_val != 0 else random.choice([-1, 1])
                if new_val == old_val:
                    continue
                
                # Update sums: W[k*n+i][c] changes
                k = r // n
                i = r % n
                delta = new_val - old_val
                
                # For each j, sums[i][j][k] += U[i*m+j][c] * V[j*p+k][c] * delta
                for j in range(m):
                    sums[i][j][k] += U[i * m + j][c] * V[j * p + k][c] * delta
                
                W[r][c] = new_val
            
            # Check if we solved it
            # Quick check: count correct
            # Optimization: just check if all are 1
            # For speed, we can track a "bad count"
            # But for simplicity, let's just check every 100 steps or if we think we're close
            
            # Actually, let's just check if all are correct every few steps
            if step % 50 == 0 or step == max_steps - 1:
                if all(sums[i][j][k] == 1 for i in range(n) for j in range(m) for k in range(p)):
                    return True
        
        # Final check
        if all(sums[i][j][k] == 1 for i in range(n) for j in range(m) for k in range(p)):
            return True
        
        return False

    for target_rank in range(min_rank, max_search_rank + 1):
        if time.time() > deadline - 5:
            break
        
        # Number of attempts per rank
        # For lower ranks, more attempts might be needed, but probability is low.
        # For ranks close to 42, it's easier.
        # Let's allocate attempts based on rank distance from 42.
        if target_rank < 30:
            attempts = 50
        elif target_rank < 40:
            attempts = 100
        else:
            attempts = 200
        
        found = False
        
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
            
            # Generate random matrices with small integer entries
            U = gen_sparse_matrix(nm, target_rank)
            V = gen_sparse_matrix(mp, target_rank)
            W = gen_sparse_matrix(pn, target_rank)
            
            # Try local search to fix it
            if local_search(U, V, W, target_rank, max_steps=500):
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                found = True
                break
        
        if found:
            # If we found a solution, we can continue to try for even lower ranks
            # but we must ensure we don't exceed time.
            continue
    
    return best
# EVOLVE-BLOCK-END
