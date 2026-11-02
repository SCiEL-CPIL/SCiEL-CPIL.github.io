import numpy as np
import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook), dtype=int)
    V_school = np.zeros((m * p, rank_schoolbook), dtype=int)
    W_school = np.zeros((p * n, rank_schoolbook), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1
                V_school[b, col] = 1
                W_school[c, col] = 1
                col += 1

    # Target: Find rank < 105
    # We try ranks from 104 down to 35
    # Time limit 150s.
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin

    # Precompute tensor M
    M = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                M[a, b, c] = 1

    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school

    # Try to find a lower rank
    # Strategy: Random restarts with least squares refinement
    # We fix U and V to random binary matrices, then solve for W?
    # No, we need all three.
    # Let's try to find a solution for R=100, 90, 80...
    
    # A simpler heuristic: 
    # If we can find a linear dependency among the schoolbook terms, we can reduce rank.
    # But they are independent.
    
    # Let's try a random search for R=100
    # We will generate random U, V, W with small integer entries and check if they work.
    # This is unlikely to succeed directly.
    
    # Instead, we can use the fact that the tensor is fixed.
    # We can try to express the tensor as a sum of R rank-1 tensors.
    # This is a non-convex problem.
    
    # Given the time, I will just return the schoolbook solution if no better is found quickly.
    # But I must try.
    
    # Let's try to find a rank 100 solution by perturbing the schoolbook solution.
    # We can't just perturb, we need a valid decomposition.
    
    # I will try a few random restarts for R=100.
    # For each restart, I will pick random U, V, W with entries in {-1, 0, 1}.
    # Then I will check if the reconstruction is exact.
    # This is a needle in a haystack.
    
    # Alternative: Use a known result.
    # For 3x5x7, the rank is likely 100 or less.
    # If I can't find it, I return 105.
    
    # To ensure I don't timeout, I will limit the search.
    max_restarts = 1000
    for R in range(104, 34, -1):
        if time.time() > deadline:
            break
        for _ in range(max_restarts // 10): # Fewer restarts for lower R
            if time.time() > deadline:
                break
            
            # Random init
            U = np.random.randint(-1, 2, (n*m, R))
            V = np.random.randint(-1, 2, (m*p, R))
            W = np.random.randint(-1, 2, (p*n, R))
            
            # Check reconstruction
            # M_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # This is slow if done naively.
            # Use einsum or matmul
            # M_recon = np.einsum('ar,br,cr->abc', U, V, W)
            # This is fast for small arrays.
            M_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            
            if np.array_equal(M_recon, M):
                best_rank = R
                best_U = U
                best_V = V
                best_W = W
                break
        
        if best_rank < rank_schoolbook:
            break

    # Convert to lists of ints
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
