import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 63

    # 1. Construct Schoolbook (Baseline) - always valid, rank 63
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_schoolbook
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin

    # Helper to check if a solution is valid (exact decomposition with half-integer entries)
    def is_valid_solution(U, V, W, r):
        """
        Checks if U, V, W define a valid rank-r decomposition.
        Entries must be half-integers (or integers), and the tensor must be exactly 1s.
        """
        # 1. Check half-integer constraint
        def is_half_int(x):
            x2 = x * 2
            # Check closeness to integer
            return np.all(np.abs(x2 - np.round(x2)) < 1e-6)
            
        if not (is_half_int(U) and is_half_int(V) and is_half_int(W)):
            return False

        # 2. Check exact tensor decomposition
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We want T[i, j, k] == 1 for all i, j, k
        
        # Efficient computation using matrix multiplication
        # T is a 3D tensor. We can view it as a matrix unfolding or just compute directly.
        # Given small sizes, direct loop is fine, but we can be smarter.
        
        # Let's compute T efficiently.
        # T[i, j, k] = sum_c U[i*m+j, c] (V * W_elementwise)[j*p+k, c]
        
        # Let M[i, c] = U[i*m+j, c] ... no, U depends on j too.
        
        # Simplest robust check for small sizes:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum over c
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Optimization logic
    # We search for lower ranks.
    # Known lower bound for 3x3x7?
    # 3x3x3 has rank 23 (Strassen 1973).
    # 3x3x4 has rank 23? No, 3x3x4 is rank 23?
    # Actually, let's just search.
    
    # The "structured" approach from the prompt suggests SDP, but for this size,
    # a smart local search or a specialized algebraic approach might be better.
    # However, the prompt specifically asks for a structural change.
    
    # Let's try a different representation:
    # Instead of random initialization, we can try to exploit symmetries or
    # known decompositions of smaller blocks.
    
    # But a more direct structural change: 
    # Use a "sum of squares" style local search with a stronger initialization 
    # derived from the schoolbook, or use a genetic algorithm?
    
    # Given the stagnation, maybe the random restarts in the seed program are too weak 
    # or the landscape is hard.
    
    # Let's try a different optimizer: Nelder-Mead (derivative-free) is often slow, 
    # but for small dimensions it can find global minima if initialized well.
    # Or, better yet, let's try to **construct** a solution using a known algorithm 
    # for matrix multiplication if one exists for these dimensions.
    
    # For 3x3x7, the schoolbook is 63.
    # Can we do better?
    # 3x3x3 is 23.
    # 3x3x4?
    # If we multiply 3x3 by 3x7, we can think of it as 7 independent 3x3 multiplications? No, the output is 3x7.
    # It's essentially 7 separate 3x3 multiplications where the right-hand side matrices are different.
    # Wait, the tensor is T_{ijk} = A_{ik} B_{kj} ? No.
    # Standard matrix mult: C_{ik} = sum_j A_{ij} B_{jk}.
    # Here: C is 3x7. A is 3x3. B is 3x7.
    # C_{i,k} = sum_{j=0}^2 A_{i,j} B_{j,k}.
    # So the tensor is T_{i,j,k} = 1.
    # This is exactly the standard matrix multiplication tensor for m=3 (inner dimension), n=3 (rows of A), p=7 (cols of B).
    
    # The rank of the 3x3x3 tensor is 23.
    # The rank of the 3x3x7 tensor... 
    # Since the inner dimension is 3, and the other dimensions are 3 and 7.
    # We can view this as 7 separate 3x3 multiplications? No, the A matrix is shared.
    # The tensor is the Kronecker product? No.
    # T_{i,j,k} is 1 if indices are in range.
    # This is the tensor for $M_{3,3,7}$.
    
    # Known results:
    # $M_{3,3,3}$ rank 23.
    # $M_{3,3,4}$ rank ?
    # $M_{3,3,7}$ rank ?
    
    # If we can find a decomposition of rank < 63, we win.
    # Since 3x3x3 is 23, maybe 3x3x7 is not much more?
    # Actually, the rank of $M_{m,n,p}$ is at most $mn p$ (schoolbook).
    # It is also at least $\max(mn, mp, np)$.
    # Here max(9, 21, 21) = 21.
    # So rank is between 21 and 63.
    
    # Let's try to find a decomposition of rank ~ 30-40.
    
    # Strategy:
    # Use `scipy.optimize.differential_evolution` for global optimization?
    # It's expensive but robust for small dimensions.
    # Dimension: r * (9 + 21 + 21) = 51r.
    # If r=30, dim = 1530. Too high for DE.
    
    # Alternative: 
    # Use the fact that we can permute the basis.
    # Or, use a "block" approach.
    
    # Let's stick to a refined local search but with a **different initialization strategy**.
    # The seed program used random Gaussian.
    # Let's try initializing with the schoolbook columns, then **pruning** or **recombining**.
    # Or, use a "warm start" from a lower rank known solution if we had one.
    
    # Since we don't have a known solution, let's try a **structured random** initialization.
    # For example, use Rademacher matrices (entries +/- 1) scaled down.
    # This often leads to better-conditioned problems.
    
    # Also, we can try to **fix** some variables.
    # E.g., set U to be identity-like? No, that forces rank to be high.
    
    # Let's try **Nelder-Mead** on a reduced set of parameters?
    # No, let's try **simulated annealing** style local search with half-integer rounding.
    
    # Actually, the most promising structural change for "stagnation" in such tensor problems is to use **tensor train** or **MPS** (Matrix Product State) decompositions if they apply, but these are for high-dimensional tensors, not necessarily minimal rank for bilinear maps.
    
    # Let's try a different optimization algorithm: **Powell's method** or **BFGS** with careful scaling.
    # The seed used `least_squares` with `trf`.
    # Let's try `bfgs` from `scipy.optimize.minimize` on the squared residual.
    # It might handle the landscape differently.
    
    # And crucially, let's try **multiple scales** of initialization.
    
    # Let's implement a search that:
    # 1. Tries ranks from 40 up to 62.
    # 2. For each rank, tries several initializations:
    #    a. Random Gaussian
    #    b. Random Rademacher (scaled)
    #    c. Schoolbook-based (first r columns, then noise)
    # 3. Uses `scipy.optimize.minimize` with `BFGS` or `Powell`.
    # 4. Checks for half-integer validity.
    
    # Since the problem is small (63 constraints), we can be more aggressive.
    
    ranks_to_try = list(range(35, rank_schoolbook))  # Try 35 to 62
    
    # We'll use a time-based budget.
    # Let's allocate time per rank.
    # If we have 140s, and ~25 ranks, that's ~5s per rank.
    # 5s is enough for a few BFGS iterations on ~1500 variables.
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
            
        remaining_time = deadline - time.time()
        if remaining_time < 10:
            break
            
        # Number of restarts per rank
        num_restarts = 3 if r > 50 else 5
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize
            if restart == 0:
                # Random Gaussian
                scale = 1.0 / np.sqrt(r)
                x0 = np.random.randn(r * (n * m + m * p + p * n)) * scale
            elif restart == 1:
                # Random Rademacher
                x0 = np.random.choice([-1, 1], size=r * (n * m + m * p + p * n)) / np.sqrt(r)
            else:
                # Schoolbook-based
                # Take first r columns of schoolbook
                # This is tricky because the schoolbook has distinct 1s.
                # Let's just add noise to the schoolbook if r < 63.
                # Actually, schoolbook is rank 63. If r < 63, we can't just take a subset.
                # Let's use random again with different seed.
                np.random.seed(restart * 1000 + r)
                scale = 1.0 / np.sqrt(r)
                x0 = np.random.randn(r * (n * m + m * p + p * n)) * scale

            # Define objective: sum of squared residuals
            def objective(x):
                U = x[:r * n * m].reshape(n * m, r)
                V = x[r * n * m : r * n * m + r * m * p].reshape(m * p, r)
                W = x[r * n * m + r * m * p :].reshape(p * n, r)
                
                # Compute residual
                # T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                # We want T[i, j, k] == 1.
                
                # Vectorized computation:
                # For each i, j:
                #   U_row = U[i*m + j, :]  (size r)
                #   For each k:
                #     V_row = V[j*p + k, :] (size r)
                #     W_row = W[k*n + i, :] (size r)
                #     T[i, j, k] = dot(U_row, V_row * W_row)
                
                # Let's compute all T at once.
                # T is (n, m, p).
                T = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        u_row = U[i * m + j, :]
                        for k in range(p):
                            v_row = V[j * p + k, :]
                            w_row = W[k * n + i, :]
                            T[i, j, k] = np.dot(u_row, v_row * w_row)
                
                residual = T - 1.0
                return np.sum(residual ** 2)

            try:
                # Use BFGS
                res = opt.minimize(
                    objective, 
                    x0, 
                    method='BFGS', 
                    options={'maxiter': 200, 'gtol': 1e-6}
                )
                
                if res.fun < 1e-6:
                    U = res.x[:r * n * m].reshape(n * m, r)
                    V = res.x[r * n * m : r * n * m + r * m * p].reshape(m * p, r)
                    W = res.x[r * n * m + r * m * p :].reshape(p * n, r)
                    
                    if is_valid_solution(U, V, W, r):
                        best_rank = r
                        best_result = {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                        # If we find a valid solution, we can break inner loop
                        # and continue to lower ranks.
                        break
                        
            except Exception as e:
                pass
                
    return best_result

import scipy.optimize as opt
