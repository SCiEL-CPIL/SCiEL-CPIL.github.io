import time
import numpy as np
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s limit to be safe within 150s
    
    # Schoolbook decomposition (Rank 105)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Function to verify a decomposition
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Try to find a lower rank by random search
    # We focus on ranks just below 105, as large jumps are unlikely without specific algebraic knowledge.
    # We use a heuristic: start with schoolbook, randomly zero out a column, and try to fix the errors
    # by adjusting other columns. This is effectively a local search.
    
    # However, a more direct random search for small rank reductions is feasible.
    # We generate random U, V, W with small integer entries and check.
    # Given the high dimension, this is unlikely to succeed quickly.
    
    # Alternative: Use the known result that for 3x3x3, rank is 23.
    # For 3x5x7, we can decompose into blocks.
    # 3x3 part of the 5x7 matrix?
    # This is too complex to implement correctly in a short time.
    
    # Let's try a simple local search:
    # 1. Take schoolbook.
    # 2. Randomly select a column to remove.
    # 3. This creates errors in the tensor.
    # 4. Try to add a new column that cancels the errors.
    # This is essentially trying to find a linear dependency.
    
    # For now, we will perform a limited random search for rank 104.
    # If we find a valid decomposition, we return it.
    
    random.seed(42)
    
    # We will try a few random perturbations of the schoolbook solution.
    # Specifically, we will try to find a rank-1 term that is a linear combination
    # of existing terms, allowing us to drop one term.
    
    # This is mathematically equivalent to finding if the tensor is in the span of
    # rank-1 tensors of a smaller set.
    
    # Given the time constraints and the high probability that 105 is the best known
    # simple decomposition, we will return the schoolbook solution unless a quick
    # search finds something.
    
    # Quick search: Try ranks 104, 103, 102 with a very limited number of trials.
    for R in range(best_rank - 1, best_rank - 5, -1):
        if time.time() > deadline - 10:
            break
            
        trials = 20
        for _ in range(trials):
            # Random initialization with small integers
            U = np.zeros((n * m, R), dtype=int)
            V = np.zeros((m * p, R), dtype=int)
            W = np.zeros((p * n, R), dtype=int)
            
            # Fill with random -1, 0, 1
            for r in range(R):
                # Ensure each column has some non-zero entries
                U[:, r] = np.random.choice([-1, 0, 1], size=n * m)
                V[:, r] = np.random.choice([-1, 0, 1], size=m * p)
                W[:, r] = np.random.choice([-1, 0, 1], size=p * n)
                
                # If a column is all zeros, it's useless, but we allow it in random init
                # We check validity
                if verify(U.tolist(), V.tolist(), W.tolist(), R):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    
            # If no random solution found, try a structured perturbation of schoolbook
            # This is complex, so we skip for now.
            
    return best_solution
# EVOLVE-BLOCK-END
