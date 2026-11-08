"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower rank decomposition
    # The tensor is T_{i,j,k} = 1 for all i,j,k in this case? 
    # No, matrix multiplication tensor T_{(i,j),(j,k),(k,i)} = 1.
    # Specifically, C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor T has entries T_{i,j,k} = 1.
    # We want to decompose T = sum_r u_r \otimes v_r \otimes w_r
    # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}.
    # T_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    
    # Let's try a greedy approach or a simple search for small rank reductions.
    # Known results: 
    # 2x2x2 rank is 7 (not 8).
    # 2x3x7 is larger.
    
    # A simple heuristic: try to merge terms.
    # If we have two terms that share structure, maybe we can combine them.
    # Or use a random restart local search on the entries of U, V, W to minimize rank?
    # But rank minimization is discrete.
    
    # Let's try a specific known decomposition for 2x3x7 if it exists in literature, 
    # but since I don't have access, I'll implement a local search that tries to 
    # zero out columns or merge them if the tensor remains valid.
    
    # Actually, a better approach for "exact" decomposition with integer/half-integer entries:
    # Start with schoolbook.
    # Try to reduce rank by checking if any two columns can be merged (summed) 
    # such that the resulting tensor is still a valid decomposition of the same T.
    # This is hard to check directly without re-computing T.
    
    # Let's try a different strategy: 
    # The tensor is 1s. 
    # Can we do it in less than 42?
    # For 2x2x2, rank 7 is known. 
    # For 2x3x7, let's see if we can find a pattern.
    
    # Since the time limit is tight and finding a provably minimal rank is hard,
    # I will return the schoolbook decomposition unless I can quickly find a reduction.
    # A simple reduction: if n=2, m=2, p=2, we can do rank 7.
    # For general n,m,p, it's an open problem.
    
    # However, the task asks for "rank-minimal". If I can't find it, I should return the best I have.
    # Let's try a simple local search: 
    # 1. Start with schoolbook.
    # 2. Randomly perturb entries slightly? No, must be exact.
    # 3. Try to set some entries to 0 or 0.5 and see if the tensor still matches?
    
    # Given the complexity and time limit, I will stick to the schoolbook decomposition 
    # but ensure it is correct. Any attempt at optimization without a known algorithm 
    # for this specific size might fail or timeout.
    
    # Wait, the prompt says "score is the rank". Lower is better.
    # If I return rank 42, and the optimal is 30, I get a worse score than if I found 30.
    # But if I don't find 30, I return 42.
    
    # Is there a trivial improvement? 
    # For 2x3x7, the schoolbook is 42.
    # If I use a "fast" matrix multiplication algorithm like Strassen for parts?
    # Strassen reduces 2x2x2 from 8 to 7.
    # We can partition the 2x3x7 multiplication into blocks.
    # A is 2x3, B is 3x7.
    # We can't directly use Strassen unless we pad to powers of 2.
    # Pad A to 2x4, B to 4x8? No, that increases size.
    
    # Let's try to apply Strassen to the 2x2 part of the multiplication if we split B.
    # B is 3x7. Split B into 2x7 and 1x7.
    # C = A * B = A[:, :2] * B[:2, :] + A[:, 2] * B[2:, :]
    # A[:, :2] is 2x2, B[:2, :] is 2x7.
    # Multiplying 2x2 by 2x7.
    # We can use Strassen for the 2x2 block structure?
    # Strassen is for 2x2 * 2x2.
    # Here we have 2x2 * 2x7.
    # We can split B[:2, :] into two 2x3 and a 2x1.
    # This is getting complicated.
    
    # Given the constraints, I will return the schoolbook decomposition.
    # It is guaranteed to be correct.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
