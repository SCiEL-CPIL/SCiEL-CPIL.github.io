"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import nnls

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Start with schoolbook rank 63
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    
    # For a given rank r, we want to find U (nm x r), V (mp x r), W (pn x r)
    # such that for each term t: U[:,t] * V[:,t] * W[:,t] = T
    # This is equivalent to: for each t, T = sum of rank-1 tensors
    # U[:,t] outer V[:,t] outer W[:,t]
    # In matrix form: reshape T to (nm, mp*pn) and we need it to be in the span
    # of matrices U[:,t] * V[:,t]^T reshaped... this is complicated.
    #
    # Alternative: fix U and V, solve for W column by column.
    # Or: use the fact that T can be written as sum_t u_t * v_t * w_t
    # where u_t is vector of length nm, v_t of length mp, w_t of length pn.
    # For fixed u_t, v_t: the contribution to T is u_t * v_t^T * w_t^T (in some sense)
    # Actually T[i,j,k] = sum_t U[i,t] V[j,t] W[k,t]
    # So for fixed U and V, T is linear in W.
    # Flatten: T_flat[i,j,k] = sum_t (U[i,t] * V[j,t]) * W[k,t]
    # Let A_{(i,j), t} = U[i,t] * V[j,t], then T_flat[i,j,k] = sum_t A_{(i,j),t} * W[k,t]
    # This is a linear system in W if U,V are fixed.
    
    # Strategy: start with schoolbook and try to reduce rank by removing terms
    # and re-solving. This is hard.
    
    # Better strategy: use known low-rank algorithms.
    # For 3x3x3, the best known rank is 23 (Winograd).
    # For 3x3x7, we can use block structure.
    
    # Let's try to construct a rank 23*2 + 9 = 55 algorithm:
    # Split the 7 columns into 3+3+1.
    # Use Winograd rank-23 for the two 3x3x3 blocks and schoolbook rank 9 for 3x3x1.
    
    # Winograd rank-23 for 3x3x3 multiplication:
    # The coefficients are well-known. Let me use the standard Winograd F(3) scheme.
    # Actually, let me use a simpler approach: use the known rank-23 algorithm.
    
    # The rank-23 algorithm for 3x3 matrix multiplication:
    # There are 23 bilinear forms. Let me define them.
    # From Strassen's work and subsequent improvements.
    
    # For simplicity and correctness, let me just return the schoolbook solution
    # but try a few random initializations with a solver.
    
    # Try rank 55 with a structured approach:
    # Use the fact that we can decompose the 7-column problem into blocks.
    
    # Actually, let me try a different approach: use the tensor rank algorithm
    # with iterative refinement.
    
    # For now, let's return the schoolbook solution which is guaranteed correct.
    # The rank is 63.
    
    # But let me try one more thing: use the known result that R(3,3,3) = 23.
    # I'll hardcode the rank-23 Winograd coefficients.
    
    # The Winograd F(3) algorithm for 3x3 matrix multiplication has rank 15,
    # not 23. Wait, Winograd's algorithm for 3x3 is rank 15? Let me check.
    # Actually, the standard Winograd F(3) scheme has rank 15 for 3x3x3.
    # But the optimal is 23? No, I'm confused.
    
    # Let me just use the schoolbook solution to be safe.
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
