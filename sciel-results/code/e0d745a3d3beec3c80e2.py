"""
Matrix multiplication tensor decomposition for 2x5 * 5x6.

Strategy:
1. Start with the schoolbook decomposition (rank 60).
2. Use a greedy rank-reduction algorithm:
   - At each step, try to merge two columns of the decomposition
     by finding a linear combination that preserves the tensor structure.
   - Specifically, we look for columns c1, c2 such that there exists
     a scalar alpha where the "product" of the column vectors in the
     tensor space can be combined.
3. The key insight: two columns (u1, v1, w1) and (u2, v2, w2) can be
   merged into a single column if the tensor T = u1⊗v1⊗w1 + u2⊗v2⊗w2
   can be written as u'⊗v'⊗w' for some u', v', w'. This is equivalent
   to the two rank-1 tensors being "compatible" in a way that allows
   a rank-1 sum.

A more practical approach: use the fact that for matrix multiplication,
the tensor is T[i,j,k] = 1 for all (i,j,k). We can try to find a
decomposition with fewer terms by exploiting symmetries.

Actually, let me use a different structural approach:
- Use the fact that 2x5x6 matrix multiplication can be decomposed
  by splitting the middle dimension and using block structure.

Let me implement a proper search using the "tensor rank" reduction
technique where we try to find linear dependencies among the columns.

The most reliable approach within time limits:
1. Start with schoolbook (rank 60).
2. For each pair of columns, check if they can be merged by solving
   a system of equations. If u1⊗v1⊗w1 + u2⊗v2⊗w2 = u'⊗v'⊗w',
   then we reduce rank by 1.

For two rank-1 tensors to sum to a rank-1 tensor, we need:
u1⊗v1⊗w1 + u2⊗v2⊗w2 = u'⊗v'⊗w'

This is a strong condition. A simpler sufficient condition:
If w1 and w2 are linearly dependent (or one is a scalar multiple),
and similarly for u and v, then we can merge.

Let me implement a practical greedy algorithm that tries to reduce rank
by finding and eliminating redundant columns.
"""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Safety margin before 120s limit
    
    # Initialize with schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Convert to numpy for efficient operations
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    best_U, best_V, best_W = U_np, V_np, W_np
    best_rank = rank
    
    # Greedy rank reduction: try to merge columns
    # Two columns can be merged if their corresponding rank-1 tensors
    # sum to another rank-1 tensor.
    
    # For efficiency, we'll try a limited number of merge attempts
    max_iterations = 200
    iteration = 0
    
    current_rank = rank
    U_cur = U_np.copy()
    V_cur = V_np.copy()
    W_cur = W_np.copy()
    
    while current_rank > 1 and iteration < max_iterations:
        if time.time() > deadline - 5:
            break
        iteration += 1
        
        merged = False
        # Try to find a pair of columns that can be merged
        for c1 in range(current_rank):
            if time.time() > deadline - 5:
                break
            for c2 in range(c1 + 1, current_rank):
                # Check if columns c1 and c2 can be merged
                # The tensor contribution is:
                # T = u1⊗v1⊗w1 + u2⊗v2⊗w2
                # We want to see if this equals u'⊗v'⊗w'
                
                u1 = U_cur[:, c1]
                v1 = V_cur[:, c1]
                w1 = W_cur[:, c1]
                u2 = U_cur[:, c2]
                v2 = V_cur[:, c2]
                w2 = W_cur[:, c2]
                
                # A sufficient condition for mergeability:
                # If w1 and w2 are proportional, and u1 and u2 are proportional
                # (or v1 and v2 are proportional), we can merge.
                
                # Check if w1 and w2 are linearly dependent
                if np.allclose(w2, w1 * np.sum(w1 * w2) / (np.sum(w1 * w1) + 1e-10), atol=1e-6):
                    # w2 ≈ alpha * w1
                    alpha = np.sum(w1 * w2) / (np.sum(w1 * w1) + 1e-10)
                    # Then T = (u1 + alpha*u2) ⊗ v1 ⊗ w1 + (u2 - u2) ⊗ v2 ⊗ w2
                    # Actually: u1⊗v1⊗w1 + u2⊗v2⊗w2 = u1⊗v1⊗w1 + u2⊗v2⊗(alpha*w1)
                    # = (u1⊗v1 + alpha*u2⊗v2) ⊗ w1
                    # This is NOT rank 1 unless u1⊗v1 + alpha*u2⊗v2 is rank 1.
                    
                    # Check if u1⊗v1 + alpha*u2⊗v2 is rank 1
                    # Form the matrix M = u1*v1^T + alpha*u2*v2^T
                    M = np.outer(u1, v1) + alpha * np.outer(u2, v2)
                    # Check if M has rank 1
                    if M.size > 0:
                        try:
                            sv = np.linalg.svd(M, compute_uv=False)
                            if len(sv) >= 1 and sv[0] > 1e-6:
                                if len(sv) == 1 or sv[1] < 1e-6 * sv[0]:
                                    # Rank 1! We can merge.
                                    # New column: u' = u1, v' = v1, w' = w1
                                    # But we need to absorb the second term.
                                    # Actually, the merged tensor is M ⊗ w1
                                    # We can represent this as a single column if
                                    # we set u' = u1, v' = v1, and adjust...
                                    # This gets complicated. Let me use a simpler approach.
                                    pass
                        except:
                            pass
                elif np.allclose(w1, w2 * np.sum(w1 * w2) / (np.sum(w2 * w2) + 1e-10), atol=1e-6):
                    # w1 ≈ alpha * w2
                    alpha = np.sum(w1 * w2) / (np.sum(w2 * w2) + 1e-10)
                    M = np.outer(u1, v1) + np.outer(u2, v2) * alpha
                    if M.size > 0:
                        try:
                            sv = np.linalg.svd(M, compute_uv=False)
                            if len(sv) >= 1 and sv[0] > 1e-6:
                                if len(sv) == 1 or sv[1] < 1e-6 * sv[0]:
                                    pass
                        except:
                            pass
                
                # Try a more direct approach: solve for merge
                # Assume we can write u1⊗v1⊗w1 + u2⊗v2⊗w2 = u'⊗v'⊗w'
                # This is a system of equations. Let's try a specific case:
                # If u1 = u2 (up to scale), then we can merge.
                if np.allclose(u1, u2, atol=1e-6):
                    # u1 = u2, so T = u1⊗(v1⊗w1 + v2⊗w2)
                    # Need v1⊗w1 + v2⊗w2 to be rank 1
                    M = np.outer(v1, w1) + np.outer(v2, w2)
                    if M.size > 0:
                        try:
                            sv = np.linalg.svd(M, compute_uv=False)
                            if len(sv) >= 1 and sv[0] > 1e-6:
                                if len(sv) == 1 or sv[1] < 1e-6 * sv[0]:
                                    # Merge: new column has u' = u1, and v', w' from SVD
                                    # v' = svd_left[0], w' = svd_right[0] * sv[0]
                                    Utmp, S, Vtmp = np.linalg.svd(M, full_matrices=False)
                                    v_new = Utmp[:, 0]
                                    w_new = Vtmp[0, :] * S[0]
                                    
                                    # Create new column
                                    U_cur[:, c1] = u1
                                    V_cur[:, c1] = v_new
                                    W_cur[:, c1] = w_new
                                    
                                    # Remove column c2
                                    U_cur = np.delete(U_cur, c2, axis=1)
                                    V_cur = np.delete(V_cur, c2, axis=1)
                                    W_cur = np.delete(W_cur, c2, axis=1)
                                    current_rank -= 1
                                    merged = True
                                    break
                        except:
                            pass
                elif np.allclose(v1, v2, atol=1e-6):
                    # v1 = v2, so T = (u1⊗w1 + u2⊗w2) ⊗ v1
                    M = np.outer(u1, w1) + np.outer(u2, w2)
                    if M.size > 0:
                        try:
                            Utmp, S, Vtmp = np.linalg.svd(M, full_matrices=False)
                            if len(S) >= 1 and S[0] > 1e-6:
                                if len(S) == 1 or S[1] < 1e-6 * S[0]:
                                    u_new = Utmp[:, 0]
                                    w_new = Vtmp[0, :] * S[0]
                                    
                                    U_cur[:, c1] = u_new
                                    V_cur[:, c1] = v1
                                    W_cur[:, c1] = w_new
                                    
                                    U_cur = np.delete(U_cur, c2, axis=1)
                                    V_cur = np.delete(V_cur, c2, axis=1)
                                    W_cur = np.delete(W_cur, c2, axis=1)
                                    current_rank -= 1
                                    merged = True
                                    break
                        except:
                            pass
                elif np.allclose(w1, w2, atol=1e-6):
                    # w1 = w2, so T = (u1⊗v1 + u2⊗v2) ⊗ w1
                    M = np.outer(u1, v1) + np.outer(u2, v2)
                    if M.size > 0:
                        try:
                            Utmp, S, Vtmp = np.linalg.svd(M, full_matrices=False)
                            if len(S) >= 1 and S[0] > 1e-6:
                                if len(S) == 1 or S[1] < 1e-6 * S[0]:
                                    u_new = Utmp[:, 0]
                                    v_new = Vtmp[0, :] * S[0]
                                    
                                    U_cur[:, c1] = u_new
                                    V_cur[:, c1] = v_new
                                    W_cur[:, c1] = w1
                                    
                                    U_cur = np.delete(U_cur, c2, axis=1)
                                    V_cur = np.delete(V_cur, c2, axis=1)
                                    W_cur = np.delete(W_cur, c2, axis=1)
                                    current_rank -= 1
                                    merged = True
                                    break
                        except:
                            pass
            
            if merged:
                break
        
        if not merged:
            # No more simple merges possible with this heuristic
            # Try a random perturbation approach
            if time.time() > deadline - 10:
                break
            
            # Try to find a linear combination of all columns that gives a new rank-1 tensor
            # This is more complex, so we'll stop here for now
            break
    
    # Verify the decomposition
    if current_rank < best_rank:
        # Check if the decomposition is valid
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Should equal 1 for all (i,j,k)
        valid = True
        for i in range(n):
            if not valid:
                break
            for j in range(m):
                if not valid:
                    break
                for k in range(p):
                    val = 0.0
                    for c in range(current_rank):
                        val += U_cur[i * m + j, c] * V_cur[j * p + k, c] * W_cur[k * n + i, c]
                    if abs(val - 1.0) > 1e-4:
                        valid = False
                        break
        
        if valid:
            best_U = U_cur
            best_V = V_cur
            best_W = W_cur
            best_rank = current_rank
    
    # Convert back to lists with integer/half-integer entries
    def snap_to_half_integer(x):
        return round(x * 2) / 2.0
    
    U_out = [[snap_to_half_integer(best_U[i, c]) for c in range(best_rank)] for i in range(n * m)]
    V_out = [[snap_to_half_integer(best_V[i, c]) for c in range(best_rank)] for i in range(m * p)]
    W_out = [[snap_to_half_integer(best_W[i, c]) for c in range(best_rank)] for i in range(p * n)]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
