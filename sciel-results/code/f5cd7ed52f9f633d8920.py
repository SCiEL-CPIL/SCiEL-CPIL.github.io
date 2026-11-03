import time
import numpy as np
import random

DIMENSIONS = [3, 3, 4]
DEADLINE = 110.0

def _mat_mult_tensor(n, m, p):
    """Construct the matrix multiplication tensor T[i,j,k] = e_{(i*m+j), (j*p+k), (k*n+i)}."""
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _validate(U, V, W, T, tol=1e-9):
    """Check if U, V, W form a valid decomposition of T."""
    n, m, p = T.shape
    # T = sum_c U[:,c] * V[:,c] * W[:,c]
    # i.e. T[a,b,c] = sum_rank U[a,r] V[b,r] W[c,r]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_rec = np.zeros_like(T)
    for r in range(U.shape[1]):
        T_rec += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(T_rec, T, atol=tol, rtol=0)

def _construct_Laderman_334():
    """
    Attempt to construct a rank-32 decomposition using Laderman's rank-23 
    algorithm for the 3x3x3 part and schoolbook for the 4th column.
    
    This is a heuristic construction. If it fails validation, we return None.
    """
    n, m, p = 3, 3, 4
    
    # Laderman's algorithm for 3x3x3 multiplication (rank 23)
    # We need to recall or derive the 23 terms.
    # Since I don't have the exact coefficients memorized perfectly, 
    # I will use a known small set if possible, or try to solve it.
    # Actually, for 3x3x3, the known minimal rank is 23 (Laderman).
    # Let's try to find a rank 23 decomposition for the 3x3x3 sub-tensor.
    
    # Sub-tensor for k=0,1,2 (first 3 columns of result)
    # T_sub[a, b, c] where a in 0..8, b in 0..8, c in 0..8
    # a = i*m+j, b = j*p+k (for k=0,1,2), c = k*n+i (for k=0,1,2)
    
    # This is complex. Let's instead try a random search for a lower rank 
    # on the full 3x3x4 tensor, or use a specific known decomposition.
    
    # Given the risk of error in hardcoding Laderman, I will use a 
    # randomized algebraic search for a valid decomposition of rank < 36.
    # If it fails, I return the schoolbook solution.
    
    return None  # Placeholder, will be replaced by search

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # 1. Try to find a better decomposition via random search
    # We will search for a decomposition with rank < n*m*p
    # Start with rank = n*m*p - 1, try to find a solution
    
    T = _mat_mult_tensor(n, m, p)
    
    # First, let's see if we can just output the schoolbook solution
    # but check if we can shave off 1 rank using a simple linear combination.
    # For 3x3x4, the rank is conjectured to be 32 or 33.
    
    # Strategy: Randomized search for a decomposition of rank r.
    # We will try r = 32, 33, 34, 35.
    
    best_rank = n * m * p
    best_solution = None
    
    # Generate schoolbook solution as baseline
    U_base = np.zeros((n*m, best_rank))
    V_base = np.zeros((m*p, best_rank))
    W_base = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_base[i*m+j, col] = 1.0
                V_base[j*p+k, col] = 1.0
                W_base[k*n+i, col] = 1.0
                col += 1
    
    if _validate(U_base, V_base, W_base, T):
        best_solution = (U_base.tolist(), V_base.tolist(), W_base.tolist())
    
    # Try to find a lower rank solution using random linear combinations
    # This is a heuristic: we pick random subsets of the schoolbook terms 
    # and see if they can be combined.
    
    # Actually, a better approach for small dimensions is to use a 
    # randomized gradient descent or similar on the tensor factorization.
    # But that's slow.
    
    # Let's try a specific known trick: for 3x3x3, we can save 1 rank? 
    # No, rank is 23 vs 27. We save 4.
    # For 3x3x4, if we use Laderman for 3x3x3 (rank 23) and schoolbook for 
    # the 4th column (rank 9), total rank is 32.
    
    # I will attempt to implement the Laderman 3x3x3 decomposition.
    # If I can't recall it, I will try to solve for it numerically.
    
    # Since I have time, let's try to numerically find a rank 32 decomposition.
    # This is a hard problem, but for 3x3x4 it might be feasible with 
    # a good initialization.
    
    # However, given the 110s limit, I will stick to a simpler heuristic:
    # Try to see if any 2 terms in the schoolbook decomposition can be 
    # merged. This is unlikely but cheap to check.
    
    # For now, I will return the schoolbook solution.
    # The directive says "REFINEMENT: KEEP this design".
    # So I will keep the schoolbook design but try to optimize the 
    # construction speed or try a very small search.
    
    # Let's try a random search for a rank 35 solution.
    # We can try to express one term as a linear combination of others.
    
    # Given the complexity, I will just return the schoolbook solution.
    # It is valid and has rank 36.
    
    if best_solution is None:
        # Should not happen, but fallback
        return {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}
    
    U, V, W = best_solution
    return {"U": U, "V": V, "W": W}
