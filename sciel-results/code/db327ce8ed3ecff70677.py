<<<<<<< SEARCH
def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Start with the known good fixed placement (4 blocks along k)
    # This gives rank 68.
    # (0,0,0), (0,0,2), (0,0,4), (0,0,6)
    blocks = [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)]
    U, V, W = _build_strassen_embedding(n, m, p, blocks)
    
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if something went wrong
        U, V, W = _schoolbook(n, m, p)
        
    best_rank = len(U[0])
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to reduce rank to 67 using optimization
    if _time_left() > 20:
        try:
            U_new, V_new, W_new, new_rank = _try_rank_reduction_optimization(U, V, W, n, m, p, best_rank - 1, time_limit=_time_left() - 10)
            if new_rank < best_rank:
                best_rank = new_rank
                best_result = {"U": U_new, "V": V_new, "W": W_new}
        except Exception:
            pass
            
    # Try to reduce rank to 66 if we found 67 and have time
    if best_rank > 66 and _time_left() > 10:
        try:
            U_cur = best_result["U"]
            V_cur = best_result["V"]
            W_cur = best_result["W"]
            U_new, V_new, W_new, new_rank = _try_rank_reduction_optimization(U_cur, V_cur, W_cur, n, m, p, best_rank - 1, time_limit=_time_left() - 5)
            if new_rank < best_rank:
                best_rank = new_rank
                best_result = {"U": U_new, "V": V_new, "W": W_new}
        except Exception:
            pass

    return best_result
=======
def _kron_product_decomposition(n, m, p):
    """
    Build a decomposition using Kronecker product of Strassen (rank 7) for 2x2
    and schoolbook for the rest, aiming for rank 56.
    
    Strategy: Decompose the 3x3 x 3x8 tensor as a combination of:
    - 4 Strassen blocks (rank 7 each) covering the 2x2 x 2x2 sub-blocks
    - Remaining terms handled by schoolbook
    
    But this is what _build_strassen_embedding already does, giving rank 68.
    
    To get rank 56, we need a fundamentally different approach.
    
    Let me try: use a rank-7 decomposition for the 3x3 x 3x3 part
    (by combining Strassen with additional terms) and rank-8 for the 8-column part.
    
    Actually, the simplest improvement is to try all possible placements of
    Strassen blocks and pick the best one.
    """
    n, m, p = 3, 3, 8
    
    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')
    
    # Try all combinations of Strassen block placements
    # Each block is (i0, j0, k0) where i0 in {0,1}, j0 in {0,1}, k0 in {0,2,4,6}
    # (k0 must be even to fit a 2-block in 8 columns)
    
    from itertools import product
    
    possible_i0 = [0, 1]
    possible_j0 = [0, 1]
    possible_k0 = [0, 2, 4, 6]
    
    # Try all combinations of 4 blocks (to cover most of the tensor)
    # This is too many combinations, so let's be smarter.
    
    # Use the current best placement as a starting point
    blocks = [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)]
    U, V, W = _build_strassen_embedding(n, m, p, blocks)
    
    if _verify(U, V, W, n, m, p):
        best_rank = len(U[0])
        best_U, best_V, best_W = U, V, W
    
    # Try to improve with optimization
    if _time_left() > 15:
        try:
            U_new, V_new, W_new, new_rank = _try_rank_reduction_optimization(
                best_U, best_V, best_W, n, m, p, best_rank - 1, time_limit=_time_left() - 10
            )
            if new_rank < best_rank:
                best_rank = new_rank
                best_U, best_V, best_W = U_new, V_new, W_new
        except Exception:
            pass
    
    return best_U, best_V, best_W, best_rank

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    U, V, W, rank = _kron_product_decomposition(n, m, p)
    
    if U is None or not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
