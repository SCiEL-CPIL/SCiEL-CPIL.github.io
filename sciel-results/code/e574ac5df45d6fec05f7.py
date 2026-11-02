<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    Strategy:
    Start with the rank-35 Strassen-based decomposition (5 blocks x 7 Strassen products).
    The key insight is that all 5 blocks share the same A matrix, so the U vectors repeat.
    This creates linear dependencies in the decomposition that can be exploited.
    
    We use a structured approach:
    1. Build the rank-35 decomposition
    2. Identify that U vectors are shared across blocks
    3. Use the shared structure to find a lower-rank decomposition
    
    The tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    Since U vectors repeat, we can potentially combine terms.
    
    Specifically, for each Strassen product type r (0..6), we have 5 blocks using the same U vector.
    The V and W vectors differ per block. We can potentially combine these.
    
    However, the V vectors are supported on disjoint sets of indices (different column pairs),
    and the W vectors are also supported on disjoint sets. So the terms are actually independent.
    
    Let's verify: For block b, V is supported on indices {2b, 2b+1, 10+2b, 10+2b+1}.
    For block b', V is supported on different indices. So V vectors from different blocks are orthogonal.
    Similarly for W.
    
    This means the 35 terms are linearly independent, and rank 35 is minimal for this structure.
    
    But wait - we can potentially do better by not using the block structure.
    Let's try a direct optimization approach to find if rank < 35 is possible.
    
    Actually, let's think about this differently. The matrix multiplication tensor for
    2x2 by 2x10 has a specific structure. Let's compute its rank directly.
    
    The tensor T has shape (4, 20, 20). We can try to find its tensor rank.
    
    Lower bounds:
    - Matrix rank of any matricization
    - T as (4, 400): rank <= 4
    - T as (20, 80): rank <= 20
    - T as (20, 80): rank <= 20
    
    So the tensor rank is at least 20, but likely higher.
    
    Let's try to find a decomposition with rank < 35 using optimization.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin
    
    # Build the rank-35 decomposition first
    U35, V35, W35 = _build_strassen_decomposition(n, m, p)
    
    # Convert to numpy arrays for computation
    U35_np = np.array(U35, dtype=float)
    V35_np = np.array(V35, dtype=float)
    W35_np = np.array(W35, dtype=float)
    
    # Build the full tensor
    T = np.einsum('ir,jr,kr->ijk', U35_np, V35_np, W35_np)
    
    # Verify the tensor is correct
    # T[i,j,k] should equal sum over middle index of A[i,mid] * B[mid,k]
    # But we're just using the decomposition as-is
    
    # Try to find a lower-rank decomposition
    best_result = {"U": U35, "V": V35, "W": W35}
    best_rank = 35
    
    # The key insight: since U vectors repeat across blocks, we can potentially
    # combine the V and W vectors.
    
    # For each Strassen product type r (0..6), we have 5 blocks.
    # U vectors are the same for all 5 blocks.
    # V vectors are different (different column pairs).
    # W vectors are different (different output positions).
    
    # We can write: sum_{b=0}^{4} U_r ⊗ V_{r,b} ⊗ W_{r,b}
    # = U_r ⊗ (sum_b V_{r,b} ⊗ W_{r,b})  -- but this doesn't make sense dimensionally
    
    # Actually, we can't combine them this way because V and W have different supports.
    
    # Let's try a different approach: use the fact that the tensor has a specific structure
    # and try to find a lower-rank decomposition via optimization.
    
    # Try ranks from 28 up to 34
    for target_rank in range(28, 35):
        if time.time() > deadline - 30:
            break
        
        result = _try_rank_decomposition(T, n, m, p, target_rank, time_budget=12)
        if result is not None:
            U_new, V_new, W_new = result
            # Verify the decomposition
            if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                best_result = {"U": U_new, "V": V_new, "W": W_new}
                best_rank = target_rank
                break  # Found a valid lower rank, stop
    
    return best_result
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    Strategy:
    Start with the rank-35 Strassen-based decomposition (5 blocks x 7 Strassen products).
    Then try to find a lower-rank decomposition using structured initialization
    and improved optimization.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin
    
    # Build the rank-35 decomposition first
    U35, V35, W35 = _build_strassen_decomposition(n, m, p)
    
    # Convert to numpy arrays for computation
    U35_np = np.array(U35, dtype=float)
    V35_np = np.array(V35, dtype=float)
    W35_np = np.array(W35, dtype=float)
    
    # Build the full tensor
    T = np.einsum('ir,jr,kr->ijk', U35_np, V35_np, W35_np)
    
    # Verify the tensor is correct
    if not _verify_tensor(T, n, m, p):
        return {"U": U35, "V": V35, "W": W35}
    
    # Try to find a lower-rank decomposition
    best_result = {"U": U35, "V": V35, "W": W35}
    best_rank = 35
    
    # Try ranks from 30 up to 34 (skip lower ranks that are unlikely to work)
    for target_rank in range(30, 35):
        if time.time() > deadline - 25:
            break
        
        result = _try_rank_decomposition_improved(T, n, m, p, target_rank, 
                                                    U35_np, V35_np, W35_np, 
                                                    time_budget=15)
        if result is not None:
            U_new, V_new, W_new = result
            # Verify the decomposition
            if _verify_decomposition_fast(U_new, V_new, W_new, n, m, p):
                best_result = {"U": U_new, "V": V_new, "W": W_new}
                best_rank = target_rank
                break  # Found a valid lower rank, stop
    
    return best_result
>>>>>>> REPLACE

