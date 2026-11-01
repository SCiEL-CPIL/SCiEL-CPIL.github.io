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


def _build_strassen_decomposition(n, m, p):
    """Build the rank-35 Strassen-based decomposition."""
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_A = [
        [1, 0, 0, 1],  # P1
        [0, 0, 1, 1],  # P2
        [1, 0, 0, 0],  # P3
        [0, 0, 0, 1],  # P4
        [1, 1, 0, 0],  # P5
        [-1, 0, 1, 0], # P6
        [0, 1, 0, -1]  # P7
    ]
    
    strassen_B = [
        [1, 0, 0, 1],  # P1 (e+h)
        [1, 0, 0, 0],  # P2 (e)
        [0, 1, 0, -1], # P3 (f-h)
        [-1, 0, 1, 0], # P4 (g-e)
        [0, 0, 0, 1],  # P5 (h)
        [1, 1, 0, 0],  # P6 (e+f)
        [0, 0, 1, 1]   # P7 (g+h)
    ]
    
    w_contrib = [
        [0] * 4 for _ in range(7)
    ]
    
    w_contrib[0][0] = 1
    w_contrib[0][3] = 1
    
    w_contrib[1][2] = 1
    w_contrib[1][3] = -1
    
    w_contrib[2][1] = 1
    w_contrib[2][3] = 1
    
    w_contrib[3][0] = 1
    w_contrib[3][2] = 1
    
    w_contrib[4][0] = -1
    w_contrib[4][1] = 1
    
    w_contrib[5][3] = 1
    
    w_contrib[6][0] = 1
    
    col = 0
    for b in range(5):
        for r in range(7):
            for i in range(4):
                U[i][col] = strassen_A[r][i]
            
            V[2*b][col] = strassen_B[r][0]
            V[2*b+1][col] = strassen_B[r][1]
            V[10+2*b][col] = strassen_B[r][2]
            V[10+2*b+1][col] = strassen_B[r][3]
            
            W[4*b][col] = w_contrib[r][0]
            W[4*b+2][col] = w_contrib[r][1]
            W[4*b+1][col] = w_contrib[r][2]
            W[4*b+3][col] = w_contrib[r][3]
            
            col += 1
            
    return U, V, W


def _try_rank_decomposition(T, n, m, p, target_rank, time_budget=12):
    """
    Try to find a rank-target_rank decomposition of tensor T.
    Uses random initialization + gradient descent on the decomposition parameters.
    """
    dim_u = n * m  # 4
    dim_v = m * p  # 20
    dim_w = p * n  # 20
    
    start_time = time.time()
    
    # Try multiple random initializations
    for trial in range(8):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization
        U = np.random.randn(dim_u, target_rank) * 0.5
        V = np.random.randn(dim_v, target_rank) * 0.5
        W = np.random.randn(dim_w, target_rank) * 0.5
        
        # Gradient descent with adaptive learning rate
        lr = 0.05
        min_lr = 1e-6
        patience = 50
        no_improve = 0
        best_loss = float('inf')
        
        for step in range(800):
            if time.time() - start_time > time_budget:
                break
            
            # Compute current tensor estimate
            T_est = np.einsum('ir,jr,kr->ijk', U, V, W)
            diff = T - T_est
            loss = np.sum(diff**2)
            
            # Track best
            if loss < best_loss:
                best_loss = loss
                no_improve = 0
                U_best = U.copy()
                V_best = V.copy()
                W_best = W.copy()
            else:
                no_improve += 1
            
            if no_improve > patience and lr < min_lr:
                break
            
            # Compute gradients
            grad_U = 2 * np.einsum('ijk,jr,kr->ir', diff, V, W)
            grad_V = 2 * np.einsum('ijk,ir,kr->jr', diff, U, W)
            grad_W = 2 * np.einsum('ijk,ir,jr->kr', diff, U, V)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Adaptive learning rate
            if no_improve > 20:
                lr *= 0.5
                if lr < min_lr:
                    break
            
            # Check convergence
            if loss < 1e-10:
                break
        
        # Final check using best found
        if 'U_best' in locals():
            T_final = np.einsum('ir,jr,kr->ijk', U_best, V_best, W_best)
            if np.max(np.abs(T - T_final)) < 1e-8:
                # Convert to list format
                U_list = U_best.tolist()
                V_list = V_best.tolist()
                W_list = W_best.tolist()
                return U_list, V_list, W_list
    
    return None


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly represents matrix multiplication."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    # Build the tensor from the decomposition
    T = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
    
    # Now verify that T represents matrix multiplication
    # T[i,j,k] where:
    # i = i_A * m + j_A (i_A in 0..n-1, j_A in 0..m-1)
    # j = j_B * p + k_B (j_B in 0..m-1, k_B in 0..p-1)
    # k = k_C * n + i_C (k_C in 0..p-1, i_C in 0..n-1)
    
    # T[i,j,k] should equal sum over mid of A[i_A, mid] * B[mid, k_C]
    # where i = i_A * m + j_A, so j_A = i % m, i_A = i // m
    # j = j_B * p + k_B, so k_B = j % p, j_B = j // p
    # k = k_C * n + i_C, so i_C = k % n, k_C = k // n
    
    # For the decomposition to be correct, we need:
    # T[i,j,k] = sum_{mid} A[i_A, mid] * B[mid, k_C]
    # where i_A = i // m, j_A = i % m, j_B = j // p, k_B = j % p, i_C = k % n, k_C = k // n
    
    # But this is only true if j_A == j_B (the middle index matches)
    # Otherwise T[i,j,k] = 0
    
    # Let's verify by checking a few random instances
    np.random.seed(42)
    
    for _ in range(20):
        # Random A (n x m) and B (m x p)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        
        # Flatten A and B according to the indexing
        A_flat = A.flatten()  # index: i_A * m + j_A
        B_flat = B.flatten()  # index: j_B * p + k_B
        
        # Compute C_flat using the tensor
        C_flat = np.zeros(p * n)
        for i in range(n * m):
            for j in range(m * p):
                for k in range(p * n):
                    C_flat[k] += A_flat[i] * B_flat[j] * T[i, j, k]
        
        # Reshape C_flat: index k_C * n + i_C
        C_check = C_flat.reshape(p, n).T
        
        if not np.allclose(C, C_check, atol=1e-8):
            return False
    
    return True
# EVOLVE-BLOCK-END
