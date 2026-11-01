"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    Strategy:
    1. Build the rank-35 Strassen-based decomposition (5 blocks x 7 Strassen products).
    2. Attempt rank reduction using a structured optimization approach:
       - Use the Kronecker product structure: T = T_strassen ⊗ T_rect
       - T_strassen is the 2x2x2 matrix multiplication tensor (rank 7)
       - T_rect is the 2x1x5 rectangular multiplication tensor (rank 2)
       - The naive Kronecker product gives rank 7*2 = 14 per "super-block", but we have 5 blocks
       - Actually, the structure is: for each of 5 column-pairs, we do a 2x2 * 2x2 multiplication
       - The A matrix is shared, so U vectors repeat across blocks
       
    3. Key insight for rank reduction:
       - The tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
       - For the 5-block structure, U vectors are identical across blocks for each Strassen product
       - V and W vectors have disjoint supports across blocks
       - This means the 35 terms are linearly independent in the standard decomposition
       
    4. However, we can try to find a lower-rank decomposition by:
       - Using the fact that the tensor has a specific algebraic structure
       - Trying optimization with multiple random restarts
       - Using the rank-35 decomposition as a warm start for the optimizer
       
    5. Lower bound analysis:
       - The tensor T has shape (4, 20, 20)
       - Matricization T as (4, 400): rank <= 4
       - Matricization T as (20, 80): rank <= 20
       - So tensor rank >= 20, but likely higher due to the 5 independent blocks
       
    We'll try to find a decomposition with rank < 35 using optimization, starting from
    the rank-35 Strassen decomposition as a warm start.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin (10s buffer before 120s limit)
    
    # Build the rank-35 decomposition first
    U35, V35, W35 = _build_strassen_decomposition(n, m, p)
    
    # Convert to numpy arrays
    U35_np = np.array(U35, dtype=float)
    V35_np = np.array(V35, dtype=float)
    W35_np = np.array(W35, dtype=float)
    
    # Build the full tensor for verification
    T_target = np.einsum('ir,jr,kr->ijk', U35_np, V35_np, W35_np)
    
    # Verify the tensor is correct
    if not _verify_tensor_correctness(T_target, n, m, p):
        # Fallback: return the rank-35 decomposition
        return {"U": U35, "V": V35, "W": W35}
    
    # Try to find a lower-rank decomposition
    best_result = {"U": U35, "V": V35, "W": W35}
    best_rank = 35
    
    # Try ranks from 25 up to 34 (we know 35 works)
    for target_rank in range(25, 35):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        success = False
        for trial in range(15):
            if time.time() > deadline - 15:
                break
            
            result = _try_rank_decomposition(T_target, n, m, p, target_rank, 
                                            time_budget=8, trial=trial)
            if result is not None:
                U_new, V_new, W_new = result
                # Verify the decomposition
                if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                    best_result = {"U": U_new, "V": V_new, "W": W_new}
                    best_rank = target_rank
                    success = True
                    break
        
        if success:
            break  # Found a valid lower rank, stop
    
    return best_result


def _build_strassen_decomposition(n, m, p):
    """Build the rank-35 Strassen-based decomposition."""
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_A = [
        [1, 0, 0, 1],  # P1: (a+d)
        [0, 0, 1, 1],  # P2: (c+d)
        [1, 0, 0, 0],  # P3: a
        [0, 0, 0, 1],  # P4: d
        [1, 1, 0, 0],  # P5: (a+b)
        [-1, 0, 1, 0], # P6: (c-a)
        [0, 1, 0, -1]  # P7: (b-d)
    ]
    
    strassen_B = [
        [1, 0, 0, 1],  # P1: (e+h)
        [1, 0, 0, 0],  # P2: e
        [0, 1, 0, -1], # P3: (f-h)
        [-1, 0, 1, 0], # P4: (g-e)
        [0, 0, 0, 1],  # P5: h
        [1, 1, 0, 0],  # P6: (e+f)
        [0, 0, 1, 1]   # P7: (g+h)
    ]
    
    # W contributions for each Strassen product to the local 2x2 C matrix
    # Local C indices: 0->C00, 1->C01, 2->C10, 3->C11
    # M00 = P1 + P4 - P5 + P7
    # M01 = P3 + P5
    # M10 = P2 + P4
    # M11 = P1 - P2 + P3 + P6
    
    w_contrib = [
        [1, 0, 0, 1],   # P1: C00 +1, C11 +1
        [0, 0, 1, -1],  # P2: C10 +1, C11 -1
        [0, 1, 0, 1],   # P3: C01 +1, C11 +1
        [1, 0, 1, 0],   # P4: C00 +1, C10 +1
        [-1, 1, 0, 0],  # P5: C00 -1, C01 +1
        [0, 0, 0, 1],   # P6: C11 +1
        [1, 0, 0, 0]    # P7: C00 +1
    ]
    
    col = 0
    for b in range(5):
        for r in range(7):
            # U: same for all blocks
            for i in range(4):
                U[i][col] = strassen_A[r][i]
            
            # V: specific to block b
            # B inputs for block b:
            # e = B[0, 2b] -> index 2b
            # f = B[0, 2b+1] -> index 2b+1
            # g = B[1, 2b] -> index 10 + 2b
            # h = B[1, 2b+1] -> index 10 + 2b + 1
            
            V[2*b][col] = strassen_B[r][0]
            V[2*b+1][col] = strassen_B[r][1]
            V[10+2*b][col] = strassen_B[r][2]
            V[10+2*b+1][col] = strassen_B[r][3]
            
            # W: specific to block b
            # C outputs for block b:
            # C00: i=0, k_global=2b -> index (2b)*2 + 0 = 4b
            # C01: i=0, k_global=2b+1 -> index (2b+1)*2 + 0 = 4b+2
            # C10: i=1, k_global=2b -> index (2b)*2 + 1 = 4b+1
            # C11: i=1, k_global=2b+1 -> index (2b+1)*2 + 1 = 4b+3
            
            W[4*b][col] = w_contrib[r][0]
            W[4*b+2][col] = w_contrib[r][1]
            W[4*b+1][col] = w_contrib[r][2]
            W[4*b+3][col] = w_contrib[r][3]
            
            col += 1
            
    return U, V, W


def _try_rank_decomposition(T, n, m, p, target_rank, time_budget=8, trial=0):
    """
    Try to find a rank-target_rank decomposition of tensor T.
    Uses random initialization + gradient descent with adaptive learning rate.
    """
    dim_u = n * m  # 4
    dim_v = m * p  # 20
    dim_w = p * n  # 20
    
    start_time = time.time()
    
    # Different random seeds for different trials
    np.random.seed(12345 + trial * 100)
    
    # Random initialization
    U = np.random.randn(dim_u, target_rank) * 0.5
    V = np.random.randn(dim_v, target_rank) * 0.5
    W = np.random.randn(dim_w, target_rank) * 0.5
    
    # Gradient descent with adaptive learning rate
    lr = 0.02
    min_lr = 1e-7
    patience = 30
    no_improve = 0
    best_loss = float('inf')
    
    U_best = U.copy()
    V_best = V.copy()
    W_best = W.copy()
    
    for step in range(1500):
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
        
        # Early stopping if no improvement
        if no_improve > patience and lr < min_lr:
            break
        
        # Compute gradients
        grad_U = 2 * np.einsum('ijk,jr,kr->ir', diff, V, W)
        grad_V = 2 * np.einsum('ijk,ir,kr->jr', diff, U, W)
        grad_W = 2 * np.einsum('ijk,ir,jr->kr', diff, U, V)
        
        # Update with momentum-like behavior
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Adaptive learning rate
        if no_improve > 15:
            lr *= 0.5
            if lr < min_lr:
                break
        
        # Check convergence
        if loss < 1e-12:
            break
    
    # Final check using best found
    T_final = np.einsum('ir,jr,kr->ijk', U_best, V_best, W_best)
    if np.max(np.abs(T - T_final)) < 1e-8:
        # Convert to list format
        U_list = U_best.tolist()
        V_list = V_best.tolist()
        W_list = W_best.tolist()
        return U_list, V_list, W_list
    
    return None


def _verify_tensor_correctness(T, n, m, p):
    """
    Verify that the tensor T correctly represents matrix multiplication.
    T[i,j,k] where:
    i = i_A * m + j_A (i_A in 0..n-1, j_A in 0..m-1)
    j = j_B * p + k_B (j_B in 0..m-1, k_B in 0..p-1)
    k = k_C * n + i_C (k_C in 0..p-1, i_C in 0..n-1)
    
    T[i,j,k] should equal sum over mid of A[i_A, mid] * B[mid, k_C]
    where the sum is non-zero only if j_A == j_B (the middle index matches)
    """
    # For the tensor to be correct, T[i,j,k] should be:
    # 1 if i_A == i_C and j_A == j_B and j_A == k_C... wait, let me think again.
    
    # Actually, for matrix multiplication C = A * B:
    # C[i_C, k_C] = sum_{j} A[i_C, j] * B[j, k_C]
    
    # In the tensor T[i,j,k]:
    # i = i_A * m + j_A
    # j = j_B * p + k_B
    # k = k_C * n + i_C
    
    # T[i,j,k] should contribute to C[i_C, k_C] when j_A == j_B
    # And the contribution is A[i_A, j_A] * B[j_B, k_B]
    # But we want C[i_C, k_C] = sum_j A[i_C, j] * B[j, k_C]
    # So T[i,j,k] should be 1 if i_A == i_C and j_A == j_B and k_B == k_C, else 0
    
    # Let's verify this
    np.random.seed(42)
    for _ in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        
        A_flat = A.flatten()  # index: i_A * m + j_A
        B_flat = B.flatten()  # index: j_B * p + k_B
        
        # Compute C_flat using the tensor
        C_flat = np.zeros(p * n)
        for i in range(n * m):
            i_A = i // m
            j_A = i % m
            for j in range(m * p):
                j_B = j // p
                k_B = j % p
                if j_A != j_B:
                    continue
                for k in range(p * n):
                    k_C = k // n
                    i_C = k % n
                    if i_A == i_C and k_B == k_C:
                        C_flat[k] += A_flat[i] * B_flat[j] * T[i, j, k]
        
        # Reshape C_flat: index k_C * n + i_C
        C_check = C_flat.reshape(p, n).T
        
        if not np.allclose(C, C_check, atol=1e-8):
            return False
    
    return True


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly represents matrix multiplication."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the tensor from the decomposition
    T = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
    
    return _verify_tensor_correctness(T, n, m, p)
# EVOLVE-BLOCK-END
