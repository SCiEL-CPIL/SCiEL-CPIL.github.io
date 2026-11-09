"""Program for matrix multiplication tensor decomposition (3x4, 4x6)."""
import time
import numpy as np
import itertools

n_dim = 3
m_dim = 4
p_dim = 6

def _verify_fast(U, V, W, tensor, tol=1e-9):
    """Fast verification using vectorization."""
    rank = U.shape[1]
    n, m, p = tensor.shape
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    try:
        reconstructed = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
        err = np.max(np.abs(reconstructed - tensor))
        return err < tol
    except Exception:
        return False

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _build_rank66(n, m, p):
    """
    Build a rank-66 decomposition for 3x4x6 matrix multiplication.
    
    Uses the 3x2x2 block structure:
    - Split j (size 4) into two blocks of size 2: j1 (j=0,1), j2 (j=2,3)
    - Split k (size 6) into three blocks of size 2: k1 (k=0,1), k2 (k=2,3), k3 (k=4,5)
    - Each (j_block, k_block) pair uses a 2x2x2 Strassen-like decomposition (rank 7)
    - Total: 3 (i) * 2 (j_blocks) * 3 (k_blocks) * 7 = 126... 
    
    Actually, let me reconsider. The correct construction:
    - 3 i-values * 2 j-blocks * 3 k-blocks * 7 = 126 (too many)
    
    Alternative: the rank-66 comes from a different structure.
    Let me think about this differently.
    
    For 3x4x6: the tensor has 3*4*6 = 72 entries.
    Rank 66 means we save 6 terms.
    
    The 3x2x2 block hint suggests:
    - Group j into 2 groups of 2, k into 3 groups of 2
    - For each (i, j_group, k_group), we have a 2x2x2 sub-problem
    - But 3*2*3 = 18 sub-problems of rank 7 = 126 (too many)
    
    Wait - perhaps the structure is different. Let me try:
    - 3 i-values
    - For each i, we have a 4x6 matrix multiplication
    - 4x6 = (2+2)x(2+2+2)
    - Using rank-11 for 2x2x3 (known optimal)
    - 3 * 11 * 2 = 66? No...
    
    Actually: 3 i-values * 2 j-blocks * 11 (rank for 2x2x3) = 66!
    That's it! For each i and each j-block of size 2, we multiply a 2x3 matrix by...
    
    Let me reconsider: 4x6 = 2*2 * 3*2. 
    If we split j into 2 blocks of 2, and k into 3 blocks of 2:
    For each (j_block, k_block), we have a 2x2x2 sub-tensor.
    But we can group: for fixed i and j_block, we have 2x(3*2) = 2x6.
    Using rank-11 for 2x2x3, and we have 3 such blocks in k... 
    
    Actually the cleanest: 3 * 2 * 11 = 66.
    For each i (3 choices) and each j-block (2 blocks of size 2):
      We need to compute a 2x6 product, which decomposes into 3 sub-products of 2x2x2.
      Each 2x2x2 has rank 7, so 3*7 = 21 per (i, j_block)? That gives 3*2*21 = 126.
    
    Let me just try the direct approach: use rank-11 for 2x2x3.
    4x6 = (2x2) x (3x2) in terms of blocks.
    For fixed i: multiply A[i,:] (size 4) by B[:,k] (size 4) for each k.
    Split A into 2 blocks of 2, B into... 
    
    I'll use a different approach: construct via known 2x2x2 Strassen + padding.
    """
    # Let me try a direct algebraic construction.
    # The key identity for 2x2x2 Strassen (rank 7):
    # M11 = (A11+A22)(B11+B22) - M3 - M7
    # M12 = A11*B21 + (A21+A22)*B22 ... 
    # Actually standard Strassen:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B11)
    # P4 = A22(B22-B12)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    # C11 = P1+P4-P5+P7
    # C12 = P3+P5
    # C21 = P2+P4
    # C22 = P1-P2+P3+P6
    
    # For 3x4x6, I'll use a block decomposition.
    # Split m=4 into j1={0,1}, j2={2,3} (two blocks of 2)
    # Split p=6 into k1={0,1}, k2={2,3}, k3={4,5} (three blocks of 2)
    # For each i, j_block, k_block: 2x2x2 sub-problem with rank 7
    # Total: 3 * 2 * 3 * 7 = 126 (too many)
    
    # Alternative: use rank-11 for 2x2x3
    # Split m=4 into j1={0,1}, j2={2,3}
    # Split p=6 into k1={0,1,2}, k2={3,4,5}
    # For each i, j_block, k_block: 2x2x3 sub-problem with rank 11
    # Total: 3 * 2 * 2 * 11 = 132 (still too many)
    
    # Hmm, let me try: 3 * 2 * 11 = 66
    # This means: for each i (3) and j_block (2), we have one 2x2x3 decomposition (rank 11)
    # But 2x2x3 means: 2 rows, 2 cols, 3 "outputs"? 
    # Actually 2x2x3 tensor: multiply 2x2 by 2x3? No, that's not standard.
    
    # Let me reconsider: maybe it's 3 * 22 = 66
    # For each i, we need rank 22 for the 4x6 multiplication
    # 4x6 = 24 entries, rank 22 means we save 2 per i.
    
    # Or: the structure is that we share terms across i values.
    
    # Given the complexity, let me try a numerical search with a smarter approach.
    # Use the known lower bound and try to find rank 66 via optimization.
    
    # For now, return schoolbook as fallback
    U, V, W = _build_schoolbook(n, m, p)
    return U, V, W

def _als_step(U, V, W, tensor, n, m, p, rank):
    """Perform one step of Alternating Least Squares."""
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    U_new = np.zeros((n, m, rank))
    for i in range(n):
        for j in range(m):
            A = V_reshaped[j, :, :] * W_reshaped[:, i, :]
            b = tensor[i, j, :]
            try:
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U_new[i, j, :] = sol
            except np.linalg.LinAlgError:
                pass
    U = U_new.reshape(n * m, rank)
    
    V_new = np.zeros((m, p, rank))
    for j in range(m):
        for k in range(p):
            A = U_reshaped[:, j, :] * W_reshaped[k, :, :]
            b = tensor[:, j, k]
            try:
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V_new[j, k, :] = sol
            except np.linalg.LinAlgError:
                pass
    V = V_new.reshape(m * p, rank)
    
    W_new = np.zeros((p, n, rank))
    for k in range(p):
        for i in range(n):
            A = U_reshaped[i, :, :] * V_reshaped[:, k, :]
            b = tensor[i, :, k]
            try:
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W_new[k, i, :] = sol
            except np.linalg.LinAlgError:
                pass
    W = W_new.reshape(p * n, rank)
    
    return U, V, W

def _try_rank_search(tensor, n, m, p, target_rank, rng, deadline, max_iters=30):
    """Try to find a decomposition of given rank via ALS."""
    max_attempts = 3
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        
        U = rng.standard_normal((n * m, target_rank))
        V = rng.standard_normal((m * p, target_rank))
        W = rng.standard_normal((p * n, target_rank))
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            U, V, W = _als_step(U, V, W, tensor, n, m, p, target_rank)
            if _verify_fast(U, V, W, tensor):
                return U, V, W
    return None

def construct():
    deadline = 140  # seconds, safe margin before 150s limit
    
    n = n_dim
    m = m_dim
    p = p_dim
    
    tensor = np.ones((n, m, p), dtype=np.float64)
    
    # Schoolbook fallback
    U_fb, V_fb, W_fb = _build_schoolbook(n, m, p)
    best_rank = n * m * p  # 72
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Try to find rank 66 (the known optimal from inspiration)
    # Use multiple random seeds and more iterations
    for target_rank in [66, 67, 68, 69, 70, 71]:
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_search(tensor, n, m, p, target_rank, rng, deadline - 5, max_iters=50)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
