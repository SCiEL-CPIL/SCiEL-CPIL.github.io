"""Matrix multiplication 2x2x15 tensor decomposition using sliding-window Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

def _build_strassen_2x2x2():
    """
    Build Strassen's rank-7 decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) following the trace convention W[k*n+i, r].
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Term 1: M1 = (A00 + A11) * (B00 + B11)
    U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
    W[0*2+0, 0] = 1; W[1*2+1, 0] = 1
    
    # Term 2: M2 = (A10 + A11) * B00
    U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[0*2+1, 1] = 1; W[1*2+1, 1] = -1
    
    # Term 3: M3 = A00 * (B01 - B11)
    U[0*2+0, 2] = 1
    V[0*2+1, 2] = 1; V[1*2+1, 2] = -1
    W[1*2+0, 2] = 1; W[1*2+1, 2] = 1
    
    # Term 4: M4 = A11 * (B10 - B00)
    U[1*2+1, 3] = 1
    V[1*2+0, 3] = 1; V[0*2+0, 3] = -1
    W[0*2+0, 3] = 1; W[0*2+1, 3] = 1
    
    # Term 5: M5 = (A00 + A01) * B11
    U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[0*2+0, 4] = -1; W[1*2+0, 4] = 1
    
    # Term 6: M6 = (A10 - A00) * (B00 + B01)
    U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1; V[0*2+1, 5] = 1
    W[1*2+1, 5] = 1
    
    # Term 7: M7 = (A01 - A11) * (B10 + B11)
    U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1; V[1*2+1, 6] = 1
    W[0*2+0, 6] = 1
    
    return U, V, W

def _build_schoolbook_2x2x1():
    """
    Build schoolbook rank-4 decomposition for 2x2 x 2x1 matrix multiplication.
    Returns U (4x4), V (2x4), W (2x4).
    """
    n, m, p = 2, 2, 1
    rank = 4
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
            V[j * p, col] = 1
            W[0 * n + i, col] = 1
            col += 1
            
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    rank = U.shape[1]
    # Check shapes
    assert U.shape == (n*m, rank), f"U shape {U.shape} != {(n*m, rank)}"
    assert V.shape == (m*p, rank), f"V shape {V.shape} != {(m*p, rank)}"
    assert W.shape == (p*n, rank), f"W shape {(p*n, rank)} != {W.shape}"
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                expected = 1.0
                if abs(val - expected) > 1e-10:
                    return False, (i, j, k, val, expected)
    return True, None

def _build_sliding_window(n, m, p):
    """
    Build a sliding-window decomposition for 2x2xp using overlapping Strassen blocks.
    For p=15, we use 7 Strassen blocks covering columns (0,1), (1,2), (2,3), ..., (13,14)
    and one schoolbook block for column 14 (or handle the edge differently).
    
    Actually, let's think more carefully:
    - Block b covers columns b and b+1 for b = 0, 1, ..., p-2
    - That's p-1 = 14 blocks, each rank 7, total rank 98. That's worse than 53.
    
    Alternative: Use non-overlapping blocks but with a different strategy.
    
    Let me reconsider. The key insight for sliding window is to share the U matrix
    across adjacent blocks. But in the tensor decomposition, each block is independent
    in the rank dimension.
    
    A better approach: Try to find a lower-rank decomposition for 2x2x3 or 2x2x4
    and use those as building blocks.
    
    For now, let's try a direct approach: use 5 blocks of 2x2x3.
    We need a rank-10 decomposition for 2x2x3.
    """
    n, m, p = n, m, p
    
    # Try to construct 2x2x3 with rank 10
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for k=0,1,2
    # This is equivalent to 3 separate 2x2x1 multiplications, each rank 4, total rank 12.
    # But we can share some terms.
    
    # Let's try a constructive approach:
    # Use Strassen for columns 0,1 (rank 7) and add 3 terms for column 2.
    # But we need to be careful about cross-terms.
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    
    # For 2x2x3, we have:
    # Columns 0,1: handled by Strassen (7 terms)
    # Column 2: need 4 more terms (schoolbook) but we can potentially share 1 term
    # Total: 7 + 4 = 11, or 7 + 3 = 10 if we can share 1 term.
    
    # Let's try rank 10:
    rank_223 = 10
    U_223 = np.zeros((n * m, rank_223))
    V_223 = np.zeros((m * 3, rank_223))
    W_223 = np.zeros((3 * n, rank_223))
    
    # First 7 terms: Strassen for columns 0,1
    U_223[:, 0:7] = U_str
    for j in range(m):
        for k_local in range(2):
            V_223[j * 3 + k_local, 0:7] = V_str[j * 2 + k_local, :]
    for k_local in range(2):
        for i in range(n):
            W_223[k_local * n + i, 0:7] = W_str[k_local * n + i, :]
    
    # Terms 7,8,9 for column 2:
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # Term 7: A[0,0]*B[0,2] -> C[0,2]
    U_223[0*2+0, 7] = 1
    V_223[0*3+2, 7] = 1
    W_223[2*2+0, 7] = 1
    
    # Term 8: A[1,0]*B[0,2] -> C[1,2]
    U_223[1*2+0, 8] = 1
    V_223[0*3+2, 8] = 1
    W_223[2*2+1, 8] = 1
    
    # Term 9: (A[0,1] + A[1,1])*B[1,2] -> C[0,2] += A[0,1]*B[1,2], C[1,2] += A[1,1]*B[1,2]
    U_223[0*2+1, 9] = 1
    U_223[1*2+1, 9] = 1
    V_223[1*3+2, 9] = 1
    W_223[2*2+0, 9] = 1
    W_223[2*2+1, 9] = 1
    
    # Verify 2x2x3
    valid_223, _ = _verify(U_223, V_223, W_223, n, m, 3)
    
    if not valid_223:
        # Fall back to rank 11 for 2x2x3
        rank_223 = 11
        U_223 = np.zeros((n * m, rank_223))
        V_223 = np.zeros((m * 3, rank_223))
        W_223 = np.zeros((3 * n, rank_223))
        
        U_223[:, 0:7] = U_str
        for j in range(m):
            for k_local in range(2):
                V_223[j * 3 + k_local, 0:7] = V_str[j * 2 + k_local, :]
        for k_local in range(2):
            for i in range(n):
                W_223[k_local * n + i, 0:7] = W_str[k_local * n + i, :]
        
        # Terms 7,8,9,10 for column 2 (schoolbook)
        U_223[0*2+0, 7] = 1
        V_223[0*3+2, 7] = 1
        W_223[2*2+0, 7] = 1
        
        U_223[1*2+0, 8] = 1
        V_223[0*3+2, 8] = 1
        W_223[2*2+1, 8] = 1
        
        U_223[0*2+1, 9] = 1
        V_223[1*3+2, 9] = 1
        W_223[2*2+0, 9] = 1
        
        U_223[1*2+1, 10] = 1
        V_223[1*3+2, 10] = 1
        W_223[2*2+1, 10] = 1
        
        valid_223, _ = _verify(U_223, V_223, W_223, n, m, 3)
        if not valid_223:
            return None
    
    # Now use 5 blocks of 2x2x3 for p=15
    total_rank = rank_223 * 5
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for b in range(5):
        start_k = b * 3
        U[:, col_offset:col_offset+rank_223] = U_223
        for j in range(m):
            for k_local in range(3):
                k_global = start_k + k_local
                V[j * p + k_global, col_offset:col_offset+rank_223] = V_223[j * 3 + k_local, :]
        for k_local in range(3):
            k_global = start_k + k_local
            for i in range(n):
                W[k_global * n + i, col_offset:col_offset+rank_223] = W_223[k_local * n + i, :]
        col_offset += rank_223
    
    valid, _ = _verify(U, V, W, n, m, p)
    if valid:
        return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Try sliding window with 2x2x3 blocks
    result = _build_sliding_window(n, m, p)
    
    if result is not None:
        U, V, W = result
        valid, _ = _verify(U, V, W, n, m, p)
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to Strassen-based approach with rank 53
    block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
    num_blocks = len(block_sizes)
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook_2x2x1()
    
    total_rank = 7 * 7 + 4  # 53
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for b in range(num_blocks):
        block_size = block_sizes[b]
        start_k = sum(block_sizes[:b])
        
        if block_size == 2:
            U[:, col_offset:col_offset+7] = U_str
            for j in range(m):
                for k_local in range(2):
                    k_global = start_k + k_local
                    V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
            for k_local in range(2):
                k_global = start_k + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * n + i, :]
            col_offset += 7
        else:
            U[:, col_offset:col_offset+4] = U_sb
            for j in range(m):
                V[j * p + start_k, col_offset:col_offset+4] = V_sb[j * 1, :]
            for i in range(n):
                W[start_k * n + i, col_offset:col_offset+4] = W_sb[0 * n + i, :]
            col_offset += 4
            
    valid, _ = _verify(U, V, W, n, m, p)
    if not valid:
        # Fall back to pure schoolbook
        total_rank_fb = n * m * p
        U_fb = np.zeros((n * m, total_rank_fb))
        V_fb = np.zeros((m * p, total_rank_fb))
        W_fb = np.zeros((p * n, total_rank_fb))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, idx] = 1
                    V_fb[j*p+k, idx] = 1
                    W_fb[k*n+i, idx] = 1
                    idx += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
