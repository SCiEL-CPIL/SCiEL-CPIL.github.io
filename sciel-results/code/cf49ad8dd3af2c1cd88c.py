"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_222():
    """Return Strassen decomposition for 2x2x2 multiplication (rank 7)."""
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    return U, V, W

def _schoolbook_221():
    """Return schoolbook decomposition for 2x2x1 multiplication (rank 4)."""
    U = np.eye(4)
    V = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1]], dtype=float)
    W = np.array([[1, 1, 0, 0],
                  [0, 0, 1, 1]], dtype=float)
    return U, V, W

def _build_rank53():
    """Build the guaranteed rank-53 decomposition (7 Strassen blocks + 1 schoolbook)."""
    n, m, p = DIMENSIONS
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_strassen, V_strassen, W_strassen = _strassen_222()
    U_school, V_school, W_school = _schoolbook_221()
    
    col_offset = 0
    for block in range(7):
        k0 = 2 * block
        k1 = 2 * block + 1
        for r in range(7):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i * m + j][c] = int(U_strassen[i * 2 + j, r])
            for j in range(2):
                V[j * p + k0][c] = int(V_strassen[j * 2, r])
                V[j * p + k1][c] = int(V_strassen[j * 2 + 1, r])
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W[kk * n + i][c] = int(W_strassen[k * 2 + i, r])
        col_offset += 7
    
    k_last = 14
    for r in range(4):
        c = col_offset + r
        for i in range(2):
            for j in range(2):
                U[i * m + j][c] = int(U_school[i * 2 + j, r])
        for j in range(2):
            V[j * p + k_last][c] = int(V_school[j, r])
        for i in range(2):
            W[k_last * n + i][c] = int(W_school[i, r])
    
    return {"U": U, "V": V, "W": W}

def _verify_decomp(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank_block(n, m, p_block, target_rank, deadline, max_seeds=5):
    """
    Try to find a rank-target_rank decomposition for M_{n,m,p_block}.
    Uses a structured approach: try small integer/half-integer patterns.
    Returns (U_block, V_block, W_block) or None.
    """
    # For small p_block, we can enumerate or use known results
    # M_{2,2,2} has rank 7 (Strassen)
    # M_{2,2,3} might have rank 10 or 11
    # M_{2,2,4} might have rank 14 or 15
    
    # For now, we'll use a grid search for small cases
    if p_block == 2:
        if target_rank >= 7:
            U, V, W = _strassen_222()
            if target_rank > 7:
                # Pad with zeros
                U = np.hstack([U, np.zeros((4, target_rank-7))])
                V = np.hstack([V, np.zeros((4, target_rank-7))])
                W = np.hstack([W, np.zeros((4, target_rank-7))])
            return U, V, W
        return None
    
    if p_block == 3:
        # Try to find rank 10 for M_{2,2,3}
        # This is known to be achievable
        # Use a heuristic: 3x Strassen-like but optimized
        # For simplicity, use 3x schoolbook (rank 12) if target_rank >= 12
        if target_rank >= 12:
            U = np.zeros((4, target_rank))
            V = np.zeros((6, target_rank))
            W = np.zeros((6, target_rank))
            U_sb, V_sb, W_sb = _schoolbook_221()
            for block in range(3):
                k0 = block
                for r in range(4):
                    c = block * 4 + r
                    for i in range(2):
                        for j in range(2):
                            U[i*2+j, c] = U_sb[i*2+j, r]
                    for j in range(2):
                        V[j*3+k0, c] = V_sb[j, r]
                    for i in range(2):
                        W[k0*2+i, c] = W_sb[i, r]
            if target_rank > 12:
                U = np.hstack([U, np.zeros((4, target_rank-12))])
                V = np.hstack([V, np.zeros((6, target_rank-12))])
                W = np.hstack([W, np.zeros((6, target_rank-12))])
            return U, V, W
        return None
    
    if p_block == 4:
        # Try rank 14 for M_{2,2,4}
        # Use 2x Strassen (rank 14)
        if target_rank >= 14:
            U_str, V_str, W_str = _strassen_222()
            U = np.zeros((4, target_rank))
            V = np.zeros((8, target_rank))
            W = np.zeros((8, target_rank))
            for block in range(2):
                k0 = 2 * block
                for r in range(7):
                    c = block * 7 + r
                    for i in range(2):
                        for j in range(2):
                            U[i*2+j, c] = U_str[i*2+j, r]
                    for j in range(2):
                        V[j*4+k0, c] = V_str[j*2, r]
                        V[j*4+k0+1, c] = V_str[j*2+1, r]
                    for k in range(2):
                        for i in range(2):
                            W[(k0+k)*2+i, c] = W_str[k*2+i, r]
            if target_rank > 14:
                U = np.hstack([U, np.zeros((4, target_rank-14))])
                V = np.hstack([V, np.zeros((8, target_rank-14))])
                W = np.hstack([W, np.zeros((8, target_rank-14))])
            return U, V, W
        return None
    
    # For larger p_block, use combinations
    # p_block = 5: 2x2 + 1x1 = 7+4 = 11
    # p_block = 6: 3x2 = 21 or 2x2+1x2 = 7+7=14 (if 1x2 exists with rank 7?)
    # Actually M_{2,2,1} has rank 4, so 2x2+1x1+1x1+1x1+1x1 = 7*2 + 4*3 = 26 for p=6
    # Better: 3x M_{2,2,2} = 21 for p=6
    # Or 2x M_{2,2,2} + 1x M_{2,2,2} = same
    
    # General approach: decompose p_block into sum of 2s and 1s
    # Use as many 2-blocks as possible
    num_2blocks = p_block // 2
    num_1blocks = p_block % 2
    base_rank = num_2blocks * 7 + num_1blocks * 4
    
    if target_rank >= base_rank:
        U_str, V_str, W_str = _strassen_222()
        U_sb, V_sb, W_sb = _schoolbook_221()
        
        U = np.zeros((4, target_rank))
        V = np.zeros((2*p_block, target_rank))
        W = np.zeros((2*p_block, target_rank))
        
        col_offset = 0
        k_offset = 0
        for _ in range(num_2blocks):
            for r in range(7):
                c = col_offset + r
                for i in range(2):
                    for j in range(2):
                        U[i*2+j, c] = U_str[i*2+j, r]
                for j in range(2):
                    V[j*p_block+k_offset, c] = V_str[j*2, r]
                    V[j*p_block+k_offset+1, c] = V_str[j*2+1, r]
                for k in range(2):
                    for i in range(2):
                        W[(k_offset+k)*2+i, c] = W_str[k*2+i, r]
            col_offset += 7
            k_offset += 2
        
        for _ in range(num_1blocks):
            for r in range(4):
                c = col_offset + r
                for i in range(2):
                    for j in range(2):
                        U[i*2+j, c] = U_sb[i*2+j, r]
                for j in range(2):
                    V[j*p_block+k_offset, c] = V_sb[j, r]
                for i in range(2):
                    W[k_offset*2+i, c] = W_sb[i, r]
            col_offset += 4
            k_offset += 1
        
        if target_rank > col_offset:
            U = np.hstack([U, np.zeros((4, target_rank-col_offset))])
            V = np.hstack([V, np.zeros((2*p_block, target_rank-col_offset))])
            W = np.hstack([W, np.zeros((2*p_block, target_rank-col_offset))])
        
        return U, V, W
    
    return None

def _build_rank50():
    """
    Try to build a rank-50 decomposition.
    Strategy: 15 = 7*2 + 1, so 7 blocks of 2 + 1 block of 1
    Rank = 7*7 + 4 = 53 (this is the same as before)
    
    Alternative: Try to find better decompositions for some blocks.
    For p=15, we can try:
    - 5 blocks of p=3: if M_{2,2,3} has rank 10, total = 50
    - 3 blocks of p=5: if M_{2,2,5} has rank 16, total = 48
    - 1 block of p=15 directly
    
    Let's try the p=3 approach with rank 10 each.
    """
    n, m, p = DIMENSIONS
    # Try 5 blocks of p=3 with rank 10 each
    blocks = []
    for block in range(5):
        result = _try_rank_block(n, m, 3, 10, time.time() + 10)
        if result is None:
            return None
        blocks.append(result)
    
    total_rank = 50
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for block in range(5):
        U_b, V_b, W_b = blocks[block]
        k_start = 3 * block
        col_offset = 10 * block
        for r in range(10):
            c = col_offset + r
            for i in range(n):
                for j in range(m):
                    U[i * m + j][c] = float(U_b[i * m + j, r])
            for j in range(m):
                for k in range(3):
                    V[j * p + k_start + k][c] = float(V_b[j * 3 + k, r])
            for k in range(3):
                for i in range(n):
                    W[(k_start + k) * n + i][c] = float(W_b[k * 2 + i, r])
    
    # Verify
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    if _verify_decomp(U_arr, V_arr, W_arr, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None

def _build_rank48():
    """
    Try to build a rank-48 decomposition.
    Strategy: 15 = 3*5, so 3 blocks of p=5
    If M_{2,2,5} has rank 16, total = 48
    """
    n, m, p = DIMENSIONS
    # Try 3 blocks of p=5 with rank 16 each
    blocks = []
    for block in range(3):
        result = _try_rank_block(n, m, 5, 16, time.time() + 10)
        if result is None:
            return None
        blocks.append(result)
    
    total_rank = 48
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for block in range(3):
        U_b, V_b, W_b = blocks[block]
        k_start = 5 * block
        col_offset = 16 * block
        for r in range(16):
            c = col_offset + r
            for i in range(n):
                for j in range(m):
                    U[i * m + j][c] = float(U_b[i * m + j, r])
            for j in range(m):
                for k in range(5):
                    V[j * p + k_start + k][c] = float(V_b[j * 5 + k, r])
            for k in range(5):
                for i in range(n):
                    W[(k_start + k) * n + i][c] = float(W_b[k * 2 + i, r])
    
    # Verify
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    if _verify_decomp(U_arr, V_arr, W_arr, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Try rank-48 first (3 blocks of p=5 with rank 16)
    if time.time() < deadline:
        result = _build_rank48()
        if result is not None:
            return result
    
    # Try rank-50 (5 blocks of p=3 with rank 10)
    if time.time() < deadline:
        result = _build_rank50()
        if result is not None:
            return result
    
    # Fallback to rank-53
    return _build_rank53()
# EVOLVE-BLOCK-END
