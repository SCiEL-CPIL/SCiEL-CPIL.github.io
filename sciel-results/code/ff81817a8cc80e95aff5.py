"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the tensor using vectorized ops."""
    U = np.array(U, dtype=np.int64)
    V = np.array(V, dtype=np.int64)
    W = np.array(W, dtype=np.int64)
    
    # Generate all valid indices
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    a_idx = i_idx * m + j_idx
    b_idx = j_idx * p + k_idx
    c_idx = k_idx * n + i_idx
    
    # Calculate values for valid terms: sum_r U[a,r] * V[b,r] * W[c,r]
    # We need to compute this for each valid (a,b,c)
    # This can be done by iterating over the valid indices or using advanced indexing
    # Since the number of valid terms is n*m*p = 64, a loop is fast enough, 
    # but let's try to vectorize the dot product part if possible.
    # Actually, just looping over 64 items is extremely fast in Python if we pre-fetch rows.
    
    for a, b, c in zip(a_idx, b_idx, c_idx):
        val = np.dot(U[a], V[b] * W[c])
        if val != 1:
            return False
            
    # Check spurious terms
    # Total terms in (a,b,c) space is (n*m) * (m*p) * (p*n) = 4 * 32 * 32 = 4096
    # Valid terms are 64.
    # We can create a 3D array of zeros and mark valid ones, then check the rest.
    # Or simply iterate over all and skip valid ones.
    
    valid_set = set(zip(a_idx, b_idx, c_idx))
    
    # To speed up spurious check, we can construct the full tensor of values 
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # This is a tensor contraction.
    # T = np.einsum('ar,br,cr->abc', U, V, W)
    # Then check T[valid] == 1 and T[~valid] == 0.
    
    T = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    
    # Check valid
    for a, b, c in zip(a_idx, b_idx, c_idx):
        if T[a, b, c] != 1:
            return False
            
    # Check spurious
    # Create a mask of valid indices
    mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    mask[a_idx, b_idx, c_idx] = True
    
    # Check if any spurious term is non-zero
    if np.any(T[~mask] != 0):
        return False
        
    return True

def _construct_strassen_2x16():
    n, m, p = 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        # B indices for this block
        b00 = 2*t
        b01 = 2*t + 1
        b10 = 16 + 2*t
        b11 = 16 + 2*t + 1
        
        # C indices for this block
        c00 = 4*t
        c01 = 4*t + 1
        c10 = 4*t + 2
        c11 = 4*t + 3
        
        # Strassen intermediates
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 (c00) = M1 + M4 - M5 + M7
        # C01 (c01) = M2 + M4
        # C10 (c10) = M3 + M5
        # C11 (c11) = M1 - M2 + M3 + M6
        
        # M1
        U[0][col] += 1; U[3][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        
        # M2
        U[2][col] += 1; U[3][col] += 1
        V[b00][col] += 1
        W[c01][col] += 1; W[c11][col] -= 1
        col += 1
        
        # M3
        U[0][col] += 1
        V[b01][col] += 1; V[b11][col] -= 1
        W[c10][col] += 1; W[c11][col] += 1
        col += 1
        
        # M4
        U[3][col] += 1
        V[b10][col] += 1; V[b00][col] -= 1
        W[c00][col] += 1; W[c01][col] += 1
        col += 1
        
        # M5
        U[0][col] += 1; U[1][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1; W[c10][col] += 1
        col += 1
        
        # M6
        U[2][col] += 1; U[0][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7
        U[1][col] += 1; U[3][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00][col] += 1
        col += 1
        
    return U, V, W

def _try_rank56_with_shared_structure():
    """
    Attempt to find a rank-56 decomposition by exploiting structure.
    The key idea: if we can share some Strassen factors across blocks,
    we might reduce the rank below 56.
    
    For 2x2x16, the tensor has 4*2*16 = 128 valid entries.
    A rank-r decomposition gives r*(4+32+32) = 68r free parameters
    (before the bilinear constraint).
    
    The current approach gives rank 56. Let's see if we can do better
    by finding a more clever factorization.
    
    One approach: group the 16 columns into 4 groups of 4, and use
    a 2x2x4 tensor decomposition for each group.
    
    The 2x2x4 tensor has 4*2*4 = 32 valid entries.
    The known rank of 2x2x4 is... let's think.
    Actually, 2x2x4 is equivalent to multiplying a 2x2 matrix by a 2x4 matrix.
    This can be viewed as two independent 2x2x2 multiplications (rank 7 each),
    giving rank 14. But can we do better?
    
    Actually, for 2x2x4, we can use the fact that the output has 8 entries.
    Let's try to find a rank-13 or rank-12 decomposition.
    
    Alternatively, we can try to find a direct rank-r decomposition
    for the full 2x2x16 tensor using a more global approach.
    """
    n, m, p = 2, 2, 16
    # Try to find a lower rank decomposition
    # The theoretical lower bound is at least 128/68 ≈ 1.88, so rank >= 2
    # But the actual minimum rank is likely much higher.
    
    # For now, return None to indicate we couldn't find a better decomposition
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # First try the known rank-56 Strassen-based decomposition
    U, V, W = _construct_strassen_2x16()
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback (should not happen)
    rank = n * m * p
    U_fb = [[0] * rank for _ in range(n * m)]
    V_fb = [[0] * rank for _ in range(m * p)]
    W_fb = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-END
