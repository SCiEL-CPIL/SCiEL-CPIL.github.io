"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p, offset=0):
    """Return (U, V, W) lists for schoolbook decomposition of n×m times m×p.
    U has shape (n*m, n*m*p), V has shape (m*p, n*m*p), W has shape (p*n, n*m*p).
    """
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, r

def _strassen_4x4_4x8():
    """Build Strassen decomposition for 4x4 * 4x8.
    
    A is 4x4, viewed as 2x2 blocks of 2x2.
    B is 4x8, viewed as 2x4 blocks (2 rows, 4 cols each).
    C is 4x8, viewed as 2x4 blocks.
    
    Block indices:
    A = [[A11, A12],
         [A21, A22]]  where each Ai is 2x2
    B = [[B11, B12],
         [B21, B22]]  where each Bj is 2x4
    C = [[C11, C12],
         [C21, C22]]  where each Ck is 2x4
    
    Standard Strassen:
    M1 = (A11+A22) * (B11+B22)
    M2 = (A21+A22) * B11
    M3 = A11 * (B12-B22)
    M4 = A22 * (B21-B11)
    M5 = (A11+A12) * B22
    M6 = (A21-A11) * (B11+B12)
    M7 = (A12-A22) * (B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M3 + M2 - M6
    """
    n_sub, m_sub, p_sub = 2, 2, 4
    sub_rank = n_sub * m_sub * p_sub  # 16
    total_rank = 7 * sub_rank  # 112
    
    n, m, p = 4, 4, 8
    U = [[0] * total_rank for _ in range(n * m)]   # (16, 112)
    V = [[0] * total_rank for _ in range(m * p)]   # (32, 112)
    W = [[0] * total_rank for _ in range(p * n)]   # (32, 112)
    
    # alpha[t]: block-row, block-col -> coeff for A
    # beta[t]: block-row, block-col -> coeff for B
    # gamma[t]: block-row, block-col -> coeff for C
    # Block (br, bc) of A covers rows br*2..br*2+1, cols bc*2..bc*2+1
    # Block (br, bc) of B covers rows br*2..br*2+1, cols bc*4..bc*4+3
    # Block (cr, cc) of C covers rows cr*2..cr*2+1, cols cc*4..cc*4+3
    
    alpha = [
        {(0,0):1, (1,1):1},      # M1: A11+A22
        {(1,0):1, (1,1):1},      # M2: A21+A22
        {(0,0):1},               # M3: A11
        {(1,1):1},               # M4: A22
        {(0,0):1, (0,1):1},      # M5: A11+A12
        {(1,0):1, (0,0):-1},     # M6: A21-A11
        {(0,1):1, (1,1):-1},     # M7: A12-A22
    ]
    beta = [
        {(0,0):1, (1,1):1},      # M1: B11+B22
        {(0,0):1},               # M2: B11
        {(0,1):1, (1,1):-1},     # M3: B12-B22
        {(1,0):1, (0,0):-1},     # M4: B21-B11
        {(1,1):1},               # M5: B22
        {(0,0):1, (0,1):1},      # M6: B11+B12
        {(1,0):1, (1,1):1},      # M7: B21+B22
    ]
    # gamma: which C blocks each M contributes to, with sign
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 - M6
    gamma = [
        {(0,0):1, (1,1):1},      # M1: +C11, +C22
        {(1,0):1, (1,1):-1},     # M2: +C21, -C22
        {(0,1):1, (1,1):-1},     # M3: +C12, -C22
        {(0,0):1, (1,0):1},      # M4: +C11, +C21
        {(0,0):-1, (0,1):1},     # M5: -C11, +C12
        {(1,1):-1},              # M6: -C22
        {(0,0):1},               # M7: +C11
    ]
    
    # Build schoolbook for 2x2 * 2x4
    U_sub, V_sub, W_sub, _ = _schoolbook(2, 2, 4)
    
    # Now build full U, V, W
    for t in range(7):
        off = t * sub_rank
        for c in range(sub_rank):
            # In schoolbook for 2x2*2x4, column c corresponds to (i_s, j_s, k_s)
            # where c = i_s * m_sub * p_sub + j_s * p_sub + k_s = 8*i_s + 4*j_s + k_s
            i_s = c // 8
            j_s = (c % 8) // 4
            k_s = c % 4
            
            # Update U: for each (br, bc) in alpha[t]
            for (br, bc), a_coeff in alpha[t].items():
                full_i = br * 2 + i_s
                full_j = bc * 2 + j_s
                U[full_i * 4 + full_j][off + c] += a_coeff
            
            # Update V: for each (br, bc) in beta[t]
            for (br, bc), b_coeff in beta[t].items():
                full_j = br * 2 + j_s
                full_k = bc * 4 + k_s
                V[full_j * 8 + full_k][off + c] += b_coeff
            
            # Update W: for each (cr, cc) in gamma[t]
            for (cr, cc), g_coeff in gamma[t].items():
                full_k = cc * 4 + k_s
                full_i = cr * 2 + i_s
                W[full_k * 4 + full_i][off + c] += g_coeff
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    try:
        result = _strassen_4x4_4x8()
        
        # Verify the decomposition numerically
        U, V, W = result["U"], result["V"], result["W"]
        n, m, p = 4, 4, 8
        rank = len(U[0])
        
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # For matrix multiplication tensor: T[i,j,k] = 1 for all i,j,k
        # Check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return _schoolbook_fallback()
        
        return result
    except Exception:
        return _schoolbook_fallback()

def _schoolbook_fallback():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U, V, W, _ = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
