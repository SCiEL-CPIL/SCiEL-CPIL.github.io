import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _strassen_2x2_2x2():
    """
    Returns U, V, W for 2x2 * 2x2 -> 2x2 with rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices:
    U[i*2+j, r]
    V[j*2+k, r]
    W[k*2+i, r]
    """
    n = m = p = 2
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen terms
    # M0 = (A00+A11)(B00+B11) -> C00+1, C11-1
    # M1 = (A10+A11)B00 -> C10+1, C11-1
    # M2 = A00(B01-B11) -> C01+1, C11+1
    # M3 = A11(B10-B00) -> C00+1, C10+1
    # M4 = (A00+A01)B11 -> C01+1, C11+1
    # M5 = (A10-A00)(B00+B01) -> C00-1
    # M6 = (A01-A11)(B10+B11) -> C00+1
    
    terms = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:-1}),
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {1:1, 3:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {0:-1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1})
    ]
    
    for r, (ac, bc, cc) in enumerate(terms):
        for ai, val in ac.items():
            i, j = divmod(ai, 2)
            U[i * m + j][r] += val
        for bi, val in bc.items():
            j, k = divmod(bi, 2)
            V[j * p + k][r] += val
        for ci, val in cc.items():
            i, k = divmod(ci, 2)
            W[k * n + i][r] += val
            
    return U, V, W

def _schoolbook_block(n, m, p, i_off, j_off, k_off, i_len, j_len, k_len):
    """
    Returns U, V, W for a schoolbook decomposition of a sub-block.
    The sub-block corresponds to A[i_off:i_off+i_len, j_off:j_off+j_len] * B[j_off:j_off+j_len, k_off:k_off+k_len]
    contributing to C[i_off:i_off+i_len, k_off:k_off+k_len].
    U, V, W are full size (n*m, rank), (m*p, rank), (p*n, rank) but mostly zeros.
    """
    rank = i_len * j_len * k_len
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for di in range(i_len):
        i = i_off + di
        for dj in range(j_len):
            j = j_off + dj
            for dk in range(k_len):
                k = k_off + dk
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W

def _add_decompositions(decomp1, decomp2):
    """
    Concatenates two decompositions.
    decomp1, decomp2 are dicts with U, V, W.
    """
    U1, V1, W1 = decomp1['U'], decomp1['V'], decomp1['W']
    U2, V2, W2 = decomp2['U'], decomp2['V'], decomp2['W']
    
    r1 = len(U1[0])
    r2 = len(U2[0])
    r_total = r1 + r2
    
    n_m = len(U1)
    m_p = len(V1)
    p_n = len(W1)
    
    U = [[0.0] * r_total for _ in range(n_m)]
    V = [[0.0] * r_total for _ in range(m_p)]
    W = [[0.0] * r_total for _ in range(p_n)]
    
    for row in range(n_m):
        for r in range(r1):
            U[row][r] = U1[row][r]
        for r in range(r2):
            U[row][r1 + r] = U2[row][r]
            
    for row in range(m_p):
        for r in range(r1):
            V[row][r] = V1[row][r]
        for r in range(r2):
            V[row][r1 + r] = V2[row][r]
            
    for row in range(p_n):
        for r in range(r1):
            W[row][r] = W1[row][r]
        for r in range(r2):
            W[row][r1 + r] = W2[row][r]
            
    return {'U': U, 'V': V, 'W': W}

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=8
    
    # Part 1: i in {0,1}, j in {0,1}. Strassen on 4 blocks of 2x2x2.
    # B is 3x8. We use rows 0,1 of B (j=0,1).
    # Cols of B are 0..7. We split into 4 pairs: (0,1), (2,3), (4,5), (6,7).
    
    decomp_total = None
    
    # Initialize with zeros? Or just accumulate.
    U_total = [[0.0] * 0 for _ in range(n*m)] # placeholder
    V_total = [[0.0] * 0 for _ in range(m*p)]
    W_total = [[0.0] * 0 for _ in range(p*n)]
    
    # We will build lists of terms and then combine.
    # Actually, _add_decompositions is fine.
    
    current = {'U': [[0.0] for _ in range(n*m)], 'V': [[0.0] for _ in range(m*p)], 'W': [[0.0] for _ in range(p*n)]}
    # This is rank 1, but we will add to it.
    # Better to start with None and handle first add.
    
    current = None
    
    # 1. Strassen parts: i in {0,1}, j in {0,1}
    # 4 blocks of B cols: k in {0,1}, {2,3}, {4,5}, {6,7}
    for block_idx in range(4):
        k_off = block_idx * 2
        # Get Strassen 2x2x2 decomp
        U_s, V_s, W_s = _strassen_2x2_2x2()
        
        # We need to map this 2x2x2 decomp to the full 3x3x8 space.
        # U_s is 4x7. Indices i*2+j for i,j in {0,1}.
        # V_s is 4x7. Indices j*2+k for j in {0,1}, k in {0,1} (local).
        # W_s is 4x7. Indices k*2+i for k in {0,1} (local), i in {0,1}.
        
        U_full = [[0.0] * 7 for _ in range(n*m)]
        V_full = [[0.0] * 7 for _ in range(m*p)]
        W_full = [[0.0] * 7 for _ in range(p*n)]
        
        # Map U
        for local_i in range(2):
            for local_j in range(2):
                global_i = local_i
                global_j = local_j
                local_idx = local_i * 2 + local_j
                global_idx = global_i * m + global_j
                for r in range(7):
                    U_full[global_idx][r] = U_s[local_idx][r]
                    
        # Map V
        for local_j in range(2):
            for local_k in range(2):
                global_j = local_j
                global_k = k_off + local_k
                local_idx = local_j * 2 + local_k
                global_idx = global_j * p + global_k
                for r in range(7):
                    V_full[global_idx][r] = V_s[local_idx][r]
                    
        # Map W
        for local_k in range(2):
            for local_i in range(2):
                global_k = k_off + local_k
                global_i = local_i
                local_idx = local_k * 2 + local_i
                global_idx = global_k * n + global_i
                for r in range(7):
                    W_full[global_idx][r] = W_s[local_idx][r]
                    
        current_block = {'U': U_full, 'V': V_full, 'W': W_full}
        
        if current is None:
            current = current_block
        else:
            current = _add_decompositions(current, current_block)
            
    # 2. Schoolbook: i in {0,1}, j=2
    # A[0:2, 2] * B[2, :]
    # i_len=2, j_len=1, k_len=8
    # i_off=0, j_off=2, k_off=0
    U_sb1, V_sb1, W_sb1 = _schoolbook_block(n, m, p, 0, 2, 0, 2, 1, 8)
    current = _add_decompositions(current, {'U': U_sb1, 'V': V_sb1, 'W': W_sb1})
    
    # 3. Schoolbook: i=2, j in {0,1}
    # A[2, 0:2] * B[0:2, :]
    # i_len=1, j_len=2, k_len=8
    # i_off=2, j_off=0, k_off=0
    U_sb2, V_sb2, W_sb2 = _schoolbook_block(n, m, p, 2, 0, 0, 1, 2, 8)
    current = _add_decompositions(current, {'U': U_sb2, 'V': V_sb2, 'W': W_sb2})
    
    # 4. Schoolbook: i=2, j=2
    # A[2, 2] * B[2, :]
    # i_len=1, j_len=1, k_len=8
    # i_off=2, j_off=2, k_off=0
    U_sb3, V_sb3, W_sb3 = _schoolbook_block(n, m, p, 2, 2, 0, 1, 1, 8)
    current = _add_decompositions(current, {'U': U_sb3, 'V': V_sb3, 'W': W_sb3})
    
    return current

# EVOLVE-BLOCK-END
