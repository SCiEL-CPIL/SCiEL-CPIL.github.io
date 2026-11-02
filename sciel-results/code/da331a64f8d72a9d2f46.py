"""Rewritten program for Matrix multiplication 2,2,13"""

import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=13
    # We use Strassen for 2x2x2 blocks and schoolbook for the remaining 2x2x1 block
    # 13 columns = 6 * 2 + 1
    # Rank for 2x2x2 is 7
    # Rank for 2x2x1 is 4
    # Total rank = 6 * 7 + 4 = 46
    
    rank = 46
    
    # U: (n*m, rank) = (4, 46)
    # V: (m*p, rank) = (26, 46)
    # W: (p*n, rank) = (26, 46)
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    
    # Strassen decomposition for a 2x2 * 2x2 block
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A * B
    # U indices: i*m + j -> a_ij
    #   0: a00, 1: a01, 2: a10, 3: a11
    # V indices: j*p + k -> b_jk
    # W indices: k*n + i -> c_ki (trace convention W[k*n+i, r])
    
    def add_strassen_block(k0, k1):
        nonlocal col
        
        # M1: (a00+a11)(b00+b11) -> contributes to c00, c11
        # U: a00=1, a11=1
        U[0, col] += 1; U[3, col] += 1
        # V: b00=1, b11=1 -> V[0*k0 + ... wait, V index is j*p + k
        # b00 is at j=0,k=k0 -> index 0*p + k0 = k0
        # b11 is at j=1,k=k1 -> index 1*p + k1 = p + k1
        V[k0, col] += 1; V[p + k1, col] += 1
        # W: c00 at k=k0,i=0 -> 0*n + k0 = k0
        #     c11 at k=k1,i=1 -> 1*n + k1 = n + k1
        W[k0, col] += 1; W[n + k1, col] += 1
        col += 1
        
        # M2: (a10+a11)b00 -> contributes to c10, c11
        U[2, col] += 1; U[3, col] += 1
        V[k0, col] += 1  # b00
        # c10 at k=k0,i=1 -> n+k0
        # c11 at k=k1,i=1 -> n+k1
        W[n + k0, col] += 1; W[n + k1, col] -= 1
        col += 1
        
        # M3: a00(b01-b11) -> contributes to c00, c01
        U[0, col] += 1
        V[p + k1, col] += 1  # b11
        V[k1, col] -= 1      # b01
        # c00 at k=k0,i=0 -> k0
        # c01 at k=k1,i=0 -> k1
        W[k0, col] += 1; W[k1, col] += 1
        col += 1
        
        # M4: a11(b10-b00) -> contributes to c00, c01
        U[3, col] += 1
        V[p + k0, col] += 1  # b10
        V[k0, col] -= 1      # b00
        W[k0, col] += 1; W[k1, col] += 1
        col += 1
        
        # M5: (a00+a01)b11 -> contributes to c00, c01
        U[0, col] += 1; U[1, col] += 1
        V[p + k1, col] += 1  # b11
        W[k0, col] -= 1; W[k1, col] += 1
        col += 1
        
        # M6: (a10-a00)(b00+b01) -> contributes to c11
        U[2, col] += 1; U[0, col] -= 1
        V[k0, col] += 1; V[k1, col] += 1
        W[n + k1, col] += 1
        col += 1
        
        # M7: (a01-a11)(b10+b11) -> contributes to c00
        U[1, col] += 1; U[3, col] -= 1
        V[p + k0, col] += 1; V[p + k1, col] += 1
        W[k0, col] += 1
        col += 1
    
    # Process 6 blocks of 2 columns
    for block_idx in range(6):
        k0 = block_idx * 2
        k1 = block_idx * 2 + 1
        add_strassen_block(k0, k1)
    
    # Process last column k=12
    k = 12
    # Schoolbook for 2x2 * 2x1
    # c00 = a00*b0k + a01*b1k
    # c01 = a10*b0k + a11*b1k
    
    # Term 1: a00 * b0k -> c00
    U[0, col] += 1
    V[k, col] += 1
    W[k, col] += 1  # c00 at k=k, i=0
    col += 1
    
    # Term 2: a01 * b1k -> c00
    U[1, col] += 1
    V[p + k, col] += 1
    W[k, col] += 1
    col += 1
    
    # Term 3: a10 * b0k -> c10
    U[2, col] += 1
    V[k, col] += 1
    W[n + k, col] += 1  # c10 at k=k, i=1
    col += 1
    
    # Term 4: a11 * b1k -> c10
    U[3, col] += 1
    V[p + k, col] += 1
    W[n + k, col] += 1
    col += 1
    
    # Convert to lists for the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
