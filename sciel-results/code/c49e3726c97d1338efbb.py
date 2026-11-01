import numpy as np
import time

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    
    try:
        U, V, W = _bclr_4x4(n, m, p)
        return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    # Fallback to trivial rank 64
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _bclr_4x4(n, m, p):
    """
    BCLR Rank-49 decomposition for 4x4 matrix multiplication.
    Uses Kronecker product of 2x2 Strassen factors with index permutations.
    """
    # 1. Define 2x2 Strassen Factors
    # U2, V2, W2 are 4x7 matrices
    # Indices: U2[2*i + j, r], V2[2*j + k, r], W2[2*k + i, r]
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # Strassen 2x2 definitions:
    # M0 = (A00+A11)(B00+B11) -> C00+=M0, C11+=M0
    # M1 = (A10+A11)B00     -> C10+=M1, C11-=M1
    # M2 = A00(B01-B11)     -> C01+=M2, C11+=M2
    # M3 = A11(B10-B00)     -> C00+=M3, C10+=M3
    # M4 = (A00+A01)B11     -> C01+=M4, C00-=M4
    # M5 = (A10-A00)(B00+B01) -> C11+=M5
    # M6 = (A01-A11)(B10+B11) -> C00+=M6
    
    # U2 (A factors)
    # r=0: A00, A11
    U2[0, 0] = 1; U2[3, 0] = 1
    # r=1: A10, A11
    U2[2, 1] = 1; U2[3, 1] = 1
    # r=2: A00
    U2[0, 2] = 1
    # r=3: A11
    U2[3, 3] = 1
    # r=4: A00, A01
    U2[0, 4] = 1; U2[1, 4] = 1
    # r=5: A10, -A00
    U2[2, 5] = 1; U2[0, 5] = -1
    # r=6: A01, -A11
    U2[1, 6] = 1; U2[3, 6] = -1
    
    # V2 (B factors)
    # r=0: B00, B11
    V2[0, 0] = 1; V2[3, 0] = 1
    # r=1: B00
    V2[0, 1] = 1
    # r=2: B01, -B11
    V2[1, 2] = 1; V2[3, 2] = -1
    # r=3: B10, -B00
    V2[2, 3] = 1; V2[0, 3] = -1
    # r=4: B11
    V2[3, 4] = 1
    # r=5: B00, B01
    V2[0, 5] = 1; V2[1, 5] = 1
    # r=6: B10, B11
    V2[2, 6] = 1; V2[3, 6] = 1
    
    # W2 (C factors)
    # r=0: C00, C11
    W2[0, 0] = 1; W2[3, 0] = 1
    # r=1: C10, -C11
    W2[2, 1] = 1; W2[3, 1] = -1
    # r=2: C01, C11
    W2[1, 2] = 1; W2[3, 2] = 1
    # r=3: C00, C10
    W2[0, 3] = 1; W2[2, 3] = 1
    # r=4: C01, -C00
    W2[1, 4] = 1; W2[0, 4] = -1
    # r=5: C11
    W2[3, 5] = 1
    # r=6: C00
    W2[0, 6] = 1
    
    # 2. Kronecker Product
    # U4_kron = U2 ⊗ U2
    # Rows: 16, Cols: 49
    U4_kron = np.kron(U2, U2)
    V4_kron = np.kron(V2, V2)
    W4_kron = np.kron(W2, W2)
    
    # 3. Permute Rows to match 4x4 indexing
    # U2 index: 2*i1 + j1. U2 kron index: (2*i1+j1)*4 + (2*i2+j2)
    # Target U4 index: 4*i + j where i=2*i1+i2, j=2*j1+j2
    # Target: 4*(2*i1+i2) + (2*j1+j2) = 8*i1 + 4*i2 + 2*j1 + j2
    
    # We need a permutation array perm_U such that U4[perm_U[k], :] = U4_kron[k, :]
    # Actually, we want U4[target_idx, :] = U4_kron[kron_idx, :]
    # So U4 = U4_kron[inv_perm, :]
    
    def get_permutation(n_size, kron_type):
        # n_size is 4
        # kron_type: 'U', 'V', 'W'
        perm = np.zeros(n_size*n_size, dtype=int)
        
        for i1 in range(2):
            for i2 in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        # Target Index
                        if kron_type == 'U':
                            # i = 2*i1+i2, j = 2*j1+j2
                            target = 4*(2*i1+i2) + (2*j1+j2)
                            # Kron Index
                            # r_A = 2*i1+j1, r_B = 2*i2+j2
                            kron_idx = (2*i1+j1)*4 + (2*i2+j2)
                        elif kron_type == 'V':
                            # j = 2*j1+j2, k = 2*k1+k2 (using k as second index in V2)
                            # V2 index: 2*j + k
                            # In kron: r_A = 2*j1+k1, r_B = 2*j2+k2
                            # Wait, V2 is defined with indices 2*j+k.
                            # So r_A corresponds to (j1, k1), r_B to (j2, k2).
                            # Target: 4*j + k where j=2*j1+j2, k=2*k1+k2
                            # Target: 8*j1 + 4*j2 + 2*k1 + k2
                            # Kron: (2*j1+k1)*4 + (2*j2+k2) = 8*j1 + 4*k1 + 2*j2 + k2
                            # Note: In V2, the second index is k. So j1, k1 are the "outer" block coords.
                            # My loop variables: j1, j2 are the 'j' part. k1, k2 are the 'k' part.
                            # So for V:
                            # Outer block indices: j1, k1. Inner: j2, k2.
                            # r_A = 2*j1 + k1
                            # r_B = 2*j2 + k2
                            # Target: 4*(2*j1+j2) + (2*k1+k2)
                            # Kron: (2*j1+k1)*4 + (2*j2+k2)
                            # My loop: i1->j1, i2->j2, j1->k1, j2->k2? 
                            # Let's use explicit variables for clarity.
                            pass
                        elif kron_type == 'W':
                            # W2 index: 2*k + i
                            # Target: 4*k + i where k=2*k1+k2, i=2*i1+i2
                            # Target: 8*k1 + 4*k2 + 2*i1 + i2
                            # Kron: (2*k1+i1)*4 + (2*k2+i2)
                            pass
                            
        # Re-do with clear variables
        perm = np.zeros(n_size*n_size, dtype=int)
        
        if kron_type == 'U':
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            target = 4*(2*i1+i2) + (2*j1+j2)
                            kron_idx = (2*i1+j1)*4 + (2*i2+j2)
                            perm[target] = kron_idx
                            
        elif kron_type == 'V':
            # V2 rows indexed by 2*j + k
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            target = 4*(2*j1+j2) + (2*k1+k2)
                            kron_idx = (2*j1+k1)*4 + (2*j2+k2)
                            perm[target] = kron_idx
                            
        elif kron_type == 'W':
            # W2 rows indexed by 2*k + i
            for k1 in range(2):
                for k2 in range(2):
                    for i1 in range(2):
                        for i2 in range(2):
                            target = 4*(2*k1+k2) + (2*i1+i2)
                            kron_idx = (2*k1+i1)*4 + (2*k2+i2)
                            perm[target] = kron_idx
                            
        return perm

    perm_U = get_permutation(4, 'U')
    perm_V = get_permutation(4, 'V')
    perm_W = get_permutation(4, 'W')
    
    U4 = U4_kron[perm_U, :]
    V4 = V4_kron[perm_V, :]
    W4 = W4_kron[perm_W, :]
    
    return U4.tolist(), V4.tolist(), W4.tolist()

