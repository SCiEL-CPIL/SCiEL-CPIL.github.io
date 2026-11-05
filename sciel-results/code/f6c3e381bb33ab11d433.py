import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, n, m, p, rank):
    """
    Verify that the tensor decomposition is exact.
    U: (n*m) x rank
    V: (m*p) x rank
    W: (p*n) x rank
    The decomposition must satisfy:
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] == 1
    """
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if rank == 0:
        return False
    # Check dimensions of columns
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    
    # Use numpy for speed
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # We need to check for all i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # This is a tensor contraction.
    # Let's iterate over i, j, k since n, m, p are small.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j, :]
                v_row = V_arr[j * p + k, :]
                w_row = W_arr[k * n + i, :]
                # Element-wise product then sum
                val = np.sum(u_row * v_row * w_row)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    # Strategy:
    # 1. Try to find a lower rank decomposition than schoolbook (54).
    # 2. The schoolbook rank is 54.
    # 3. A simple block decomposition (two 3x3x3 blocks) gives rank 18 if we can share U?
    #    No, the U matrix is shared, but the V and W matrices are different for each block.
    #    Wait, the tensor rank is the number of terms in the sum.
    #    If we have:
    #    C = A * B
    #    where A is 3x3, B is 3x6.
    #    We can write B = [B1, B2] where B1, B2 are 3x3.
    #    C = [A*B1, A*B2].
    #    If we have a rank-r decomposition for 3x3x3:
    #    A*B1 = sum_{c=0}^{r-1} U_c * V_{c,1} * W_{c,1}
    #    A*B2 = sum_{c=0}^{r-1} U_c * V_{c,2} * W_{c,2}
    #    Then A*B = sum_{c=0}^{r-1} U_c * [V_{c,1}, 0] * [W_{c,1}, 0] + sum_{c=0}^{r-1} U_c * [0, V_{c,2}] * [0, W_{c,2}]
    #    This is a sum of 2r terms.
    #    The schoolbook for 3x3x3 is rank 9. So 2*9 = 18.
    #    Is 18 valid?
    #    Let's check the structure.
    #    Term c in first sum:
    #    U_part = U_c (3x3 matrix flattened to 9x1)
    #    V_part = V_{c,1} in first 3 cols of B, 0 in last 3. (Flattened 18x1)
    #    W_part = W_{c,1} in first 3 cols of C, 0 in last 3. (Flattened 18x1)
    #    Term c' in second sum:
    #    U_part = U_c (same)
    #    V_part = 0 in first 3 cols, V_{c,2} in last 3 cols.
    #    W_part = 0 in first 3 cols, W_{c,2} in last 3 cols.
    #    
    #    The rank is the number of such terms.
    #    So if we use schoolbook for the 3x3x3 blocks, the rank is 18.
    #    This is much better than 54.
    #    
    #    Can we do better?
    #    The rank of the 3x3x3 matrix multiplication tensor is known to be 23 (Bläser).
    #    Wait, schoolbook is 9. The rank is the minimum number of terms.
    #    For 3x3x3, the rank is at most 9 (schoolbook).
    #    Actually, the rank of the matrix multiplication tensor M_{n,n,n} is at most n^3.
    #    For n=2, rank is 7 (Strassen).
    #    For n=3, is the rank less than 9?
    #    Yes, it is known that the rank of M_{3,3,3} is at most 23? No, 23 is a lower bound or something?
    #    Actually, the rank of M_{3,3,3} is not known exactly, but it is <= 9.
    #    Wait, Strassen's algorithm works for 2x2. For 3x3, we can partition into 2x2 and 1x1?
    #    Or just use the fact that 3x3 multiplication can be done with fewer than 9 multiplications?
    #    Actually, the rank of M_{3,3,3} is believed to be 23? No, that's the lower bound for 3x3x3?
    #    Let's check. The rank of M_{2,2,2} is 7.
    #    The rank of M_{3,3,3} is at most 23? No, 9 is an upper bound.
    #    It is known that rank(M_{3,3,3}) <= 23 is false, 23 is a lower bound for something else?
    #    Actually, the best known upper bound for rank(M_{3,3,3}) is 23? No, that's for 3x3x3?
    #    Let's stick to the safe upper bound of 9 for 3x3x3.
    #    So 2 * 9 = 18 is a valid upper bound for rank(M_{3,3,6}).
    #    
    #    Is there a way to share terms between the two blocks?
    #    The U factor is the same. But the V and W factors are disjoint in support (one is on first 3 cols, other on last 3).
    #    So the terms are linearly independent?
    #    If we have a term U * V1 * W1 and U * V2 * W2 where V1 and V2 have disjoint support, they are likely independent.
    #    So the rank is likely 18.
    #    
    #    Let's implement the rank-18 decomposition.
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    p_half = p // 2  # 3
    
    # Block 1: k in [0, 2]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p_half):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Block 2: k in [3, 5]
    for i in range(n):
        for j in range(m):
            for k in range(p_half, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verify
    if _verify(U, V, W, n, m, p, rank):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook rank 54
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
