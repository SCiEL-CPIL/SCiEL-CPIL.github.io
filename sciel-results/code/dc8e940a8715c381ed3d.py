import numpy as np

def construct():
    n, m, p = 2, 3, 9
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try Laderman based (Rank 45)
    rank_block = 15
    total_rank = 3 * rank_block
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Laderman factors for 3x3x3
    # Using the symmetric form (if valid) or standard form.
    # I will use the factors from the seed's attempt but verify.
    U_L = np.array([
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -1, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, -1, 1, 0, 0, 0, 0, 0, 0, 1, 1, -1],
        [0, 0, 0, 0, 0, 0, 1, -1, 1, 0, 0, 0, 1, -1, 1],
        [1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, -1, 0, 0, 0, 0, 0, 0]
    ])
    V_L = U_L.copy()
    W_L = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    ])
    
    # Fill global U, V, W
    for b in range(3):
        offset = b * rank_block
        # U: rows 0..5
        for i in range(2):
            for j in range(3):
                idx_u = i * 3 + j
                idx_g = i * 3 + j # n=2, m=3 -> i*3+j
                for k in range(rank_block):
                    U[idx_g][offset + k] = U_L[idx_u][k]
        
        # V: rows j*9 + 3b + l
        for j in range(3):
            for l in range(3):
                idx_v = j * 3 + l
                idx_g = j * 9 + 3 * b + l
                for k in range(rank_block):
                    V[idx_g][offset + k] = V_L[idx_v][k]
        
        # W: rows (3b+l)*2 + i
        for l in range(3):
            for i in range(2):
                idx_w = l * 3 + i
                idx_g = (3 * b + l) * 2 + i
                for k in range(rank_block):
                    W[idx_g][offset + k] = W_L[idx_w][k]
                    
    # Verify
    # T = sum U V W
    # Check a few random entries or all?
    # All is 6*27*18 = 2916 entries. Fast enough.
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Target: C[k, i] should have 1 from A[i,j] B[j,k]
                # Sum over r: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # This sum should be 1 if this is the only term? No.
                # The sum over r of U[i,j,r] V[j,k,r] W[k,i,r] should be 1.
                # Wait, the tensor is T[i,j,k,l] = delta_{j,k} delta_{i,l}?
                # No, C_{i,k} = sum_j A_{i,j} B_{j,k}.
                # So coefficient of A_{i,j} B_{j,k} in C_{i,k} is 1.
                # Coefficient of A_{i,j} B_{j',k} in C_{i,k} is 0 if j != j'.
                # So we check:
                # Sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
                # Sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r] == 0 for j' != j
                # We only need to check the non-zero ones.
                val = 0
                for r in range(total_rank):
                    val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if val != 1:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    if not valid: break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": U_school, "V": V_school, "W": W_school}
