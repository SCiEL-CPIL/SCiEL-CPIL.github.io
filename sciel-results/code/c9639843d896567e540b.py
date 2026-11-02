"""Matrix multiplication 3x4 by 4x4 with rank-24 decomposition."""

def construct():
    n, m, p = 3, 4, 4
    rank = 24  # 4 blocks * 6 terms per block
    
    U = [[0] * rank for _ in range(n * m)]  # 12 x 24
    V = [[0] * rank for _ in range(m * p)]  # 16 x 24
    W = [[0] * rank for _ in range(p * n)]  # 12 x 24
    
    col = 0
    # Four blocks: (j_offset, k_offset) where j_offset in {0, 2}, k_offset in {0, 2}
    for j_offset in [0, 2]:
        for k_offset in [0, 2]:
            for i in range(n):
                # Term for j = j_offset: covers k = k_offset and k_offset+1
                U[i * m + j_offset][col] = 1
                V[j_offset * p + k_offset][col] = 1
                V[j_offset * p + k_offset + 1][col] = 1
                W[k_offset * n + i][col] = 1
                W[(k_offset + 1) * n + i][col] = 1
                col += 1
                
                # Term for j = j_offset + 1: covers k = k_offset and k_offset+1
                U[i * m + (j_offset + 1)][col] = 1
                V[(j_offset + 1) * p + k_offset][col] = 1
                V[(j_offset + 1) * p + k_offset + 1][col] = 1
                W[k_offset * n + i][col] = 1
                W[(k_offset + 1) * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
