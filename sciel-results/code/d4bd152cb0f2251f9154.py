"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition for 4x4 matrix multiplication
    # Uses tensor product of 2x2 Strassen decomposition
    # Rank: 7 * 7 = 49
    
    n, m, p = DIMENSIONS  # 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2 Strassen decomposition vectors
    # For 2x2 matrices flattened as [m00, m01, m10, m11]
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),  # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),  # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]),  # M7
    ]
    
    # Block-level Strassen coefficients for 4x4 (split into 2x2 blocks)
    # Each is a vector of 4 coefficients for blocks (00, 01, 10, 11)
    block_strassen = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1_block
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),  # M2_block
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),  # M3_block
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),  # M4_block
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5_block
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6_block
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]),  # M7_block
    ]
    
    def block_flatten_indices(bi, bj):
        """Return the 4 flatten indices for block (bi, bj) in a 4x4 matrix."""
        base_i = 2 * bi
        base_j = 2 * bj
        return [
            base_i * 4 + base_j,
            base_i * 4 + base_j + 1,
            (base_i + 1) * 4 + base_j,
            (base_i + 1) * 4 + base_j + 1
        ]
    
    col = 0
    for u_block, v_block, w_block in block_strassen:
        for sub_u, sub_v, sub_w in strassen_2x2:
            U_vec = [0] * 16
            V_vec = [0] * 16
            W_vec = [0] * 16
            
            for bi in range(2):
                for bj in range(2):
                    bu = u_block[bi * 2 + bj]
                    bv = v_block[bi * 2 + bj]
                    bw = w_block[bi * 2 + bj]
                    
                    indices = block_flatten_indices(bi, bj)
                    for s in range(4):
                        if bu != 0:
                            U_vec[indices[s]] += bu * sub_u[s]
                        if bv != 0:
                            V_vec[indices[s]] += bv * sub_v[s]
                        if bw != 0:
                            W_vec[indices[s]] += bw * sub_w[s]
            
            for idx in range(16):
                U[idx][col] = U_vec[idx]
                V[idx][col] = V_vec[idx]
                W[idx][col] = W_vec[idx]
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
