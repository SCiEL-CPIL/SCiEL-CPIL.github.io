"""
Matrix multiplication 5x5x5 tensor decomposition.

Approach: Block decomposition with 5 = 2 + 2 + 1 partition.
- Use Strassen's algorithm (rank 7) for 2x2x2 blocks
- Use schoolbook (rank n*m*p) for all other blocks

Total rank: 117
"""

DIMENSIONS = [5, 5, 5]


def _strassen_factors():
    """
    Return U, V, W for Strassen's 2x2x2 algorithm.
    """
    rank = 7
    U = [
        [1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 0, 0],
    ]
    V = [
        [1, 0, 0, 1, 1, 1, 0],
        [1, 0, 0, 1, -1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 1, 1, 0, 1, -1, 1],
    ]
    W = [
        [1, 0, 0, -1, 1, 0, 0],
        [0, 1, 0, 0, -1, 0, 1],
        [0, 0, 1, 1, 0, -1, 0],
        [1, 0, -1, 0, 0, 1, 0],
    ]
    return U, V, W


def _add_block_factors(U, V, W, rank_offset, nI, mK, pJ, i_off, k_off, j_off):
    """
    Add factors for a block multiplication A_IK * B_KJ -> C_IJ.
    nI, mK, pJ are the dimensions of the blocks.
    i_off, k_off, j_off are the offsets in the global 5x5 matrices.
    Returns the new rank offset.
    """
    if nI == 2 and mK == 2 and pJ == 2:
        U_s, V_s, W_s = _strassen_factors()
        rank = 7
        for r in range(rank):
            # Map U
            for l in range(4):
                i_loc = l // 2
                j_loc = l % 2
                u_idx = (i_off + i_loc) * 5 + (k_off + j_loc)
                U[u_idx][rank_offset + r] += U_s[l][r]
            
            # Map V
            for l in range(4):
                j_loc = l // 2
                k_loc = l % 2
                v_idx = (k_off + j_loc) * 5 + (j_off + k_loc)
                V[v_idx][rank_offset + r] += V_s[l][r]
            
            # Map W
            for l in range(4):
                k_loc = l // 2
                i_loc = l % 2
                w_idx = (j_off + k_loc) * 5 + (i_off + i_loc)
                W[w_idx][rank_offset + r] += W_s[l][r]
        return rank_offset + rank
    else:
        rank = nI * mK * pJ
        col = rank_offset
        for i in range(nI):
            for j in range(mK):
                for k in range(pJ):
                    u_idx = (i_off + i) * 5 + (k_off + j)
                    v_idx = (k_off + j) * 5 + (j_off + k)
                    w_idx = (j_off + k) * 5 + (i_off + i)
                    
                    U[u_idx][col] += 1
                    V[v_idx][col] += 1
                    W[w_idx][col] += 1
                    col += 1
        return rank_offset + rank


def _block_decompose_5x5x5():
    """
    Decompose 5x5x5 matrix mult into blocks using 5 = 2 + 2 + 1.
    """
    n = 5
    sizes = [2, 2, 1]
    offsets = [0, 2, 4]
    
    # Initialize U, V, W with zeros
    # We don't know the exact rank beforehand, but we can calculate it.
    # Or we can just use a list of lists and append columns?
    # Better to calculate rank first.
    total_rank = 0
    for I in range(3):
        for K in range(3):
            for J in range(3):
                nI = sizes[I]
                mK = sizes[K]
                pJ = sizes[J]
                if nI == 2 and mK == 2 and pJ == 2:
                    total_rank += 7
                else:
                    total_rank += nI * mK * pJ
    
    U = [[0] * total_rank for _ in range(n * n)]
    V = [[0] * total_rank for _ in range(n * n)]
    W = [[0] * total_rank for _ in range(n * n)]
    
    rank_offset = 0
    for I in range(3):
        for K in range(3):
            for J in range(3):
                nI = sizes[I]
                mK = sizes[K]
                pJ = sizes[J]
                i_off = offsets[I]
                k_off = offsets[K]
                j_off = offsets[J]
                
                rank_offset = _add_block_factors(U, V, W, rank_offset, nI, mK, pJ, i_off, k_off, j_off)
    
    return U, V, W


def construct():
    U, V, W = _block_decompose_5x5x5()
    return {"U": U, "V": V, "W": W}
