"""Matrix multiplication 4,4,8 - Strassen block decomposition."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
TIME_LIMIT = 145  # Safety margin before 150s kill

def construct():
    """
    Decompose 4x4 * 4x8 using Strassen at the 2x2 block level.
    Each 4x4x4 block uses 7 Strassen multipliers, each expanded with
    schoolbook 2x2x2 (rank 8), giving rank <= 56 per block.
    Total rank <= 112 (vs 128 schoolbook).
    """
    n, m, p = 4, 4, 8
    
    # Strassen 2x2 block combinations for 4x4x4
    # Each: (A_combo, B_combo) where combo is list of (block_row, block_col, coeff)
    strassen_combos = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),   # M1
        ([(1,0,1), (1,1,1)], [(0,0,1)]),              # M2
        ([(0,0,1)], [(0,1,1), (1,1,-1)]),             # M3
        ([(1,1,1)], [(1,0,1), (0,0,-1)]),             # M4
        ([(0,0,1), (0,1,1)], [(1,1,1)]),              # M5
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),    # M6
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),    # M7
    ]
    
    # C_block[ci,ck] = sum_mi c_coeff * M_i
    output_combos = {
        (0,0): {0:1, 3:1, 4:-1, 6:1},   # C11 = M1+M4-M5+M7
        (0,1): {2:1, 4:1},                # C12 = M3+M5
        (1,0): {1:1, 3:1},                # C21 = M2+M4
        (1,1): {0:1, 1:-1, 2:1, 5:1},   # C22 = M1-M2+M3+M6
    }
    
    from collections import defaultdict
    coeff_dict = defaultdict(int)
    
    for mi in range(7):
        a_combo, b_combo = strassen_combos[mi]
        for (ci, ck), mi_coeffs in output_combos.items():
            if mi not in mi_coeffs:
                continue
            c_coeff = mi_coeffs[mi]
            for di in range(2):
                for dk in range(2):
                    for dj in range(2):
                        for (ai, aj, ac) in a_combo:
                            for (bi, bj, bc) in b_combo:
                                # Valid only if A col block == B row block
                                if aj != bi:
                                    continue
                                i = 2 * ai + di
                                j = 2 * aj + dj
                                k = 2 * bj + dk
                                coeff_dict[(i, j, k)] += c_coeff * ac * bc
    
    nonzero = [(i, j, k, c) for (i, j, k), c in coeff_dict.items() if c != 0]
    rank_per_block = len(nonzero)
    total_rank = 2 * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for block in range(2):
        k_off = block * 4
        base_col = block * rank_per_block
        for idx, (i, j, k, c) in enumerate(nonzero):
            col = base_col + idx
            U[i * m + j][col] = c
            V[j * p + (k_off + k)][col] = 1
            W[(k_off + k) * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
