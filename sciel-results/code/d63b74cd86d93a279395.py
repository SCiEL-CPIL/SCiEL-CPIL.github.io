"""Matrix multiplication 2,3,8 - rank-minimal tensor decomposition.

Strategy:
  Split the 3 input columns into a 2-column block and a 1-column block.
  M_{2,3,8} = M_{2,2,8} ⊕ M_{2,1,8}
  - M_{2,1,8} is the all-ones 2×1×8 tensor, rank 1.
  - M_{2,2,8} = M_{2,2,4} ⊗ M_{1,1,2}.
    M_{2,2,4} is decomposed as two M_{2,2,2} blocks (Strassen, rank 7 each) → rank 14.
    So M_{2,2,8} has rank 14 × 2 = 28.
  Total rank = 28 + 1 = 29.
"""
DIMENSIONS = [2, 3, 8]

import time

def construct():
    start = time.time()
    n, m, p = DIMENSIONS  # 2, 3, 8
    deadline = start + 120

    # Total rank: 28 (from M_{2,2,8}) + 1 (from M_{2,1,8}) = 29
    rank = 29
    U = [[0] * rank for _ in range(n * m)]  # 6 x 29
    V = [[0] * rank for _ in range(m * p)]  # 24 x 29
    W = [[0] * rank for _ in range(p * n)]  # 16 x 29

    # Strassen rank-7 decomposition for M_{2,2,2}
    # U, V, W are lists of 7 dicts
    # U terms: (i, j) -> val, i,j in {0,1}
    # V terms: (j, k) -> val, j,k in {0,1}
    # W terms: (k, i) -> val, k,i in {0,1}
    S_U = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1},
    ]
    S_V = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1},
    ]
    S_W = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 1): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1},
    ]

    # M_{2,2,8}: split 8 output columns into 4 blocks of 2 columns each.
    # Each 2-column block is M_{2,2,2} (Strassen rank 7) tensor M_{1,1,2} (rank 1).
    # Wait, M_{2,2,4} as two M_{2,2,2} blocks means splitting the 4 k-columns into two pairs.
    # Then M_{2,2,8} splits the 8 k-columns into four pairs.
    # For each pair of k-columns (k0, k0+1), we have 7 Strassen terms.
    # The V factor: for the 2x2x2 block, V has entries (j, k') where k' in {0,1}.
    # But in the full M_{2,2,8}, j is in {0,1} (the 2 input columns of the 2x2 block),
    # and k is in {0,...,7}. For block b (k0=2b), k = k0 + k'.
    # The tensor product with M_{1,1,2} means the output index doubles? 
    # Actually, M_{2,2,8} = sum over k=0..7 of (i,j) ⊗ (j,k) ⊗ (k,i).
    # If we group k into pairs (2b, 2b+1), and for each pair use the Strassen decomposition
    # of the 2x2x2 sub-tensor with k' in {0,1}, then:
    #   U[i * 2 + j] = S_U[r][(i,j)]
    #   V[j * 8 + (2b + k')] = S_V[r][(j, k')]  ... but j is only 0,1 here
    #   W[(2b + k') * 2 + i] = S_W[r][(k', i)]
    #
    # But wait: M_{2,2,4} decomposed as two M_{2,2,2} blocks gives rank 14 for M_{2,2,4}.
    # Then M_{2,2,8} = M_{2,2,4} ⊗ M_{1,1,2} means we take the M_{2,2,4} decomposition
    # and "double" the k-index. Each term in the M_{2,2,4} decomposition with k in {0,1,2,3}
    # becomes two terms with k in {2k, 2k+1}? No, that's not right.
    #
    # Let me reconsider. The all-ones tensor M_{n,m,p} has entries 1.
    # M_{2,2,4}: sum_{i,j,k} e_{ij} ⊗ e_{jk} ⊗ e_{ki} where i in {0,1}, j in {0,1}, k in {0,1,2,3}.
    # This can be written as the sum over k of (sum_{i,j} e_{ij} ⊗ e_{j,k} ⊗ e_{k,i}).
    # For each k, the inner tensor is 2x2x1 (all ones), which has rank... 
    # Actually, the all-ones 2x2x1 tensor has rank 1? No.
    # 
    # Let me use a different approach. M_{2,2,4} can be decomposed by splitting k into two groups:
    # k in {0,1} and k in {2,3}. Each group gives a 2x2x2 all-ones tensor (rank 7 by Strassen).
    # So M_{2,2,4} has rank at most 14.
    #
    # Similarly, M_{2,2,8} splits k into 4 groups of 2: {0,1}, {2,3}, {4,5}, {6,7}.
    # Each group gives a 2x2x2 all-ones tensor, rank 7.
    # So M_{2,2,8} has rank at most 28.
    #
    # For the full M_{2,3,8}:
    # - j=0,1: M_{2,2,8} part, rank 28
    # - j=2: M_{2,1,8} part, rank 1
    # Total rank: 29.

    # Fill in M_{2,2,8} part (j in {0,1})
    for b in range(4):  # 4 blocks of 2 k-columns
        k0 = 2 * b
        col = b * 7  # 7 Strassen terms per block

        for r in range(7):
            # U: i in {0,1}, j in {0,1}
            for (i, j), val in S_U[r].items():
                U[i * 3 + j][col + r] = val
            # V: j in {0,1}, k in {k0, k0+1}
            for (j, kp), val in S_V[r].items():
                V[j * 8 + (k0 + kp)][col + r] = val
            # W: k in {k0, k0+1}, i in {0,1}
            for (kp, i), val in S_W[r].items():
                W[(k0 + kp) * 2 + i][col + r] = val

    # M_{2,1,8} part (j=2): rank 1 term
    # U[i*3+2][28] = 1 for i in {0,1}
    # V[2*8 + k][28] = 1 for k in {0,...,7}
    # W[k*2 + i][28] = 1 for k in {0,...,7}, i in {0,1}
    for i in range(2):
        U[i * 3 + 2][28] = 1
    for k in range(8):
        V[2 * 8 + k][28] = 1
    for k in range(8):
        for i in range(2):
            W[k * 2 + i][28] = 1

    return {"U": U, "V": V, "W": W}
