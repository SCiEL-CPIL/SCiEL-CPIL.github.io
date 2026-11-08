import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Try multiple block tilings and pick the one with lowest rank
    # Each tiling is a list of (i0, i1, j0, j1, k0, k1) blocks
    # Rank = 7 * num_blocks + num_uncovered_entries
    
    def compute_rank(blocks):
        covered = set()
        for (i0, i1, j0, j1, k0, k1) in blocks:
            for ii in (i0, i1):
                for jj in (j0, j1):
                    for kk in (k0, k1):
                        covered.add((ii, jj, kk))
        total = n * m * p
        uncovered = total - len(covered)
        return 7 * len(blocks) + uncovered
    
    # Generate candidate tilings
    # Strategy: try different combinations of i-pairs, j-pairs, k-pairs
    i_pairs = [(0,1), (1,2), (0,2)]
    j_pairs = [(0,1), (2,3), (0,2), (1,3), (0,3), (1,2)]
    k_pairs = [(0,1), (2,3), (4,5), (0,2), (1,3), (4,6), (0,4), (1,5), (2,6), (3,5), (3,6), (5,6)]
    
    best_rank = 78
    best_blocks = None
    
    # Exhaustive search over small subsets of blocks
    # We can have at most floor(84/8) = 10 blocks theoretically
    # But practically, blocks must have distinct (i,j,k) coverage for efficiency
    
    # Try all combinations of up to 8 blocks from a curated set
    # First, let's try systematic tilings
    
    # Tiling 1: Original (i in {0,1} only)
    t1 = [
        (0,1,0,1,0,1), (0,1,0,1,2,3), (0,1,0,1,4,5),
        (0,1,2,3,0,1), (0,1,2,3,2,3), (0,1,2,3,4,5),
    ]
    
    # Tiling 2: Mix i pairs
    t2 = [
        (0,1,0,1,0,1), (0,1,0,1,2,3), (0,1,0,1,4,5),
        (1,2,2,3,0,1), (1,2,2,3,2,3), (1,2,2,3,4,5),
    ]
    
    # Tiling 3: More mixed
    t3 = [
        (0,1,0,1,0,1), (0,1,2,3,2,3), (0,1,2,3,4,5),
        (1,2,0,1,0,1), (1,2,0,1,2,3), (1,2,0,1,4,5),
    ]
    
    # Tiling 4: Use (0,2) i pair
    t4 = [
        (0,1,0,1,0,1), (0,1,0,1,2,3), (0,1,0,1,4,5),
        (1,2,2,3,0,1), (1,2,2,3,2,3), (1,2,2,3,4,5),
    ]
    
    # Tiling 5: 7 blocks with better coverage
    t5 = [
        (0,1,0,1,0,1), (0,1,0,1,2,3), (0,1,0,1,4,5),
        (0,1,2,3,0,1), (0,1,2,3,2,3), (0,1,2,3,4,5),
        (2,2,0,1,6,6),  # invalid - need distinct indices
    ]
    
    # Let's do a smarter search: try all combinations of blocks
    # where blocks don't share the same (i0,i1,j0,j1) or (i0,i1,k0,k1)
    
    all_candidate_blocks = []
    for i0, i1 in i_pairs:
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                all_candidate_blocks.append((i0, i1, j0, j1, k0, k1))
    
    # Greedy: pick blocks that maximize new coverage
    def greedy_tiling(max_blocks=10):
        covered = set()
        chosen = []
        remaining = list(all_candidate_blocks)
        
        for _ in range(max_blocks):
            best_block = None
            best_gain = 0
            for block in remaining:
                i0, i1, j0, j1, k0, k1 = block
                new_entries = set()
                for ii in (i0, i1):
                    for jj in (j0, j1):
                        for kk in (k0, k1):
                            if (ii, jj, kk) not in covered:
                                new_entries.add((ii, jj, kk))
                gain = len(new_entries)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            if best_block is None or best_gain == 0:
                break
            chosen.append(best_block)
            i0, i1, j0, j1, k0, k1 = best_block
            for ii in (i0, i1):
                for jj in (j0, j1):
                    for kk in (k0, k1):
                        covered.add((ii, jj, kk))
            remaining.remove(best_block)
        
        return chosen, covered
    
    # Try greedy with different max_blocks
    best_rank = 78
    best_blocks = None
    
    for max_b in range(4, 12):
        blocks, covered = greedy_tiling(max_b)
        r = 7 * len(blocks) + (n*m*p - len(covered))
        if r < best_rank:
            best_rank = r
            best_blocks = blocks
    
    # Also try the fixed tilings
    for t in [t1, t2, t3, t4]:
        r = compute_rank(t)
        if r < best_rank:
            best_rank = r
            best_blocks = t
    
    if best_blocks is None:
        best_blocks = t1
        best_rank = 78
    
    rank = best_rank
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    col = 0

    def add_strassen_block(col_start, i0, i1, j0, j1, k0, k1):
        # M1
        U[i0 * m + j0, col_start] = 1
        U[i1 * m + j1, col_start] = 1
        V[j0 * p + k0, col_start] = 1
        V[j1 * p + k1, col_start] = 1
        W[k0 * n + i0, col_start] = 1
        W[k1 * n + i1, col_start] = 1

        # M2
        U[i1 * m + j0, col_start + 1] = 1
        U[i1 * m + j1, col_start + 1] = 1
        V[j0 * p + k0, col_start + 1] = 1
        W[k0 * n + i1, col_start + 1] = 1
        W[k1 * n + i1, col_start + 1] = -1

        # M3
        U[i0 * m + j0, col_start + 2] = 1
        V[j0 * p + k1, col_start + 2] = 1
        V[j1 * p + k1, col_start + 2] = -1
        W[k1 * n + i0, col_start + 2] = 1
        W[k1 * n + i1, col_start + 2] = 1

        # M4
        U[i1 * m + j1, col_start + 3] = 1
        V[j1 * p + k0, col_start + 3] = 1
        V[j0 * p + k0, col_start + 3] = -1
        W[k0 * n + i0, col_start + 3] = 1
        W[k0 * n + i1, col_start + 3] = 1

        # M5
        U[i0 * m + j0, col_start + 4] = 1
        U[i0 * m + j1, col_start + 4] = 1
        V[j1 * p + k1, col_start + 4] = 1
        W[k0 * n + i0, col_start + 4] = -1
        W[k1 * n + i0, col_start + 4] = 1

        # M6
        U[i1 * m + j0, col_start + 5] = 1
        U[i0 * m + j0, col_start + 5] = -1
        V[j0 * p + k0, col_start + 5] = 1
        V[j0 * p + k1, col_start + 5] = 1
        W[k1 * n + i1, col_start + 5] = 1

        # M7
        U[i0 * m + j1, col_start + 6] = 1
        U[i1 * m + j1, col_start + 6] = -1
        V[j1 * p + k0, col_start + 6] = 1
        V[j1 * p + k1, col_start + 6] = 1
        W[k0 * n + i0, col_start + 6] = 1

        return col_start + 7

    covered = set()
    for block in best_blocks:
        i0, i1, j0, j1, k0, k1 = block
        col = add_strassen_block(col, i0, i1, j0, j1, k0, k1)
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covered.add((ii, jj, kk))

    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                if (i0, j0, k0) in covered:
                    continue
                U[i0 * m + j0, col] = 1
                V[j0 * p + k0, col] = 1
                W[k0 * n + i0, col] = 1
                col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
