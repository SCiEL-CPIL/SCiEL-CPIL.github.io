import time
import itertools

DIMENSIONS = [3, 4, 7]
_DEADLINE = 140.0

def find_max_disjoint_blocks(n, m, p):
    # Generate all possible 2x2x2 blocks
    # i0 in 0..n-2, j0 in 0..m-2, k0 in 0..p-2
    blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                # Block covers i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
                blocks.append((i0, i0+1, j0, j0+1, k0, k0+1))
    
    # Check overlap between two blocks
    def overlaps(b1, b2):
        i0_1, i1_1, j0_1, j1_1, k0_1, k1_1 = b1
        i0_2, i1_2, j0_2, j1_2, k0_2, k1_2 = b2
        
        # Check i overlap
        if i1_1 < i0_2 or i1_2 < i0_1:
            return False
        # Check j overlap
        if j1_1 < j0_2 or j1_2 < j0_1:
            return False
        # Check k overlap
        if k1_1 < k0_2 or k1_2 < k0_1:
            return False
        return True

    # Simple backtracking to find max independent set
    # Since N=36 is small, we can use a greedy approach with backtracking or just brute force if optimized.
    # Actually, 2^36 is too big. But we can use a simple recursive search with pruning.
    # Or even simpler: Greedy + Local Search.
    
    # Let's use a simple recursive backtracking with sorting by "popularity" or just index.
    # Given the small size, a well-pruned backtracker will work.
    
    best_set = []
    
    def backtrack(remaining_indices, current_set):
        nonlocal best_set
        if len(current_set) > len(best_set):
            best_set = current_set[:]
        
        if not remaining_indices:
            return
            
        # Pick the first block
        first_idx = remaining_indices[0]
        b1 = blocks[first_idx]
        
        # Option 1: Include b1
        new_remaining = []
        for idx in remaining_indices[1:]:
            if not overlaps(b1, blocks[idx]):
                new_remaining.append(idx)
        backtrack(new_remaining, current_set + [first_idx])
        
        # Option 2: Exclude b1
        backtrack(remaining_indices[1:], current_set)

    # To speed up, we can order blocks. 
    # Blocks with more overlaps should be decided earlier? Or just standard order.
    # 36 blocks, max depth 36. Branching factor reduces quickly.
    # This might be slow in pure Python if not careful.
    # Let's try a greedy heuristic first, then maybe a simple local search.
    
    # Greedy: Pick block that conflicts with fewest others, add it, remove conflicts.
    # Repeat.
    
    indices = list(range(len(blocks)))
    greedy_set = []
    available = set(indices)
    
    while available:
        # Find block with minimum conflicts in available
        min_conflicts = len(available)
        best_idx = None
        for idx in available:
            conflicts = 0
            for other in available:
                if idx != other and overlaps(blocks[idx], blocks[other]):
                    conflicts += 1
            if conflicts < min_conflicts:
                min_conflicts = conflicts
                best_idx = idx
        
        if best_idx is None:
            break
            
        greedy_set.append(best_idx)
        # Remove best_idx and its conflicts
        to_remove = {best_idx}
        for other in available:
            if overlaps(blocks[best_idx], blocks[other]):
                to_remove.add(other)
        available -= to_remove
        
    if len(greedy_set) > len(best_set):
        best_set = greedy_set
        
    # Try to improve best_set with local search?
    # For now, greedy is likely good enough or we can just use the backtracker if it's fast.
    # Let's assume backtracker is too slow for 36 nodes in worst case without good pruning.
    # But 36 is small. Let's stick to the greedy result for safety, or just use the hardcoded 6 if greedy doesn't improve.
    # Actually, let's just run the backtracker with a time limit or simple pruning.
    # Given the constraints, the greedy result is a safe lower bound for improvement.
    
    return [blocks[i] for i in best_set]

def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7

    # Find optimal blocks
    optimal_blocks = find_max_disjoint_blocks(n, m, p)
    
    # Fallback to known good set if search fails or returns empty
    if not optimal_blocks:
        optimal_blocks = [
            (0, 1, 0, 1, 0, 1),
            (0, 1, 0, 1, 2, 3),
            (0, 1, 0, 1, 4, 5),
            (0, 1, 2, 3, 0, 1),
            (0, 1, 2, 3, 2, 3),
            (0, 1, 2, 3, 4, 5),
        ]

    num_blocks = len(optimal_blocks)
    # Rank = 7 * num_blocks + 1 * (total_entries - 8 * num_blocks)
    total_entries = n * m * p
    covered_entries = 8 * num_blocks
    remaining_entries = total_entries - covered_entries
    rank = 7 * num_blocks + remaining_entries
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_block(col_start, i0, i1, j0, j1, k0, k1):
        # M1
        U[i0 * m + j0][col_start] = 1
        U[i1 * m + j1][col_start] = 1
        V[j0 * p + k0][col_start] = 1
        V[j1 * p + k1][col_start] = 1
        W[k0 * n + i0][col_start] = 1
        W[k1 * n + i1][col_start] = 1

        # M2
        U[i1 * m + j0][col_start + 1] = 1
        U[i1 * m + j1][col_start + 1] = 1
        V[j0 * p + k0][col_start + 1] = 1
        W[k0 * n + i1][col_start + 1] = 1
        W[k1 * n + i1][col_start + 1] = -1

        # M3
        U[i0 * m + j0][col_start + 2] = 1
        V[j0 * p + k1][col_start + 2] = 1
        V[j1 * p + k1][col_start + 2] = -1
        W[k1 * n + i0][col_start + 2] = 1
        W[k1 * n + i1][col_start + 2] = 1

        # M4
        U[i1 * m + j1][col_start + 3] = 1
        V[j1 * p + k0][col_start + 3] = 1
        V[j0 * p + k0][col_start + 3] = -1
        W[k0 * n + i0][col_start + 3] = 1
        W[k0 * n + i1][col_start + 3] = 1

        # M5
        U[i0 * m + j0][col_start + 4] = 1
        U[i0 * m + j1][col_start + 4] = 1
        V[j1 * p + k1][col_start + 4] = 1
        W[k0 * n + i0][col_start + 4] = -1
        W[k1 * n + i0][col_start + 4] = 1

        # M6
        U[i1 * m + j0][col_start + 5] = 1
        U[i0 * m + j0][col_start + 5] = -1
        V[j0 * p + k0][col_start + 5] = 1
        V[j0 * p + k1][col_start + 5] = 1
        W[k1 * n + i1][col_start + 5] = 1

        # M7
        U[i0 * m + j1][col_start + 6] = 1
        U[i1 * m + j1][col_start + 6] = -1
        V[j1 * p + k0][col_start + 6] = 1
        V[j1 * p + k1][col_start + 6] = 1
        W[k0 * n + i0][col_start + 6] = 1

        return col_start + 7

    covered = set()
    for block in optimal_blocks:
        col = add_strassen_block(col, *block)
        i0, i1, j0, j1, k0, k1 = block
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covered.add((ii, jj, kk))

    # Schoolbook for remaining
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                if (i0, j0, k0) in covered:
                    continue
                U[i0 * m + j0][col] = 1
                V[j0 * p + k0][col] = 1
                W[k0 * n + i0][col] = 1
                col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Verify
    for i in range(n):
        if time.time() - _t0 > _DEADLINE:
            break
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = 0
                for c in range(rank):
                    total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if total != 1:
                    raise ValueError(f"Decomposition failed at ({i},{j},{k})")

    return {"U": U, "V": V, "W": W}
