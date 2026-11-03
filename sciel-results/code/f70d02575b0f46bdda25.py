import time
import itertools

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact tensor decomposition for 3x5 by 5x7 matrix multiplication.
    
    Structural approach: Maximum-packing of disjoint 2x2x2 Strassen blocks.
    Each Strassen block uses 7 rank-1 terms to cover 8 (i,j,k) tensor positions,
    saving 1 rank compared to schoolbook. We find the maximum independent set
    of non-overlapping Strassen blocks, then fill the rest with schoolbook.
    
    With n=3, m=5, p=7:
    - Possible I-pairs: {0,1}, {1,2}  (2 choices)
    - Possible J-pairs: 4 adjacent pairs (4 choices)
    - Possible K-pairs: 5 adjacent pairs (5 choices)
    - Total possible blocks: 2*4*5 = 40
    
    Rank = 105 - K where K = number of Strassen blocks packed.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        U_cols.append([0] * U_rows)
        V_cols.append([0] * V_rows)
        W_cols.append([0] * W_rows)
        for idx, val in u_entries.items():
            U_cols[-1][idx] = val
        for idx, val in v_entries.items():
            V_cols[-1][idx] = val
        for idx, val in w_entries.items():
            W_cols[-1][idx] = val

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2

        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2

        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2

        # M1 = (a11+a22)*(b11+b22) -> W: +1 at c11, +1 at c22
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        # M2 = (a21+a22)*b11 -> W: +1 at c21, -1 at c22
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        # M3 = a11*(b12-b22) -> W: +1 at c12, +1 at c22
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        # M4 = a22*(b21-b11) -> W: +1 at c11, +1 at c21
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        # M5 = (a11+a12)*b22 -> W: -1 at c11, +1 at c12
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        # M6 = (a21-a11)*(b11+b12) -> W: +1 at c22
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        # M7 = (a12-a22)*(b21+b22) -> W: +1 at c11
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    def get_block_cells(i1, i2, j1, j2, k1, k2):
        """Return frozenset of (i,j,k) positions covered by this block."""
        cells = set()
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    cells.add((i, j, k))
        return frozenset(cells)

    # Enumerate all possible 2x2x2 Strassen blocks
    i_pairs = [(i, i+1) for i in range(n-1)]  # (0,1), (1,2)
    j_pairs = [(j, j+1) for j in range(m-1)]  # (0,1),(1,2),(2,3),(3,4)
    k_pairs = [(k, k+1) for k in range(p-1)]  # (0,1),(1,2),(2,3),(3,4),(4,5)

    # Build list of all possible blocks with their cell sets
    all_blocks = []
    for i1, i2 in i_pairs:
        for j1, j2 in j_pairs:
            for k1, k2 in k_pairs:
                cells = get_block_cells(i1, i2, j1, j2, k1, k2)
                all_blocks.append(((i1, i2, j1, j2, k1, k2), cells))

    # Precompute conflict graph using bitmasks for speed
    block_count = len(all_blocks)
    conflict_masks = [0] * block_count
    for a in range(block_count):
        for b in range(a+1, block_count):
            if all_blocks[a][1] & all_blocks[b][1]:
                conflict_masks[a] |= (1 << b)
                conflict_masks[b] |= (1 << a)

    def greedy_pack():
        """Greedy: pick block with fewest conflicts among remaining."""
        remaining = set(range(block_count))
        selected = []
        while remaining:
            best = min(remaining, key=lambda x: bin(conflict_masks[x] & ((1 << block_count) - 1) & sum((1 << r) for r in remaining)).count('1'))
            selected.append(best)
            remaining.discard(best)
            # Remove all conflicting blocks
            mask = conflict_masks[best]
            remaining = {r for r in remaining if not (mask & (1 << r))}
        return selected

    def greedy_coloring_bound(remaining):
        """
        Compute an upper bound on MIS size using greedy coloring.
        The number of colors in a greedy coloring of the complement graph
        gives an upper bound on the independence number.
        """
        if not remaining:
            return 0
        rem_list = list(remaining)
        colors = {}
        next_color = 0
        # Order by degree in remaining (ascending for tighter bound)
        rem_sorted = sorted(rem_list, key=lambda x: bin(conflict_masks[x] & sum((1 << r) for r in remaining)).count('1'))
        for v in rem_sorted:
            # Find first available color
            used_colors = set()
            for u in remaining:
                if u != v and (conflict_masks[v] & (1 << u)):
                    if u in colors:
                        used_colors.add(colors[u])
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
            next_color = max(next_color, c + 1)
        return next_color

    def branch_and_bound_mis():
        """Find maximum independent set via branch and bound with coloring bound."""
        import sys
        sys.setrecursionlimit(10000)
        
        best_set = []
        best_size = 0
        time_limit = start_time + 120.0  # Hard stop
        
        def bbs(remaining, current):
            nonlocal best_set, best_size
            if time.time() > time_limit:
                return
            if not remaining:
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = list(current)
                return
            
            # Upper bound via coloring
            ub = greedy_coloring_bound(remaining)
            if len(current) + ub <= best_size:
                return
            
            # Pick vertex with max degree in remaining
            v = max(remaining, key=lambda x: bin(conflict_masks[x] & sum((1 << r) for r in remaining)).count('1'))
            
            # Branch 1: include v
            current.append(v)
            new_remaining = {r for r in remaining if r != v and not (conflict_masks[v] & (1 << r))}
            bbs(new_remaining, current)
            current.pop()
            
            # Branch 2: exclude v
            new_remaining2 = remaining - {v}
            bbs(new_remaining2, current)
        
        bbs(set(range(block_count)), [])
        return best_set

    def local_search_swap(selected, remaining):
        """Try to improve by swapping: remove 1 or 2 selected, add 2 or 3 from remaining."""
        selected = set(selected)
        remaining = set(remaining)
        improved = True
        while improved:
            improved = False
            # Try: remove 1 selected, add 2 remaining
            for s in list(selected):
                new_sel = selected - {s}
                candidates = [r for r in remaining if not (conflict_masks[r] & sum((1 << x) for x in new_sel))]
                for i in range(len(candidates)):
                    for j in range(i+1, len(candidates)):
                        if not (conflict_masks[candidates[i]] & (1 << candidates[j])):
                            new_selected = new_sel | {candidates[i], candidates[j]}
                            if len(new_selected) > len(selected):
                                selected = new_selected
                                remaining = set(range(block_count)) - selected
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            # Try: remove 2 selected, add 3 remaining
            if not improved:
                sel_list = list(selected)
                for i in range(len(sel_list)):
                    for j in range(i+1, len(sel_list)):
                        s1, s2 = sel_list[i], sel_list[j]
                        new_sel = selected - {s1, s2}
                        candidates = [r for r in remaining if not (conflict_masks[r] & sum((1 << x) for x in new_sel))]
                        for a_i in range(len(candidates)):
                            for b_i in range(a_i+1, len(candidates)):
                                for c_i in range(b_i+1, len(candidates)):
                                    ca, cb, cc = candidates[a_i], candidates[b_i], candidates[c_i]
                                    if (not (conflict_masks[ca] & (1 << cb)) and 
                                        not (conflict_masks[ca] & (1 << cc)) and 
                                        not (conflict_masks[cb] & (1 << cc))):
                                        new_selected = new_sel | {ca, cb, cc}
                                        if len(new_selected) > len(selected):
                                            selected = new_selected
                                            remaining = set(range(block_count)) - selected
                                            improved = True
                                            break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
        return list(selected)

    # Strategy: try B&B with time limit, fall back to greedy+local search
    mis_result = None
    if time.time() < deadline - 20:
        mis_result = branch_and_bound_mis()
    
    if mis_result and len(mis_result) > 0:
        remaining_after = set(range(block_count)) - set(mis_result)
        final_selected = local_search_swap(mis_result, remaining_after)
    else:
        best_greedy = greedy_pack()
        remaining_after = set(range(block_count)) - set(best_greedy)
        final_selected = local_search_swap(best_greedy, remaining_after)

    selected_indices = sorted(final_selected)

    # Track covered cells
    covered = set()
    for idx in selected_indices:
        block_params, cells = all_blocks[idx]
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        covered.update(cells)

    # Add schoolbook terms for uncovered (i,j,k) triples
    timeout_flag = False
    for i in range(n):
        if timeout_flag:
            break
        for j in range(m):
            if timeout_flag:
                break
            for k in range(p):
                if time.time() > deadline:
                    timeout_flag = True
                    break
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)

    # Convert to matrix format
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]

    for c in range(rank):
        for r in range(U_rows):
            U[r][c] = U_cols[c][r]
        for r in range(V_rows):
            V[r][c] = V_cols[c][r]
        for r in range(W_rows):
            W[r][c] = W_cols[c][r]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
