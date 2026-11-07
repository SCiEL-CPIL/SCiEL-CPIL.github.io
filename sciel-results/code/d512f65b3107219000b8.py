"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 55.0

    # Build adjacency as bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Precompute popcount for all masks (2^14 = 16384)
    popcount = [0] * (1 << n)
    for m in range(1, 1 << n):
        popcount[m] = popcount[m >> 1] + (m & 1)

    # Enumerate all valid parts (cliques and independent sets)
    all_parts = set()
    for mask in range(1, 1 << n):
        is_clique = True
        is_indep = True
        verts = []
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb

        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                u, v = verts[i], verts[j]
                if (adj[u] >> v) & 1:
                    is_indep = False
                else:
                    is_clique = False
            if not is_clique and not is_indep:
                break

        if is_clique or is_indep:
            all_parts.add(mask)

    # Sort by size descending
    all_parts_list = sorted(all_parts, key=lambda m: popcount[m], reverse=True)

    # Precompute for each vertex which parts contain it
    parts_by_vertex = [[] for _ in range(n)]
    for pm in all_parts_list:
        for v in range(n):
            if (pm >> v) & 1:
                parts_by_vertex[v].append(pm)

    # Memoization: best (minimum) depth at which we've solved this mask
    memo_min = {}
    memo_min[0] = 0

    # Better lower bound using matching on conflict graph
    def lower_bound(remaining):
        if remaining == 0:
            return 0
        
        # Quick check: can we cover remaining with one part?
        for pm in all_parts_list:
            if (pm & remaining) == pm:
                return 1
        
        # Greedy matching lower bound
        # Find maximum set of pairwise non-compatible parts that must be separate
        # Two parts are incompatible if their union is not a valid part
        covered = 0
        count = 0
        remaining_mask = remaining
        
        while remaining_mask:
            # Pick the vertex with fewest compatible options
            best_v = -1
            best_cnt = float('inf')
            rem = remaining_mask
            while rem:
                lsb = rem & (-rem)
                v = lsb.bit_length() - 1
                rem ^= lsb
                cnt = 0
                for pm in parts_by_vertex[v]:
                    if (pm & remaining_mask) == pm:
                        cnt += 1
                if cnt < best_cnt:
                    best_cnt = cnt
                    best_v = v
                    if cnt == 0:
                        break
            
            if best_cnt == 0:
                return popcount[remaining]
            
            # Pick the largest part containing best_v
            best_part = 1 << best_v
            best_sz = 1
            for pm in parts_by_vertex[best_v]:
                if (pm & remaining_mask) == pm:
                    sz = popcount[pm]
                    if sz > best_sz:
                        best_sz = sz
                        best_part = pm
            
            remaining_mask &= ~best_part
            count += 1
        
        return count

    # Best found so far
    best_found = [None]
    best_depth = [float('inf')]

    def search(remaining, depth, parts_list):
        if time.time() > deadline:
            return False
        if remaining == 0:
            if depth < best_depth[0]:
                best_depth[0] = depth
                best_found[0] = list(parts_list)
            return True

        # Lower bound pruning
        lb = lower_bound(remaining)
        if depth + lb > best_depth[0]:
            return False

        # Memoization check
        if remaining in memo_min:
            if memo_min[remaining] <= depth:
                return False

        # Pick the vertex with fewest options to reduce branching
        min_options = None
        min_v = -1
        rem = remaining
        while rem:
            lsb = rem & (-rem)
            v = lsb.bit_length() - 1
            rem ^= lsb
            cnt = 0
            for pm in parts_by_vertex[v]:
                if (pm & remaining) == pm:
                    cnt += 1
            if min_options is None or cnt < min_options:
                min_options = cnt
                min_v = v
                if cnt <= 1:
                    break

        if min_options == 0:
            memo_min[remaining] = depth
            return False

        # Try parts containing min_v, sorted by size descending
        candidates = [pm for pm in parts_by_vertex[min_v] if (pm & remaining) == pm]
        candidates.sort(key=lambda m: popcount[m], reverse=True)

        for part_mask in candidates:
            new_remaining = remaining & ~part_mask
            if new_remaining == remaining:
                continue

            # Prune if we know this subproblem needs too many parts
            if new_remaining in memo_min and memo_min[new_remaining] + 1 >= depth:
                continue

            parts_list.append([v for v in range(n) if (part_mask >> v) & 1])
            search(new_remaining, depth + 1, parts_list)
            parts_list.pop()

        memo_min[remaining] = depth
        return False

    # Get initial upper bound from greedy
    greedy_parts = []
    used = 0
    for v in range(n):
        if (used >> v) & 1:
            continue
        best_part = 1 << v
        best_sz = 1
        for pm in parts_by_vertex[v]:
            if (pm & ~used) == pm:
                sz = popcount[pm]
                if sz > best_sz:
                    best_sz = sz
                    best_part = pm
        greedy_parts.append([x for x in range(n) if (best_part >> x) & 1])
        used |= best_part

    best_depth[0] = len(greedy_parts)

    # Search for optimal
    parts_list = []
    search((1 << n) - 1, 0, parts_list)

    if best_found[0] is not None:
        result = [[int(v) for v in part] for part in best_found[0]]
        return result
    else:
        return [list(p) for p in greedy_parts]
# EVOLVE-BLOCK-END
