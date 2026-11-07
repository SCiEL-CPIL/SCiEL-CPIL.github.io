"""Cochromatic partition via memoized backtracking with vertex-picking and validity precomputation."""
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

N = P['n']
DEADLINE = 75.0


def _precompute_valid():
    """Precompute is_clique and is_indep for all masks using incremental DP."""
    adj = [0] * N
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    total = 1 << N
    is_clique = [False] * total
    is_indep = [False] * total
    is_valid = [False] * total

    is_clique[0] = True
    is_indep[0] = True
    is_valid[0] = True

    for m in range(1, total):
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        rest = m ^ lsb

        # Clique: rest is clique AND v adjacent to all in rest
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[m] = True

        # Independent: rest is independent AND v not adjacent to any in rest
        if is_indep[rest] and (adj[v] & rest) == 0:
            is_indep[m] = True

        is_valid[m] = is_clique[m] or is_indep[m]

    return is_valid, adj


def _greedy_partition(adj):
    """Greedy first-fit to get initial upper bound."""
    es = set()
    for u, v in P['edges']:
        es.add((u, v))
        es.add((v, u))

    def is_clique(lst):
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if (lst[i], lst[j]) not in es:
                    return False
        return True

    def is_indep(lst):
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if (lst[i], lst[j]) in es:
                    return False
        return True

    def greedy(order):
        parts = []
        for v in order:
            placed = False
            for part in parts:
                trial = part + [v]
                if is_clique(trial) or is_indep(trial):
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return parts

    best = greedy(list(range(N)))
    deg = [0] * N
    for u, v in P['edges']:
        deg[u] += 1
        deg[v] += 1
    best = min(best, greedy(sorted(range(N), key=lambda x: -deg[x])), key=len)
    best = min(best, greedy(sorted(range(N), key=lambda x: deg[x])), key=len)

    import random
    for _ in range(100):
        order = list(range(N))
        random.shuffle(order)
        best = min(best, greedy(order), key=len)

    return best


def construct():
    start = time.time()
    is_valid, adj = _precompute_valid()
    all_mask = (1 << N) - 1

    # Greedy initial upper bound
    greedy_parts = _greedy_partition(adj)
    best_k = len(greedy_parts)
    best_solution = [None] * best_k
    # Store solution as list of masks
    for i, part in enumerate(greedy_parts):
        mask = 0
        for v in part:
            mask |= (1 << v)
        best_solution[i] = mask

    # Memoization: best_k_for_mask[mask] = minimum parts used to cover 'mask'
    # We'll use a dict for sparse storage
    memo = {}  # mask -> min parts needed to cover it (from scratch)
    memo[0] = 0

    # For each vertex, precompute list of valid masks containing it, sorted by size desc
    valid_masks_by_vertex = [[] for _ in range(N)]
    for mask in range(1, 1 << N):
        if is_valid[mask]:
            # For each vertex in mask, add mask to its list
            m = mask
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                valid_masks_by_vertex[v].append(mask)
                m ^= lsb
    # Sort each list by size descending
    for v in range(N):
        valid_masks_by_vertex[v].sort(key=lambda x: -x.bit_count())

    # Precompute for each mask: max clique size and max indep size within it
    # This is expensive for all masks, so we'll compute on the fly with a helper
    # Instead, precompute max_clique_size and max_indep_size for each mask
    max_cliq = [0] * (1 << N)
    max_ind = [0] * (1 << N)
    for mask in range(1, 1 << N):
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        # Max clique in mask = max(max clique in rest, 1 + max clique in (rest & adj[v]))
        # But that's not quite right. Let's use: max clique in mask = max over v in mask of (1 + max clique in (mask & adj[v]))
        # Simpler: iterate
        best_c = 0
        best_i = 0
        m = mask
        while m:
            lsb2 = m & (-m)
            w = lsb2.bit_length() - 1
            # Clique containing w within mask: 1 + max clique in (mask & adj[w] & ~lsb2)
            sub = mask & adj[w] & ~lsb2
            # We need max clique in sub... this is getting complex
            # Let's just use the valid array: check if mask itself is clique
            m ^= lsb2
        # For efficiency, just compute max clique size by checking all subsets? No, too slow.
        # Alternative: for lower bound, use a simpler bound
        # Let's skip this and use a simpler lower bound
        pass

    # Simpler approach: lower bound = ceil(remaining_count / max_valid_size_in_remaining)
    # But we can do better: for the remaining mask, find the largest valid subset
    # Precompute largest valid subset size for each mask? That's 2^14 * 2^14 = too much.
    # Let's just use: lb = 1 if remaining != 0, and refine with greedy coloring of the cochromatic problem

    # Actually, let's use a different lower bound:
    # The chromatic number of the remaining subgraph gives a lower bound on independent sets needed
    # The clique cover number gives a lower bound on cliques needed
    # But we can mix, so it's complex. Let's use: lb >= max(1, ceil(popcount / max_possible_part_size))

    # For the search, we'll use a simple but effective approach:
    # Pick smallest vertex in remaining, try valid masks containing it

    node_count = [0]
    found_better = [False]
    best_k_ref = [best_k]
    best_sol_ref = [best_solution[:]]

    def search(rem_mask, chosen, count, max_count):
        nonlocal node_count, found_better, best_k_ref, best_sol_ref, best_k, best_solution

        if time.time() - start > DEADLINE:
            return False

        node_count[0] += 1

        if rem_mask == 0:
            if count < best_k_ref[0]:
                best_k_ref[0] = count
                best_sol_ref[0] = list(chosen)
                found_better[0] = True
            return True

        if count >= max_count:
            return False

        # Check memo: if we've seen rem_mask with <= count parts used before, prune
        if rem_mask in memo:
            if memo[rem_mask] <= count:
                return False
        # Also check: if we've already achieved a solution with count parts for this mask
        # Actually memo stores min parts from scratch, so if memo[rem_mask] <= count, prune

        # Lower bound: at least 1 more part needed
        if count + 1 >= max_count:
            return False

        # Pick smallest vertex in remaining
        lsb = rem_mask & (-rem_mask)
        v = lsb.bit_length() - 1

        # Try all valid masks containing v that are subsets of rem_mask
        for part_mask in valid_masks_by_vertex[v]:
            if part_mask & rem_mask != part_mask:
                continue
            if time.time() - start > DEADLINE:
                return False

            new_rem = rem_mask ^ part_mask
            chosen.append(part_mask)

            if search(new_rem, chosen, count + 1, max_count):
                chosen.pop()
                # Update memo for this state
                if rem_mask not in memo or count < memo[rem_mask]:
                    memo[rem_mask] = count
                return True

            chosen.pop()

        # No solution found from this state
        if rem_mask not in memo or count < memo[rem_mask]:
            memo[rem_mask] = count
        return False

    # Try k = 1, 2, ..., best_k - 1
    for k in range(1, best_k):
        if time.time() - start > DEADLINE:
            break
        found_better[0] = False
        best_k_ref[0] = k + 1  # allow up to k parts
        memo.clear()
        memo[0] = 0
        if search(all_mask, [], 0, k + 1):
            if found_better[0]:
                # Found solution with k parts
                best_k = k
                # Convert masks to lists
                result = []
                for m in best_sol_ref[0]:
                    lst = [i for i in range(N) if (m >> i) & 1]
                    result.append(lst)
                # Validate
                flat = []
                for part in result:
                    flat.extend(part)
                if sorted(flat) == list(range(N)):
                    return [[int(v) for v in part] for part in result]
                break
        # If not found, continue to larger k

    # Return the best solution found (greedy or better)
    if best_k < len(greedy_parts):
        result = []
        for m in best_sol_ref[0]:
            lst = [i for i in range(N) if (m >> i) & 1]
            result.append(lst)
        flat = []
        for part in result:
            flat.extend(part)
        if sorted(flat) == list(range(N)):
            return [[int(v) for v in part] for part in result]

    # Fallback to greedy
    return [[int(v) for v in part] for part in greedy_parts]
