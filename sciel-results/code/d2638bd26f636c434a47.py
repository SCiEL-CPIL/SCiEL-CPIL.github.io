"""Rewritten program: exact cochromatic number via branch-and-bound over maximal homogeneous sets."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

N = P['n']
DEADLINE = 85.0  # safety margin under 90s wall-clock


def _build_graph():
    n = P['n']
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return n, adj


def _maximal_cliques(adj, n):
    """Find all maximal cliques using Bron–Kerbosch with pivot."""
    results = []
    all_mask = (1 << n) - 1

    def bronk(r, p, x):
        if not p and not x:
            if r:
                results.append(r)
            return
        # Choose pivot u from p | x to minimize branching
        candidates = p | x
        if candidates:
            best_u = -1
            best_deg = -1
            tmp = candidates
            while tmp:
                lsb = tmp & (-tmp)
                u = lsb.bit_length() - 1
                deg = (p & adj[u]).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_u = u
                tmp ^= lsb
            if best_u >= 0:
                p_to_iter = p & ~(1 << best_u)
            else:
                p_to_iter = p
        else:
            p_to_iter = p

        tmp = p_to_iter
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            bronk(r | lsb, p & adj[v], x & adj[v])
            p &= ~(1 << v)
            x |= (1 << v)
            tmp ^= lsb

    bronk(0, all_mask, 0)
    return results


def _maximal_independent_sets(adj, n):
    """Find all maximal independent sets by finding maximal cliques in complement."""
    comp_adj = [0] * n
    all_mask = (1 << n) - 1
    for i in range(n):
        comp_adj[i] = all_mask ^ (1 << i) ^ adj[i]
    return _maximal_cliques(comp_adj, n)


def _greedy_partition(adj, n):
    """Greedy first-fit: returns a valid partition as list of lists."""
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            # Check if adding v keeps part homogeneous
            is_clique = True
            is_indep = True
            for u in part:
                if adj[u] & (1 << v):
                    is_indep = False
                else:
                    is_clique = False
                if not is_clique and not is_indep:
                    break
            if is_clique or is_indep:
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    return parts


def construct():
    start_time = time.time()

    n, adj = _build_graph()
    full_mask = (1 << n) - 1

    # 1. Get all maximal homogeneous sets
    max_cliques = _maximal_cliques(adj, n)
    max_indeps = _maximal_independent_sets(adj, n)

    max_homog = []
    seen = set()
    for mc in max_cliques:
        if mc not in seen:
            max_homog.append(mc)
            seen.add(mc)
    for mi in max_indeps:
        if mi not in seen:
            max_homog.append(mi)
            seen.add(mi)

    # Sort by size descending to try larger sets first
    max_homog.sort(key=lambda m: -m.bit_count())

    # Precompute for each vertex, which maximal homogeneous sets contain it
    vertex_to_sets = [[] for _ in range(n)]
    for idx, hs in enumerate(max_homog):
        for v in range(n):
            if (hs >> v) & 1:
                vertex_to_sets[v].append(idx)

    # Greedy initial upper bound
    greedy_parts = _greedy_partition(adj, n)
    best_k = len(greedy_parts)
    best_solution = greedy_parts

    # Precompute sizes for quick access
    hs_sizes = [m.bit_count() for m in max_homog]

    def solve(remaining_mask, current_parts, k_used, max_k):
        nonlocal best_k, best_solution

        if time.time() - start_time > DEADLINE:
            return

        if remaining_mask == 0:
            if k_used < best_k:
                best_k = k_used
                best_solution = list(current_parts)
            return

        # Lower bound: at least ceil(popcount(remaining) / max_size) more parts needed
        pop = remaining_mask.bit_count()

        # Find the largest maximal homogeneous set that is a subset of remaining_mask
        max_size = 1
        for hs in max_homog:
            if (hs & ~remaining_mask) == 0:
                size = hs.bit_count()
                if size > max_size:
                    max_size = size
        lb = (pop + max_size - 1) // max_size
        if k_used + lb >= max_k:
            return

        # Pick the vertex with fewest candidate homogeneous sets (most constrained)
        best_v = -1
        best_count = float('inf')
        for v in range(n):
            if (remaining_mask >> v) & 1:
                count = 0
                for idx in vertex_to_sets[v]:
                    if (max_homog[idx] & ~remaining_mask) == 0:
                        count += 1
                if count < best_count:
                    best_count = count
                    best_v = v
                if best_count == 0:
                    break

        if best_count == 0:
            return

        # Try each maximal homogeneous set containing best_v that is a subset of remaining
        # Sort candidates by size descending to cover more vertices earlier
        candidates = []
        for idx in vertex_to_sets[best_v]:
            hs = max_homog[idx]
            if (hs & ~remaining_mask) == 0:
                candidates.append((hs_sizes[idx], idx))

        candidates.sort(key=lambda x: -x[0])

        for size, idx in candidates:
            hs = max_homog[idx]
            solve(remaining_mask & ~hs, current_parts + [hs], k_used + 1, max_k)

    # Try k = 1, 2, ..., best_k - 1
    for k in range(1, best_k):
        if time.time() - start_time > DEADLINE:
            break
        solve(full_mask, [], 0, k + 1)

    # Convert best_solution (list of masks) to list of lists
    if best_solution is not None:
        # Check if best_solution is already list of lists (from greedy) or list of masks
        if isinstance(best_solution[0], int):
            result = []
            for mask in best_solution:
                part = [v for v in range(n) if (mask >> v) & 1]
                result.append(part)
            result = [p for p in result if p]
        else:
            result = best_solution
    else:
        result = greedy_parts

    # Validate: every vertex exactly once
    flat = []
    for part in result:
        flat.extend(part)
    if sorted(flat) != list(range(n)):
        # Fallback to greedy
        result = _greedy_partition(adj, n)

    # Ensure types: lists of Python ints
    final = []
    for part in result:
        final.append([int(v) for v in part])

    return final

# EVOLVE-BLOCK-END
