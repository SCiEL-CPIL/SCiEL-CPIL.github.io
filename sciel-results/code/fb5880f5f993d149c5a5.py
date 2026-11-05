"""Improved program: bitmask-based maximum clique search with multi-start hubs."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin under 90s
    N = 512

    # Precompute compatibility: compat[a] is bitmask of b in 1..N with lcm(a,b) <= N
    # Index: number x maps to bit (x-1)
    compat = [0] * (N + 1)
    for a in range(1, N + 1):
        mask = 0
        for b in range(1, N + 1):
            if a * b // gcd(a, b) <= N:
                mask |= (1 << (b - 1))
        compat[a] = mask

    # Precompute popcount for 16-bit chunks
    popcount16 = [bin(i).count('1') for i in range(1 << 16)]

    def popcount(mask):
        """Fast popcount using precomputed table for 16-bit chunks."""
        total = 0
        while mask:
            total += popcount16[mask & 0xFFFF]
            mask >>= 16
        return total

    def get_divisors(x):
        divs = []
        for d in range(1, int(x**0.5) + 1):
            if x % d == 0:
                divs.append(d)
                if d != x // d:
                    divs.append(x // d)
        return divs

    def count_divisors(x):
        return len(get_divisors(x))

    # Score each number by its divisor count (proxy for hub quality)
    div_counts = [(count_divisors(k), k) for k in range(1, N + 1)]
    div_counts.sort(key=lambda x: (-x[0], x[1]))

    # Greedy expansion: given a starting set (as bitmask), add compatible numbers
    def expand(start_mask):
        s = start_mask
        # Numbers already in set
        in_set = [False] * (N + 1)
        for x in range(1, N + 1):
            if (s >> (x - 1)) & 1:
                in_set[x] = True

        # Find all numbers compatible with entire set
        candidates_mask = (1 << N) - 1
        for x in range(1, N + 1):
            if in_set[x]:
                candidates_mask &= compat[x]

        # Greedy: repeatedly add the number with most connections among remaining candidates
        changed = True
        while changed:
            changed = False
            # Among candidates not in set, find best
            best_x = 0
            best_score = -1
            for x in range(1, N + 1):
                if in_set[x]:
                    continue
                if not ((candidates_mask >> (x - 1)) & 1):
                    continue
                # Score: number of other candidates it's compatible with
                score = popcount(candidates_mask & compat[x])
                if score > best_score:
                    best_score = score
                    best_x = x
            if best_x == 0:
                break
            s |= (1 << (best_x - 1))
            in_set[best_x] = True
            candidates_mask &= compat[best_x]
            changed = True
        return s

    # Local search: 1-swap and 2-swap improvements
    def local_search(s):
        improved = True
        while improved:
            if time.time() > deadline - 3:
                break
            improved = False
            in_set = [False] * (N + 1)
            for x in range(1, N + 1):
                if (s >> (x - 1)) & 1:
                    in_set[x] = True

            # Compute candidates compatible with all in set
            candidates_mask = (1 << N) - 1
            for x in range(1, N + 1):
                if in_set[x]:
                    candidates_mask &= compat[x]

            # Try 1-swap: remove one element, add a better one
            for rem in range(1, N + 1):
                if not in_set[rem]:
                    continue
                s_without = s & ~(1 << (rem - 1))
                # Candidates compatible with s_without
                cm = (1 << N) - 1
                for x in range(1, N + 1):
                    if (s_without >> (x - 1)) & 1:
                        cm &= compat[x]
                # Find best addable
                best_add = 0
                best_score = -1
                for add in range(1, N + 1):
                    if (s >> (add - 1)) & 1:
                        continue
                    if not ((cm >> (add - 1)) & 1):
                        continue
                    score = popcount(cm & compat[add])
                    if score > best_score:
                        best_score = score
                        best_add = add
                if best_add != 0:
                    new_s = s_without | (1 << (best_add - 1))
                    if popcount(new_s) > popcount(s):
                        s = new_s
                        improved = True
                        break
                if time.time() > deadline - 3:
                    break

            # Try 2-add: add two compatible elements at once
            if not improved:
                cm = (1 << N) - 1
                for x in range(1, N + 1):
                    if in_set[x]:
                        cm &= compat[x]
                addable = [x for x in range(1, N + 1)
                           if not in_set[x] and ((cm >> (x - 1)) & 1)]
                if len(addable) >= 2:
                    for i in range(len(addable)):
                        if time.time() > deadline - 3:
                            break
                        for j in range(i + 1, len(addable)):
                            a, b = addable[i], addable[j]
                            if (compat[a] >> (b - 1)) & 1:
                                new_s = s | (1 << (a - 1)) | (1 << (b - 1))
                                if popcount(new_s) > popcount(s):
                                    s = new_s
                                    improved = True
                                    break
                        if improved:
                            break
        return s

    # Branch and bound maximum clique on candidate set
    # candidates: list of node indices (0-based into the candidate list)
    # We work on a subgraph induced by candidates
    def max_clique_bnb(candidates, timeout=2.0):
        """Find maximum clique in the subgraph induced by candidates.
        candidates is a list of numbers (1..N). Returns the clique as a bitmask.
        """
        n = len(candidates)
        if n == 0:
            return 0

        # Build adjacency among candidates
        # adj[i] is bitmask of j (0-based) where candidates[i] and candidates[j] are compatible
        adj = [0] * n
        for i in range(n):
            ai = candidates[i]
            for j in range(i + 1, n):
                aj = candidates[j]
                if (compat[ai] >> (aj - 1)) & 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)

        # Order vertices by degree (descending) for better pruning
        degrees = [popcount(adj[i]) for i in range(n)]
        order = sorted(range(n), key=lambda i: -degrees[i])
        # Remap: new index -> old index
        remap = [0] * n
        for new_idx, old_idx in enumerate(order):
            remap[new_idx] = old_idx

        # Rebuild adjacency in new order
        adj2 = [0] * n
        for i in range(n):
            old_i = order[i]
            mask = adj[old_i]
            new_mask = 0
            for j in range(n):
                if j == i:
                    continue
                old_j = order[j]
                if (mask >> (old_j)) & 1:
                    new_mask |= (1 << j)
            adj2[i] = new_mask

        best_clique = [0] * n  # list of vertex indices in best clique
        best_size = 0

        # Bron-Kerbosch with pivot and coloring bound
        # color_sort: returns list of (vertex, color) sorted by color
        def color_sort(vertices):
            """Color the subgraph induced by vertices, return sorted list."""
            # Greedy coloring
            colors = {}
            uncolored = list(vertices)
            color = 0
            while uncolored:
                color += 1
                available = []
                remaining = []
                for v in uncolored:
                    # v can get this color if no neighbor in available has same color
                    can_color = True
                    for u in available:
                        if (adj2[v] >> u) & 1:
                            can_color = False
                            break
                    if can_color:
                        available.append(v)
                        colors[v] = color
                    else:
                        remaining.append(v)
                uncolored = remaining
            # Return vertices sorted by color (ascending), then by index
            return sorted(vertices, key=lambda v: (colors[v], v))

        def bronk(i, clique, candidates_set):
            nonlocal best_size, best_clique
            if time.time() > timeout:
                return
            if not candidates_set:
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique[:]
                return

            # Choose pivot: vertex with max degree in candidates_set
            pivot = -1
            max_deg = -1
            for v in candidates_set:
                d = popcount(adj2[v] & candidates_set)
                if d > max_deg:
                    max_deg = d
                    pivot = v

            # Branch on candidates not adjacent to pivot
            branch_candidates = [v for v in candidates_set if not ((adj2[pivot] >> v) & 1)] if pivot >= 0 else list(candidates_set)

            for v in branch_candidates:
                if time.time() > timeout:
                    return
                new_clique = clique + [v]
                new_candidates = candidates_set & adj2[v]
                # Remove v from candidates
                new_candidates &= ~(1 << v)
                bronk(i + 1, new_clique, new_candidates)
                candidates_set &= ~(1 << v)
                # Pruning: if remaining candidates can't beat best
                if popcount(candidates_set) + len(new_clique) - 1 <= best_size:
                    break

        # Initial candidates_set: all vertices
        full_set = (1 << n) - 1
        bronk(0, [], full_set)

        # Convert best_clique (new indices) back to original numbers
        result_mask = 0
        for idx in best_clique:
            num = candidates[order[idx]]
            result_mask |= (1 << (num - 1))
        return result_mask

    best_mask = 0
    best_size = 0

    # Multi-start: try each top hub's divisor set
    for cnt, hub in div_counts[:30]:
        if time.time() > deadline - 5:
            break
        divs = get_divisors(hub)
        start_mask = 0
        for d in divs:
            if d <= N:
                start_mask |= (1 << (d - 1))
        expanded = expand(start_mask)
        expanded = local_search(expanded)
        sz = popcount(expanded)
        if sz > best_size:
            best_size = sz
            best_mask = expanded

    # Also try single-element starts for high-degree nodes
    degree = [(popcount(compat[x]), x) for x in range(1, N + 1)]
    degree.sort(key=lambda x: (-x[0], x[1]))
    for deg, x in degree[:20]:
        if time.time() > deadline - 5:
            break
        start_mask = 1 << (x - 1)
        expanded = expand(start_mask)
        expanded = local_search(expanded)
        sz = popcount(expanded)
        if sz > best_size:
            best_size = sz
            best_mask = expanded

    # Try branch-and-bound on the full candidate set (all numbers)
    # This is expensive, so only try if we have time
    if time.time() < deadline - 30:
        # Try BnB on the full set
        all_numbers = list(range(1, N + 1))
        bnb_result = max_clique_bnb(all_numbers, timeout=25.0)
        bnb_size = popcount(bnb_result)
        if bnb_size > best_size:
            best_size = bnb_size
            best_mask = bnb_result

    # If we still have time, try BnB on smaller candidate sets from top hubs
    for cnt, hub in div_counts[:10]:
        if time.time() > deadline - 10:
            break
        divs = get_divisors(hub)
        # Include hub and its divisors, plus numbers compatible with all divisors
        cm = (1 << N) - 1
        for d in divs:
            cm &= compat[d]
        # Also include the divisors themselves
        for d in divs:
            cm |= (1 << (d - 1))
        # Add hub
        cm |= (1 << (hub - 1))

        candidates = [x for x in range(1, N + 1) if (cm >> (x - 1)) & 1]
        if len(candidates) > 0 and len(candidates) < 200:
            bnb_result = max_clique_bnb(candidates, timeout=8.0)
            bnb_size = popcount(bnb_result)
            if bnb_size > best_size:
                best_size = bnb_size
                best_mask = bnb_result

    # Convert to list
    result = []
    for x in range(1, N + 1):
        if (best_mask >> (x - 1)) & 1:
            result.append(x)

    if not result:
        result = [1]

    return result
# EVOLVE-BLOCK-END
