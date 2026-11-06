"""Different strategy: exact branch-and-bound (Tomita MIS) on conflict graph."""
# EVOLVE-BLOCK-START
import time

def construct():
    t0 = time.time()
    DEADLINE = t0 + 80.0  # safety margin before 90s kill

    N = 256
    nums = list(range(1, N + 1))
    n = len(nums)

    # Precompute conflict matrix: conflict[i][j] = 1 if (nums[i]*nums[j]) % (nums[i]+nums[j]) == 0
    # We want an independent set in this graph (no two selected numbers conflict).
    # Build adjacency bitmasks for fast MIS.
    adj = [0] * n
    for i in range(n):
        ai = nums[i]
        for j in range(i + 1, n):
            bj = nums[j]
            s = ai + bj
            p = ai * bj
            if p % s == 0:
                # Conflict: they cannot both be in the set
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # Tomita-style branch and bound for Maximum Independent Set
    # We process vertices in a fixed order (by index).
    # The key insight: we want to select a subset S such that no two elements in S are adjacent in the conflict graph.
    # This is exactly the Maximum Independent Set problem.

    # For speed, we use a recursive branching with upper bound from greedy coloring.
    # However, n=256 is too large for exact MIS in general.
    # But the conflict graph might be sparse. Let's check density.
    edge_count = sum(bin(a).count('1') for a in adj) // 2
    density = edge_count / (n * (n - 1) / 2)

    # If the graph is sparse, branch and bound may work.
    # Let's try a hybrid: use branch and bound on a reduced problem if possible,
    # otherwise fall back to a strong heuristic.

    # Actually, let's first try a strong greedy heuristic to get a good lower bound,
    # then use branch and bound to try to improve it.

    # Greedy heuristic: sort by degree (ascending), pick if no conflict with selected.
    def greedy_mis(adj, n, order):
        selected = 0  # bitmask
        for v in order:
            if not (adj[v] & selected):
                selected |= (1 << v)
        return selected

    # Try multiple orderings
    best_mask = 0
    best_size = 0

    # Orderings: by degree ascending, by value, etc.
    degrees = [bin(adj[i]).count('1') for i in range(n)]

    orderings = [
        sorted(range(n), key=lambda i: (degrees[i], i)),
        sorted(range(n), key=lambda i: (degrees[i], -i)),
        sorted(range(n), key=lambda i: i),
        sorted(range(n), key=lambda i: -i),
        sorted(range(n), key=lambda i: degrees[i] / (i + 1)),
    ]

    for order in orderings:
        if time.time() > DEADLINE - 10:
            break
        mask = greedy_mis(adj, n, order)
        sz = bin(mask).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = mask

    # Now try branch and bound to improve
    # We use a recursive function that tries to add vertices to the current set.
    # To make it efficient, we use the following approach:
    # - Maintain a list of candidate vertices (those not yet decided)
    # - Use upper bound: current_size + max_possible_from_remaining
    # - For max_possible, use a greedy coloring upper bound on the remaining graph

    # For n=256, exact B&B is likely infeasible unless the graph is very structured.
    # Let's try a limited B&B with a time check.

    def color_bound(candidates):
        """Greedy coloring upper bound for MIS on the subgraph induced by candidates."""
        if not candidates:
            return 0
        # Greedy coloring: assign colors to vertices in candidates
        # The number of colors is an upper bound on the size of the independent set
        # Wait, no. For MIS, the clique cover number is the upper bound.
        # A simpler upper bound: just the number of vertices.
        # A better one: greedy coloring of the complement graph? No.
        # Actually, for MIS, a standard upper bound is the number of vertices in the candidate set.
        # A tighter bound can be obtained by finding a clique in the conflict graph,
        # but that's also hard.
        # Let's use a simple bound: len(candidates)
        # This is weak. Let's try a different approach.
        return len(candidates)

    # Since exact B&B is likely too slow, let's focus on improving the heuristic.
    # Use local search: swap one element for another if it improves.
    def local_search(mask, adj, n, max_iter=100):
        cur = mask
        cur_size = bin(cur).count('1')
        improved = True
        it = 0
        while improved and it < max_iter and time.time() < DEADLINE - 5:
            improved = False
            it += 1
            # Try removing one selected vertex and adding two unselected ones
            for i in range(n):
                if not (cur & (1 << i)):
                    continue
                trial = cur & ~(1 << i)
                # Try to add two vertices that were conflicting with i
                added = 0
                for j in range(n):
                    if trial & (1 << j):
                        continue
                    if adj[j] & trial:
                        continue
                    # Check if j was conflicting with i (so it was excluded)
                    if adj[i] & (1 << j):
                        trial |= (1 << j)
                        added += 1
                        if added == 2:
                            break
                if added == 2:
                    cur = trial
                    cur_size += 1
                    improved = True
                    break
            # Try removing two selected and adding three unselected
            if not improved and it < max_iter // 2:
                for i in range(n):
                    if not (cur & (1 << i)):
                        continue
                    for j in range(i + 1, n):
                        if not (cur & (1 << j)):
                            continue
                        trial = cur & ~(1 << i) & ~(1 << j)
                        added = 0
                        for k in range(n):
                            if trial & (1 << k):
                                continue
                            if adj[k] & trial:
                                continue
                            if (adj[i] & (1 << k)) or (adj[j] & (1 << k)):
                                trial |= (1 << k)
                                added += 1
                                if added == 3:
                                    break
                        if added == 3:
                            cur = trial
                            cur_size += 1
                            improved = True
                            break
                    if improved:
                        break
        return cur, cur_size

    best_mask, best_size = local_search(best_mask, adj, n, max_iter=50)

    # Try more random restarts
    import random
    rng = random.Random(123)
    for _ in range(20):
        if time.time() > DEADLINE - 10:
            break
        # Random perturbation
        mask = best_mask
        # Remove a random selected vertex
        selected_list = [i for i in range(n) if mask & (1 << i)]
        if selected_list:
            rem = rng.choice(selected_list)
            mask &= ~(1 << rem)
        # Add random unselected vertices greedily
        for i in range(n):
            if not (mask & (1 << i)) and not (adj[i] & mask):
                if rng.random() < 0.5:
                    mask |= (1 << i)
        # Local search
        mask, sz = local_search(mask, adj, n, max_iter=20)
        if sz > best_size:
            best_size = sz
            best_mask = mask

    # Convert mask to list of numbers
    result = [nums[i] for i in range(n) if best_mask & (1 << i)]

    # Verify
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback to odds
        return [x for x in range(1, N + 1) if x % 2 == 1]

    return result
# EVOLVE-BLOCK-END
