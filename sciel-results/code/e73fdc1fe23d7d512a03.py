import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer for safety

    best_edges = None
    best_count = float('inf')
    best_n = None

    # We'll work with n=9 (K_9^5 has 126 edges, which is non-2-colorable by pigeonhole)
    # n=9 is the smallest n where K_n^5 is non-2-colorable (since ceil(9/2)=5)
    # For n=8, we can color 4 red, 4 blue, and no 5-set is monochromatic.
    # So n must be at least 9.
    # The task says 5 <= n <= 26.
    # We focus on n=9 first. If we find a very sparse one, we might try n=10?
    # But n=9 has fewer vertices, so the space of colorings is smaller (2^9=512),
    # making the set-cover formulation very fast.

    # Let's try n=9, then n=10, then n=11 if time permits.
    # Actually, smaller n is better for the bitmask approach.
    # n=9: 2^9 = 512 colorings.
    # n=10: 2^10 = 1024 colorings.
    # n=11: 2^11 = 2048 colorings.
    # The number of edges in K_n^5:
    # n=9: C(9,5) = 126
    # n=10: C(10,5) = 252
    # n=11: C(11,5) = 462

    # We will use a "Set Cover" style reduction.
    # We want to find a subset of edges E' such that every coloring has a monochromatic edge in E'.
    # Equivalently, for every coloring c, E' intersects M_c (the set of monochromatic edges under c).
    # This is a hitting set problem.
    # We can solve it greedily or via local search.

    # Since n is small, we can precompute M_c for all c.
    # M_c is a bitmask of edges.
    # Then we want to find a small set of edges S such that for all c, S & M_c != 0.

    # Algorithm:
    # 1. Start with S = all edges.
    # 2. While S is not minimal (i.e., we can remove an edge and still hit all M_c):
    #    - Find an edge e in S such that S \ {e} still hits all M_c.
    #    - Remove e.
    # This is the standard greedy reduction for hitting sets, but we can speed it up.
    # Actually, the greedy reduction described above is O(|S| * 2^n * |E|) which might be slow.
    # A faster way:
    # For each edge e, check if there exists a coloring c such that M_c & S == {e} (i.e., e is the ONLY edge in S that is monochromatic under c).
    # If such a c exists, then e is "essential" and cannot be removed.
    # If no such c exists, e can be removed.
    # So we can iterate:
    #   For each e in S:
    #      Check if e is essential.
    #      If not, remove e.
    # This takes O(|S| * 2^n) per pass.
    # Since |S| <= 126 and 2^9 = 512, this is very fast (126 * 512 ~ 65k operations).
    # We can do this until no more edges can be removed.
    # To avoid getting stuck in a local minimum (where no single edge can be removed, but a combination could),
    # we can use a randomized approach:
    #   - Shuffle edges.
    #   - Try to remove each edge in order.
    #   - If stuck, restart with a different random order or a different starting set.
    #   - Or, use simulated annealing to "add back" edges and remove others to find a smaller set.

    # Let's implement the "essential edge" check.
    # Precompute M_c for all c.
    # For a given S (bitmask), an edge e is essential if there exists c such that (M_c & S) == (1 << e).
    # We can precompute for each c the set of edges in M_c.
    # Then for a given S, we can check each c:
    #   if (M_c & S) has exactly one bit set, and that bit is e, then e is essential.
    # We can do this for all e in S.

    # Let's optimize the check:
    # For each c, let T_c = M_c & S.
    # If T_c has exactly one bit set, say e, then e is essential (for this c).
    # So we can iterate over all c, compute T_c, and if T_c is a power of 2, mark that edge as essential.
    # Then we can remove all non-essential edges.
    # This takes O(2^n) per pass, which is 512 operations. Very fast.

    # We can do this in a loop:
    #   S = all edges
    #   While True:
    #       essential = {}
    #       for c in range(2^n):
    #           T_c = M_c & S
    #           if T_c and (T_c & (T_c - 1)) == 0:  # T_c is a power of 2
    #               e = T_c.bit_length() - 1
    #               essential[e] = True
    #       # Remove all edges in S that are not in essential
    #       new_S = 0
    #       for e in range(len(all_edges)):
    #           if (S >> e) & 1:
    #               if e in essential:
    #                   new_S |= (1 << e)
    #       if new_S == S:
    #           break
    #       S = new_S
    #       # Also, we should check if S is still a valid hitting set.
    #       # It should be, because we only removed edges that were not essential.
    #       # But just in case, we can verify.

    # This algorithm finds a "minimal" hitting set (in the sense that no single edge can be removed).
    # It might not be the minimum hitting set, but it's a good start.
    # To find a smaller one, we can use local search:
    #   - Start with the minimal hitting set S.
    #   - Randomly remove an edge e from S.
    #   - Check if S \ {e} is still a valid hitting set.
    #   - If not, add back e.
    #   - If yes, keep it removed.
    #   - Repeat.
    #   - Also, we can try to "swap" edges: remove e and add f, if S \ {e} U {f} is valid and |S| is reduced (which it won't be, unless we remove more).
    #   - Actually, the goal is to minimize |S|.
    #   - So we can use a "steepest descent" approach:
    #       - For each edge e in S, check if S \ {e} is valid.
    #       - If yes, remove e.
    #       - If no, try to find an edge f not in S such that S \ {e} U {f} is valid and |S| is reduced (impossible) or at least the same (no gain).
    #       - So steepest descent only works if we can remove an edge.
    #   - To escape local minima, we can use simulated annealing:
    #       - With some probability, add a random edge f not in S.
    #       - Then try to remove edges.
    #       - Accept the new state if it's better (smaller |S|) or with some probability if it's worse.

    # Given the small size, we can afford to run many iterations of the "essential edge" reduction with different random orderings or perturbations.

    # Let's implement the core reduction first.

    def reduce_hitting_set(n, all_edges, M_c):
        # M_c is a list of bitmasks, M_c[c] is the bitmask of monochromatic edges under coloring c.
        S = (1 << len(all_edges)) - 1  # All edges
        # Verify S is a valid hitting set
        # (It should be, since it's the full graph)
        for c in range(1 << n):
            if (M_c[c] & S) == 0:
                return None  # Not a valid hitting set

        # Iteratively remove non-essential edges
        while True:
            essential = [False] * len(all_edges)
            for c in range(1 << n):
                T_c = M_c[c] & S
                if T_c and (T_c & (T_c - 1)) == 0:  # T_c is a power of 2
                    e = T_c.bit_length() - 1
                    essential[e] = True

            new_S = 0
            for e in range(len(all_edges)):
                if (S >> e) & 1:
                    if essential[e]:
                        new_S |= (1 << e)

            if new_S == S:
                break
            S = new_S

            # Verify S is still a valid hitting set
            for c in range(1 << n):
                if (M_c[c] & S) == 0:
                    return None

        return S

    # Now, let's try different n values.
    for n in [9, 10, 11]:
        if time.time() > deadline:
            break

        # Generate all edges
        all_edges = [list(e) for e in combinations(range(n), 5)]
        num_edges = len(all_edges)

        # Precompute M_c for all colorings
        M_c = [0] * (1 << n)
        edge_masks = []
        for e in all_edges:
            mask_e = 0
            for v in e:
                mask_e |= (1 << v)
            edge_masks.append(mask_e)

        for c in range(1 << n):
            bm = 0
            for i in range(num_edges):
                mask_e = edge_masks[i]
                inter = c & mask_e
                if inter == 0 or inter == mask_e:
                    bm |= (1 << i)
            M_c[c] = bm

        # Quick check: if any coloring has no monochromatic edges, skip
        if any(bm == 0 for bm in M_c):
            continue

        # Reduce the hitting set
        S = reduce_hitting_set(n, all_edges, M_c)
        if S is None:
            continue

        count = bin(S).count('1')
        if count < best_count:
            best_count = count
            best_edges = [all_edges[i] for i in range(num_edges) if (S >> i) & 1]
            best_n = n

        # If we found a very small set, we can stop early
        if best_count <= 51:
            break

    if best_edges is None:
        # Fallback: return the full graph for n=9
        n = 9
        edges = [list(e) for e in combinations(range(n), 5)]
        return {'n': n, 'edges': edges}

    return {'n': best_n, 'edges': best_edges}
