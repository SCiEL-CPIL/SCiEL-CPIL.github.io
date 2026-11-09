import time
import random
from math import gcd

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill

    def elapsed():
        return time.time() - start

    N = 256
    # Numbers 1..256, index 0..255
    # Conflict: (a+b) | (a*b) for distinct a,b
    # Structural test: let g=gcd(a,b), u=a/g, v=b/g. Then (a+b)|(a*b) iff (u+v)|g.
    def inc(a, b):
        g = gcd(a, b)
        u = a // g
        v = b // g
        return (u + v) % g == 0

    # Precompute conflict bitmasks.
    # conflicts[i] is a bitmask of all j != i such that inc(i+1, j+1) is True.
    # Using Python ints as arbitrary-precision bitsets.
    conflicts = [0] * N
    for i in range(N):
        a = i + 1
        mask = 0
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if inc(a, b):
                mask |= (1 << j)
        conflicts[i] = mask

    # Degrees
    degrees = [bin(conflicts[i]).count('1') for i in range(N)]

    # ------------------------------------------------------------------
    # Key structural insight:
    # All odd numbers in [1,256] are mutually compatible?
    # Let's verify: for odd a,b, g=gcd(a,b) is odd, u=a/g odd, v=b/g odd.
    # u+v is even. (u+v)|g means an even number divides an odd number -> impossible.
    # So all odds are mutually non-conflicting. There are 128 odds.
    # ------------------------------------------------------------------
    odds = [i for i in range(N) if (i + 1) % 2 == 1]  # indices of odd numbers
    evens = [i for i in range(N) if (i + 1) % 2 == 0]  # indices of even numbers

    # Which evens are compatible with ALL odds?
    odds_ok = []
    for e in evens:
        ok = True
        for o in odds:
            if (conflicts[e] >> o) & 1:
                ok = False
                break
        if ok:
            odds_ok.append(e)

    # ------------------------------------------------------------------
    # Now we need a maximum independent set among odds_ok,
    # where conflicts are only among evens (odds have no conflicts with each other
    # and odds_ok evens have no conflicts with odds by construction).
    # The final set = all odds + the chosen evens.
    # ------------------------------------------------------------------
    n_ev = len(odds_ok)
    # Build conflict bitmask among odds_ok indices (0..n_ev-1)
    ev_conflicts = [0] * n_ev
    for i in range(n_ev):
        e = odds_ok[i]
        mask = 0
        for j in range(n_ev):
            if i == j:
                continue
            f = odds_ok[j]
            if (conflicts[e] >> f) & 1:
                mask |= (1 << j)
        ev_conflicts[i] = mask

    ev_degrees = [bin(ev_conflicts[i]).count('1') for i in range(n_ev)]

    # ------------------------------------------------------------------
    # Maximum Independent Set via branch-and-bound with bitsets.
    # This is the core structural change: exact MIS solver.
    # ------------------------------------------------------------------
    def mis_branch_and_bound(conflict_masks, n, time_limit_fn):
        """
        Find maximum independent set in graph defined by conflict_masks.
        conflict_masks[i] is bitmask of neighbors of i.
        Returns (size, set_of_indices).
        """
        if n == 0:
            return 0, frozenset()

        # Order vertices by degree descending for better pruning
        order = sorted(range(n), key=lambda i: -bin(conflict_masks[i]).count('1'))
        # Remap: new index -> old index
        remap = [0] * n
        for new_i, old_i in enumerate(order):
            remap[new_i] = old_i

        # Rebuild conflict masks in new order
        new_masks = [0] * n
        for i in range(n):
            old_i = order[i]
            mask = conflict_masks[old_i]
            new_mask = 0
            for j in range(n):
                if i == j:
                    continue
                old_j = order[j]
                if (mask >> old_j) & 1:
                    new_mask |= (1 << j)
            new_masks[i] = new_mask

        best_size = [0]
        best_set = [frozenset()]

        # Use a recursive branch-and-bound with bitset acceleration.
        # State: (available_mask, current_size, current_set)
        # available_mask: bitmask of vertices still available to add
        # We use a greedy upper bound: the size of a maximal independent set
        # found greedily from available_mask gives an upper bound.

        def greedy_max_is(available):
            """Greedy: repeatedly pick vertex with min degree in available."""
            count = 0
            rem = available
            while rem:
                # Pick vertex with minimum degree within rem
                best_v = -1
                best_deg = n + 1
                v = (rem & (-rem)).bit_length() - 1  # lowest set bit
                # Iterate all set bits
                temp = rem
                while temp:
                    v = (temp & (-temp)).bit_length() - 1
                    deg = bin(new_masks[v] & rem).count('1')
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
                    temp &= temp - 1
                if best_v < 0:
                    break
                count += 1
                # Remove best_v and its neighbors
                rem &= ~(1 << best_v)
                rem &= ~new_masks[best_v]
            return count

        # Upper bound: greedy gives a lower bound on MIS, but we need upper bound.
        # Upper bound: number of vertices in available (trivial).
        # Better upper bound: use a coloring heuristic or just the count.
        # For bitset-based, a good upper bound is the greedy coloring of the
        # complement... but let's use: ub = popcount(available) as trivial,
        # and refine with a simple packing argument.
        # Actually, a standard bound: the size of any independent set <= 
        # the number of vertices we can greedily pack using a maximal matching
        # or just use the trivial bound plus the greedy lower bound for pruning.
        # Let's use: ub = popcount(available). If current_size + ub <= best_size, prune.

        # For better pruning, use a greedy coloring upper bound on the complement graph.
        # But for now, let's use a simpler approach: 
        # ub = number of set bits in available. This is weak but works with good ordering.
        # A stronger bound: repeatedly pick a vertex, remove it and its neighbors,
        # the count is a lower bound on MIS (not upper). 
        # Upper bound via greedy coloring of the subgraph induced by available:
        # The chromatic number of the subgraph >= size of any independent set in complement...
        # This is getting complex. Let's use a practical approach:
        # ub = popcount(available) is fine for small n, and we have n_ev <= 128.
        # With good vertex ordering and the structural property that many evens
        # are compatible, the search should be fast.

        import sys
        sys.setrecursionlimit(10000)

        def popcount(x):
            return bin(x).count('1')

        def solve(available, current_size, current_set):
            if elapsed() > time_limit_fn:
                return

            if available == 0:
                if current_size > best_size[0]:
                    best_size[0] = current_size
                    best_set[0] = frozenset(current_set)
                return

            # Upper bound: trivial
            ub = popcount(available)
            if current_size + ub <= best_size[0]:
                return

            # Choose vertex with maximum degree in available for branching
            # (or minimum degree - for independent set, we branch on include/exclude)
            # Standard: pick a vertex v in available.
            # Branch 1: include v -> remove v and its neighbors from available
            # Branch 2: exclude v -> remove v from available

            # Pick vertex: highest degree in available for better pruning
            temp = available
            best_v = -1
            best_deg = -1
            while temp:
                v = (temp & (-temp)).bit_length() - 1
                deg = popcount(new_masks[v] & available)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                temp &= temp - 1

            if best_v < 0:
                return

            # Branch 1: Include best_v
            new_available = available & ~(1 << best_v) & ~new_masks[best_v]
            new_set = set(current_set)
            new_set.add(best_v)
            solve(new_available, current_size + 1, new_set)

            # Branch 2: Exclude best_v
            new_available = available & ~(1 << best_v)
            solve(new_available, current_size, current_set)

        initial_available = (1 << n) - 1
        solve(initial_available, 0, frozenset())

        # Map back to original indices
        result_set = set()
        for idx in best_set[0]:
            result_set.add(remap[idx])
        return best_size[0], result_set

    # Run branch-and-bound on the even subgraph
    # n_ev is at most 128, but with the structural constraints it should be manageable.
    # Use a time limit for the B&B.
    bb_deadline = deadline - 5.0
    ev_size, ev_indices = mis_branch_and_bound(ev_conflicts, n_ev, bb_deadline)

    # Build final set
    final = set()
    for o in odds:
        final.add(o + 1)
    for idx in ev_indices:
        final.add(odds_ok[idx] + 1)

    # ------------------------------------------------------------------
    # If B&B didn't finish in time or found a small set, try multiple
    # randomized greedy + local search to potentially improve.
    # ------------------------------------------------------------------
    best_final_size = len(final)

    # Local search / simulated annealing on the full set
    def can_add(idx, current_set):
        """Check if index idx (0-based, value idx+1) can be added to current_set (set of 1-based values)."""
        v = idx + 1
        for u in current_set:
            if u == v:
                continue
            if inc(v, u):
                return False
        return True

    def local_search_improve(initial_set):
        """Try to improve the set via additions and swaps."""
        current = set(initial_set)
        improved = True
        while improved and elapsed() < deadline - 2:
            improved = False
            # Try to add any compatible element
            for idx in range(N):
                if idx + 1 in current:
                    continue
                if can_add(idx, current):
                    current.add(idx + 1)
                    improved = True
                    break
            # Try swaps: remove one, add two
            if not improved:
                for remove_val in list(current):
                    for add1 in range(N):
                        if add1 + 1 in current:
                            continue
                        for add2 in range(add1 + 1, N):
                            if add2 + 1 in current:
                                continue
                            # Check if removing remove_val allows adding add1 and add2
                            temp = current - {remove_val}
                            if can_add(add1, temp) and can_add(add2, temp):
                                # Also check add1 and add2 don't conflict
                                if not inc(add1 + 1, add2 + 1):
                                    current = temp | {add1 + 1, add2 + 1}
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break

        return current

    # Try a few randomized starts
    random.seed(42)
    for trial in range(20):
        if elapsed() > deadline - 10:
            break
        # Start from odds + random subset of odds_ok
        random.shuffle(odds_ok)
        rand_set = set()
        for o in odds:
            rand_set.add(o + 1)
        for e in odds_ok:
            if random.random() < 0.5:
                rand_set.add(e + 1)
        # Clean up: remove conflicts
        cleaned = set()
        for v in sorted(rand_set):
            if can_add(v - 1, cleaned):
                cleaned.add(v)
        improved = local_search_improve(cleaned)
        if len(improved) > best_final_size:
            final = improved
            best_final_size = len(final)

    # Final verification
    final_list = sorted(final)
    for i in range(len(final_list)):
        for j in range(i + 1, len(final_list)):
            a, b = final_list[i], final_list[j]
            if (a * b) % (a + b) == 0:
                # Invalid - should not happen
                return list(range(1, 257))  # fallback

    return final_list
