import time
from math import gcd
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin for 90s limit

    N = 256
    values = list(range(2, N + 1))
    size = len(values)  # 255

    # Precompute pairwise conflict: i and j are incompatible if lcm(values[i], values[j]) <= 256
    conflict = [0] * size
    for i in range(size):
        vi = values[i]
        for j in range(i + 1, size):
            vj = values[j]
            g = gcd(vi, vj)
            if vi * vj // g <= 256:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)

    # Reciprocal weights as Fractions for exact scoring
    w_frac = [Fraction(1, values[i]) for i in range(size)]
    # Float weights for fast bound computation
    w_float = [1.0 / values[i] for i in range(size)]

    # Sort indices by weight descending (small values first)
    order = sorted(range(size), key=lambda i: w_float[i], reverse=True)
    # Map from original index to position in order
    pos = [0] * size
    for p, i in enumerate(order):
        pos[i] = p

    # Build conflict in "ordered" representation: for each position p, 
    # mask of later positions that conflict
    # Actually, let's use a different approach: work with original indices but 
    # branch in order of weight

    def score_set(indices):
        """Exact score of a set of indices."""
        s = Fraction(0, 1)
        for i in indices:
            s += w_frac[i]
        return s

    # Greedy heuristic to get initial solution
    def greedy(order_seq):
        S = set()
        for i in order_seq:
            if all(gcd(values[i], values[j]) > 256 or values[i] * values[j] // gcd(values[i], values[j]) > 256 for j in S):
                S.add(i)
        return S

    # Fast greedy using bitmasks
    def greedy_fast(order_seq):
        S_mask = 0
        S_set = set()
        for i in order_seq:
            if not (conflict[i] & S_mask):
                S_mask |= (1 << i)
                S_set.add(i)
        return S_set

    # Get initial solutions
    init1 = greedy_fast(order)
    init2 = greedy_fast(list(reversed(order)))
    
    # Also try a few random orders
    import random
    random.seed(12345)
    best_init = init1
    best_init_score = score_set(init1)
    
    for init in [init1, init2]:
        s = score_set(init)
        if s > best_init_score:
            best_init_score = s
            best_init = init
    
    for _ in range(30):
        shuffled = list(range(size))
        random.shuffle(shuffled)
        init = greedy_fast(shuffled)
        s = score_set(init)
        if s > best_init_score:
            best_init_score = s
            best_init = init

    # Local search to improve
    def local_search(S_set, deadline):
        S_mask = 0
        for i in S_set:
            S_mask |= (1 << i)
        in_set = [False] * size
        for i in S_set:
            in_set[i] = True
        current_score = sum(w_float[i] for i in S_set)

        improved = True
        while improved:
            improved = False
            if time.time() - start_time > deadline:
                break

            # Try additions
            for v in range(size):
                if not in_set[v] and not (conflict[v] & S_mask):
                    S_mask |= (1 << v)
                    in_set[v] = True
                    S_set.add(v)
                    current_score += w_float[v]
                    improved = True

            # Try 1-swap
            for u in list(S_set):
                if not in_set[u]:
                    continue
                S_without_u = S_mask & ~(1 << u)
                for v in order:
                    if w_float[v] <= w_float[u]:
                        break
                    if in_set[v]:
                        continue
                    if conflict[v] & S_without_u:
                        continue
                    # Do the swap
                    S_mask = S_without_u | (1 << v)
                    in_set[u] = False
                    in_set[v] = True
                    S_set.remove(u)
                    S_set.add(v)
                    current_score += w_float[v] - w_float[u]
                    improved = True
                    break

        return S_set

    best_set = local_search(best_init, timeout - 5)
    best_score = score_set(best_set)

    # Branch and bound for exact optimization
    # We branch on vertices in order of weight (descending)
    # State: (available_mask, current_score, current_set)
    # available_mask: vertices not yet decided
    # We use the standard MWIS branch and bound

    # For efficiency, we'll work with the "order" permutation
    # Let's define: at each step, we pick the highest-weight vertex from available
    # and either include it (remove its neighbors) or exclude it

    # Precompute: for each vertex, which vertices in the "order" come after it
    # and are NOT its neighbors (i.e., can still be considered)
    
    # Actually, let's use a cleaner formulation:
    # We process vertices in order[0], order[1], ..., order[size-1]
    # At step k, we decide about order[k]
    # available: bitmask of vertices (in original indexing) that are still available
    # When we include vertex v, we remove v and all its neighbors from available
    # When we exclude v, we just remove v from available

    # Upper bound: sum of weights of all available vertices (simple bound)
    # Better bound: for each available vertex, it can only be included if we don't include its neighbors
    # Simple fractional bound: just sum all available weights (this is valid since 
    # any independent set is a subset of available)

    # Let's use a tighter bound: process available vertices in weight order,
    # greedily build a bound by including a vertex and removing its neighbors
    # This gives a valid upper bound (it's the weight of a maximal independent set 
    # in the available subgraph, which is >= any independent set)
    
    def upper_bound(available_mask):
        """Compute an upper bound on the weight of any independent set in available_mask."""
        # Simple: sum of all weights in available
        total = 0.0
        m = available_mask
        i = 0
        while m:
            if m & 1:
                total += w_float[i]
            i += 1
            m >>= 1
        return total

    # Tighter bound using greedy independent set on available subgraph
    def greedy_bound(available_mask):
        """Weight of a greedy independent set in available subgraph (valid upper bound)."""
        total = 0.0
        avail = available_mask
        while avail:
            # Find highest weight vertex in avail
            best = -1
            best_w = -1.0
            m = avail
            i = 0
            while m:
                if m & 1:
                    if w_float[i] > best_w:
                        best_w = w_float[i]
                        best = i
                i += 1
                m >>= 1
            if best < 0:
                break
            total += best_w
            # Remove best and its neighbors
            avail &= ~(1 << best)
            avail &= ~conflict[best]
        return total

    # Branch and bound
    # We'll use iterative deepening or just a priority queue approach
    # For simplicity, use DFS with pruning

    best_bb_score = best_score
    best_bb_set = set(best_set)
    nodes_explored = 0

    def branch_and_bound(available_mask, current_score, current_set, deadline):
        nonlocal best_bb_score, best_bb_set, nodes_explored

        if time.time() - start_time > deadline:
            return

        nodes_explored += 1

        # Upper bound
        ub = greedy_bound(available_mask)
        if current_score + ub <= best_bb_score + 1e-12:
            return

        # If no more vertices, update best
        if available_mask == 0:
            if current_score > best_bb_score:
                best_bb_score = current_score
                best_bb_set = set(current_set)
            return

        # Find the highest-weight vertex in available
        best_v = -1
        best_w = -1.0
        m = available_mask
        i = 0
        while m:
            if m & 1:
                if w_float[i] > best_w:
                    best_w = w_float[i]
                    best_v = i
            i += 1
            m >>= 1

        if best_v < 0:
            return

        # Branch 1: Include best_v
        new_available = available_mask & ~(1 << best_v) & ~conflict[best_v]
        new_score = current_score + w_float[best_v]
        new_set = current_set | {best_v}
        branch_and_bound(new_available, new_score, new_set, deadline)

        # Branch 2: Exclude best_v
        new_available2 = available_mask & ~(1 << best_v)
        branch_and_bound(new_available2, current_score, current_set, deadline)

    # Start B&B with all vertices available
    all_mask = (1 << size) - 1
    # Remove vertices that are in the current best? No, we need to search the full space.
    # Actually, we should start with all vertices available and current_score = 0
    # But that's a huge search space. Let's use the current best as a starting point.
    
    # Better: start B&B from scratch but with the best_score as initial bound
    # This way, the bound prunes more effectively
    
    branch_and_bound(all_mask, 0.0, set(), timeout - 1)

    # Final result
    if best_bb_score > best_score:
        best_set = best_bb_set
        best_score = best_bb_score

    result = [values[i] for i in sorted(best_set)]
    return result if result else [2]

# EVOLVE-BLOCK-END
