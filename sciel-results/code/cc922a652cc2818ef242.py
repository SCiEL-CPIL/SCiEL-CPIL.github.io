import time
import math

def construct():
    """
    Returns a list of distinct ints from 1..256, pairwise coprime, maximizing sum.
    Uses branch-and-bound Maximum Weight Independent Set on the conflict graph.
    """
    N = 256
    deadline = time.time() + 80  # safety margin before 90s kill

    # Build conflict graph: edge between i,j if gcd(i,j) > 1
    # Vertices: 1..N, indexed 0..N-1 (value = index + 1)
    # 1 is coprime to everything, so no conflicts for vertex 0
    conflict = [0] * N  # bitmask of conflicts for each vertex
    for i in range(N):
        vi = i + 1
        for j in range(i + 1, N):
            vj = j + 1
            if math.gcd(vi, vj) > 1:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)

    weights = list(range(1, N + 1))

    # Greedy initial solution: pick vertices in descending weight order
    # if they don't conflict with already-picked vertices
    picked = 0
    greedy_sum = 0
    for i in range(N - 1, -1, -1):
        if (conflict[i] & picked) == 0:
            picked |= (1 << i)
            greedy_sum += weights[i]

    best_sum = greedy_sum
    best_mask = picked

    # Precompute degree for each vertex (number of conflicts)
    degrees = [bin(conflict[i]).count('1') for i in range(N)]

    # Greedy coloring upper bound for a subgraph (cand_mask)
    def color_bound(cand_mask):
        """
        Greedy sequential coloring of the subgraph induced by cand_mask.
        Returns list of color class weight sums.
        Upper bound = sum of color class weights.
        """
        remaining = cand_mask
        color_classes = []
        while remaining:
            # Pick vertex with highest degree in remaining subgraph
            # (or highest weight as tiebreaker)
            best_v = -1
            best_deg = -1
            best_w = -1
            r = remaining
            while r:
                lsb = r & (-r)
                v = lsb.bit_length() - 1
                deg_in_rem = bin(conflict[v] & remaining).count('1')
                w = weights[v]
                if deg_in_rem > best_deg or (deg_in_rem == best_deg and w > best_w):
                    best_deg = deg_in_rem
                    best_w = w
                    best_v = v
                r ^= lsb

            # Greedily assign this vertex's color class:
            # start with best_v, then add non-conflicting vertices
            color_class_mask = 0
            color_class_weight = 0
            available = remaining
            while available:
                lsb = available & (-available)
                v = lsb.bit_length() - 1
                if (conflict[v] & color_class_mask) == 0:
                    color_class_mask |= lsb
                    color_class_weight += weights[v]
                    available ^= lsb
                else:
                    available ^= lsb

            color_classes.append(color_class_weight)
            remaining &= ~color_class_mask

        return color_classes

    # Branch and bound
    def bb(cand_mask, cur_sum, cur_mask):
        nonlocal best_sum, best_mask

        if time.time() > deadline:
            return

        if cand_mask == 0:
            if cur_sum > best_sum:
                best_sum = cur_sum
                best_mask = cur_mask
            return

        # Upper bound via greedy coloring
        color_classes = color_bound(cand_mask)
        ub = cur_sum + sum(color_classes)
        if ub <= best_sum:
            return

        # If we can't improve, prune
        if time.time() > deadline:
            return

        # Choose branching vertex: highest degree in cand_mask
        # (ties broken by highest weight)
        best_v = -1
        best_deg = -1
        best_w = -1
        r = cand_mask
        while r:
            lsb = r & (-r)
            v = lsb.bit_length() - 1
            deg = bin(conflict[v] & cand_mask).count('1')
            w = weights[v]
            if deg > best_deg or (deg == best_deg and w > best_w):
                best_deg = deg
                best_w = w
                best_v = v
            r ^= lsb

        v_bit = 1 << best_v
        v_w = weights[best_v]

        # Branch 1: include best_v
        # Remove best_v and all its neighbors from cand
        new_cand = cand_mask & ~(v_bit | conflict[best_v])
        bb(new_cand, cur_sum + v_w, cur_mask | v_bit)

        if time.time() > deadline:
            return

        # Branch 2: exclude best_v
        new_cand2 = cand_mask & ~v_bit
        bb(new_cand2, cur_sum, cur_mask)

    bb((1 << N) - 1, 0, 0)

    # Extract result from best_mask
    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(i + 1)

    # Ensure 1 is included (it should be, but verify)
    if 1 not in result:
        result.append(1)

    return result
