# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    N = 512
    # Vertices: 1..511, weight = 1/(512 - a)
    # We want max weight independent set in conflict graph
    # Edge between a and b if gcd(a,b) > 1

    # Precompute weights
    weights = [0.0] * (N + 1)
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)

    # Precompute conflict bitsets using Python ints as bitsets
    # Bit i represents vertex i (1-indexed, so bit 0 unused, bits 1..511)
    # For each vertex a, conflict_mask[a] has bits set for all b where gcd(a,b)>1
    conflict_mask = [0] * (N + 1)
    for a in range(1, N):
        mask = 0
        for b in range(1, N):
            if a != b and gcd(a, b) > 1:
                mask |= (1 << b)
        conflict_mask[a] = mask

    # Order vertices by weight descending for branching
    vertices = list(range(1, N))
    vertices.sort(key=lambda a: -weights[a])

    # Precompute for each vertex in sorted order:
    # - its index in the sorted list
    # - its conflict mask
    # - its weight
    v_index = {}
    for i, v in enumerate(vertices):
        v_index[v] = i

    n = len(vertices)
    v_weights = [weights[v] for v in vertices]
    v_conflicts = [conflict_mask[v] for v in vertices]

    # For branch and bound:
    # State: (start_idx, current_set_mask, current_weight)
    # At each step, try to include or exclude vertices[start_idx]
    # If we include it, we must exclude all its conflicts from remaining candidates
    # Upper bound: current_weight + sum of weights of all remaining vertices that
    #   are not conflicted with current_set

    # Better: use a recursive search with bitset of available vertices
    # available is a bitmask of vertices (in sorted order) that can still be chosen

    # Convert conflict masks to be in sorted-order indices
    # For vertex at sorted position i, its conflicts among sorted positions
    sorted_conflicts = [0] * n
    for i, v in enumerate(vertices):
        cm = conflict_mask[v]
        sm = 0
        for j, u in enumerate(vertices):
            if u != v and (cm >> u) & 1:
                sm |= (1 << j)
        sorted_conflicts[i] = sm

    # Upper bound: for a set of available vertices (bitmask), the max additional weight
    # is at most the sum of weights of all available vertices (relaxation: ignore conflicts)
    # We can precompute suffix sums for tighter bounds

    # For a bitmask of available vertices, upper bound = sum of weights of set bits
    # To make this fast, we'll compute it on the fly or use a precomputed table
    # Since n=511, we can't precompute all 2^511 subsets, so compute on the fly

    def bit_count_weight(available_mask):
        """Sum of weights of vertices whose bit is set in available_mask."""
        total = 0.0
        m = available_mask
        while m:
            # Get lowest set bit
            lb = m & (-m)
            idx = lb.bit_length() - 1
            total += v_weights[idx]
            m ^= lb
        return total

    # Better upper bound: greedy relaxation
    # Sort available vertices by weight descending, add them greedily ignoring conflicts
    # This gives a tighter upper bound than just sum of all weights
    def greedy_upper_bound(available_mask):
        """Greedy upper bound: add available vertices in weight order, skipping conflicts."""
        total = 0.0
        m = available_mask
        while m:
            # Find the vertex with highest weight among available
            best_idx = -1
            best_w = -1.0
            mm = m
            while mm:
                lb = mm & (-mm)
                idx = lb.bit_length() - 1
                if v_weights[idx] > best_w:
                    best_w = v_weights[idx]
                    best_idx = idx
                mm ^= lb
            if best_idx < 0:
                break
            total += v_weights[best_idx]
            # Remove this vertex and all its conflicts from available
            m &= ~(1 << best_idx)
            m &= ~sorted_conflicts[best_idx]
        return total

    # Branch and bound
    best_weight = 0.0
    best_set = 0  # bitmask of chosen vertices in sorted order

    # Initial greedy solution for a good lower bound
    def initial_greedy():
        available = (1 << n) - 1  # all vertices available
        chosen = 0
        w = 0.0
        m = available
        while m:
            best_idx = -1
            best_w = -1.0
            mm = m
            while mm:
                lb = mm & (-mm)
                idx = lb.bit_length() - 1
                if v_weights[idx] > best_w:
                    best_w = v_weights[idx]
                    best_idx = idx
                mm ^= lb
            if best_idx < 0:
                break
            w += v_weights[best_idx]
            chosen |= (1 << best_idx)
            m &= ~(1 << best_idx)
            m &= ~sorted_conflicts[best_idx]
        return w, chosen

    best_weight, best_set = initial_greedy()

    # Recursive branch and bound
    def branch_and_bound(available_mask, current_weight, current_set):
        nonlocal best_weight, best_set

        if time.time() - start_time > DEADLINE:
            return

        if available_mask == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = current_set
            return

        # Upper bound check
        ub = current_weight + greedy_upper_bound(available_mask)
        if ub <= best_weight:
            return

        # Find a vertex to branch on: pick the one with highest weight in available
        # Actually, for better pruning, pick a vertex that appears in many conflicts
        # Let's pick the highest weight vertex for simplicity
        m = available_mask
        best_idx = -1
        best_w = -1.0
        while m:
            lb = m & (-m)
            idx = lb.bit_length() - 1
            if v_weights[idx] > best_w:
                best_w = v_weights[idx]
                best_idx = idx
            m ^= lb

        if best_idx < 0:
            return

        # Branch 1: include best_idx
        new_available = available_mask & ~(1 << best_idx)
        new_available &= ~sorted_conflicts[best_idx]
        branch_and_bound(
            new_available,
            current_weight + v_weights[best_idx],
            current_set | (1 << best_idx)
        )

        # Branch 2: exclude best_idx
        branch_and_bound(
            available_mask & ~(1 << best_idx),
            current_weight,
            current_set
        )

    # Start with all vertices available
    full_mask = (1 << n) - 1
    branch_and_bound(full_mask, 0.0, 0)

    # Convert best_set back to actual values
    result = []
    m = best_set
    while m:
        lb = m & (-m)
        idx = lb.bit_length() - 1
        result.append(vertices[idx])
        m ^= lb

    # Sort result for consistency
    result.sort()

    # Verify pairwise coprime
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                # Fallback to greedy if something went wrong
                chosen = []
                for a in range(511, 0, -1):
                    if all(gcd(a, b) == 1 for b in chosen):
                        chosen.append(a)
                return chosen

    return result

# EVOLVE-BLOCK-END
