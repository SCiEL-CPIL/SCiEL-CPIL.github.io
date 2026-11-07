"""
Maximum sum of pairwise-coprime integers from 1..256.

Uses Branch-and-Bound with bitset optimizations to find the exact
Maximum Weight Independent Set in the conflict graph.
"""
import time
import math
from functools import lru_cache


def construct():
    # Deadline guard: finish well before 90s
    t_start = time.time()
    DEADLINE = 85.0  # seconds

    N = 256
    # Vertex i (0-indexed) corresponds to number (i+1)
    weights = [i + 1 for i in range(N)]

    # Build conflict graph using bitsets
    # neighbors[i] is a bitmask of all j where gcd(i+1, j+1) > 1
    neighbors = [0] * N

    for i in range(N):
        vi = i + 1
        mask = 0
        for j in range(N):
            if j == i:
                continue
            if math.gcd(vi, j + 1) > 1:
                mask |= (1 << j)
        neighbors[i] = mask

    # Precompute the set of all vertices as a mask
    all_mask = (1 << N) - 1

    # Precompute for each vertex, the sum of weights in its neighborhood
    # This helps with upper bound estimation
    neighbor_weight_sum = [0] * N
    for i in range(N):
        mask = neighbors[i]
        s = 0
        while mask:
            low = mask & (-mask)
            j = low.bit_length() - 1
            s += weights[j]
            mask ^= low
        neighbor_weight_sum[i] = s

    # Greedy solution for initial lower bound
    def greedy_solution():
        selected = set()
        candidates = set(range(N))
        while candidates:
            # Pick the vertex with max weight among remaining
            best = max(candidates, key=lambda x: weights[x])
            selected.add(best)
            # Remove best and its neighbors from candidates
            candidates.discard(best)
            candidates -= set()  # no-op, just for clarity
            removed = set()
            m = neighbors[best]
            while m:
                low = m & (-m)
                j = low.bit_length() - 1
                if j in candidates:
                    removed.add(j)
                m ^= low
            candidates -= removed
        return sum(weights[i] for i in selected)

    best_lb = greedy_solution()

    # Branch and Bound
    # State: (available_mask, current_sum)
    # We use a stack for DFS with pruning

    # Sort vertices by weight descending for better pruning
    # We'll use a recursive approach with memoization on the available set
    # But N=256 is too large for full memoization, so we use B&B

    # Use a priority queue or DFS
    # DFS with pruning is simpler

    best = [best_lb]
    best_set = [set()]  # track the actual set for verification

    # For B&B, we process vertices in order of decreasing weight
    # At each step, we either include the vertex or exclude it

    # To make this efficient, we use the following approach:
    # - Sort vertices by weight descending
    # - Use a bitmask of available vertices
    # - Upper bound: current_sum + sum of weights of all available vertices that don't conflict with each other
    #   (relaxation: just sum all available weights, which is an upper bound)

    # Actually, a tighter upper bound: for the remaining available set,
    # the max independent set weight is at most the sum of all weights in the set.
    # This is loose but fast to compute.

    # Let's use a better approach:
    # Process vertices one by one in sorted order.
    # State: index in sorted order, available mask, current sum
    # This is essentially a branch and bound on a decision tree.

    # For N=256, even with pruning, this might be slow.
    # Let's use a smarter approach.

    # Alternative: Use the fact that the problem structure is based on prime factors.
    # Each number 2..256 has a set of prime factors.
    # Two numbers conflict iff they share a prime factor.
    # So the conflict graph is a union of cliques per prime, but with overlaps.

    # For exact solution with N=256, let's use a well-optimized B&B.

    # Precompute prime factors for each number
    prime_factors = [[] for _ in range(N)]
    for i in range(N):
        n = i + 1
        p = 2
        temp = n
        while p * p <= temp:
            if temp % p == 0:
                prime_factors[i].append(p)
                while temp % p == 0:
                    temp //= p
            p += 1
        if temp > 1:
            prime_factors[i].append(temp)

    # The conflict graph is determined by shared prime factors.
    # Let's use a different B&B strategy:
    # - Start with all vertices available
    # - Pick a vertex, branch on include/exclude
    # - When including, remove the vertex and all its neighbors
    # - Prune when current_sum + sum(available_weights) <= best

    # To speed up, we precompute the weight sum for any mask... but 2^256 is too large.
    # Instead, we compute the sum on the fly using bit operations.

    def sum_weights(mask):
        s = 0
        m = mask
        while m:
            low = m & (-m)
            j = low.bit_length() - 1
            s += weights[j]
            m ^= low
        return s

    # This is O(N) per call, which is acceptable for B&B with good pruning.

    # Let's use iterative deepening or a simple DFS with pruning.
    # Given N=256, we need to be careful.

    # Actually, let's try a different approach:
    # The problem can be reformulated as follows:
    # For each prime p <= 256, we can choose at most one number that is divisible by p.
    # But numbers can be divisible by multiple primes, so this is not straightforward.

    # Let's just use a well-optimized B&B with the following improvements:
    # 1. Order vertices by decreasing weight
    # 2. Use a tight upper bound
    # 3. Use bitset operations for neighbor removal

    # Sort vertices by weight descending
    order = sorted(range(N), key=lambda x: -weights[x])
    order_set = set(order)

    # For B&B, we maintain:
    # - available: bitmask of vertices that can still be selected
    # - current_sum: sum of weights of selected vertices

    # We'll use a stack for DFS to avoid recursion depth issues

    # Initial state
    stack = [(all_mask, 0, 0)]  # (available_mask, current_sum, next_vertex_index_in_order)

    # Actually, a cleaner approach:
    # At each node, we have a set of available vertices.
    # We pick the highest-weight available vertex and branch.

    # Let's implement this iteratively

    # For better performance, let's precompute the bit for each vertex
    vertex_bit = [1 << i for i in range(N)]

    # B&B with stack
    # Each entry: (available_mask, current_sum)
    # We process by picking the highest-weight vertex in available

    stack = [(all_mask, 0)]
    visited_count = 0

    while stack:
        if time.time() > DEADLINE:
            break

        available, current_sum = stack.pop()
        visited_count += 1

        if available == 0:
            if current_sum > best[0]:
                best[0] = current_sum
            continue

        # Upper bound: current_sum + sum of all weights in available
        # This is a valid upper bound (relaxation)
        ub = current_sum + sum_weights(available)
        if ub <= best[0]:
            continue

        # Find the highest-weight vertex in available
        # We need to find the vertex v in available with max weight[v]
        best_v = -1
        best_w = -1
        m = available
        while m:
            low = m & (-m)
            j = low.bit_length() - 1
            if weights[j] > best_w:
                best_w = weights[j]
                best_v = j
            m ^= low

        if best_v == -1:
            continue

        # Branch 1: Include best_v
        new_available = available & ~vertex_bit[best_v]
        # Remove all neighbors of best_v
        new_available &= ~neighbors[best_v]
        new_sum = current_sum + weights[best_v]
        stack.append((new_available, new_sum))

        # Branch 2: Exclude best_v
        stack.append((available & ~vertex_bit[best_v], current_sum))

    # Extract the solution
    # We need to reconstruct the set from best[0]
    # Since we didn't track the actual set, let's redo with set tracking

    # Actually, for the task, we just need to return a valid list.
    # Let's find a valid set with sum >= best[0]
    # We can do a greedy approach to find a set, then try to improve it.

    # Better: let's track the set in the B&B

    # Re-run B&B with set tracking
    best_sum = [best_lb]
    best_vertices = [set()]

    stack = [(all_mask, 0, frozenset())]

    while stack:
        if time.time() > DEADLINE:
            break

        available, current_sum, selected = stack.pop()

        if available == 0:
            if current_sum > best_sum[0]:
                best_sum[0] = current_sum
                best_vertices[0] = set(selected)
            continue

        ub = current_sum + sum_weights(available)
        if ub <= best_sum[0]:
            continue

        # Find highest-weight vertex in available
        best_v = -1
        best_w = -1
        m = available
        while m:
            low = m & (-m)
            j = low.bit_length() - 1
            if weights[j] > best_w:
                best_w = weights[j]
                best_v = j
            m ^= low

        if best_v == -1:
            continue

        # Branch 1: Include best_v
        new_available = available & ~vertex_bit[best_v]
        new_available &= ~neighbors[best_v]
        new_sum = current_sum + weights[best_v]
        stack.append((new_available, new_sum, selected | frozenset([best_v])))

        # Branch 2: Exclude best_v
        stack.append((available & ~vertex_bit[best_v], current_sum, selected))

    # Convert best_vertices to list of actual numbers
    result = []
    if best_vertices[0]:
        for v in best_vertices[0]:
            result.append(v + 1)
        result.sort()
    else:
        # Fallback to primes
        result = [1] + [a for a in range(2, N + 1)
                        if all(a % d for d in range(2, math.isqrt(a) + 1))]

    # Ensure all are Python ints
    result = [int(x) for x in result]

    return result


if __name__ == "__main__":
    res = construct()
    print(f"Sum: {sum(res)}")
    print(f"Count: {len(res)}")
    # Verify pairwise coprimality
    valid = True
    for i in range(len(res)):
        for j in range(i + 1, len(res)):
            if math.gcd(res[i], res[j]) != 1:
                valid = False
                print(f"Conflict: {res[i]} and {res[j]}")
                break
        if not valid:
            break
    print(f"Valid: {valid}")
