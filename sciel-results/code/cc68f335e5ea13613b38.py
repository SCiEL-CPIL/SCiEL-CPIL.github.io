"""Rewritten solution: exact maximum independent set on the 'square-sum' graph over [1,256]."""
import time
import math
import sys

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    deadline = start + 75.0  # leave margin before 90s kill
    n = 256

    # Precompute perfect squares up to 2*256 = 512
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1

    # Build adjacency: two numbers conflict if their sum is a perfect square
    # We use 0-indexed internally: vertex i represents number i+1
    adj = [set() for _ in range(n)]
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            if a + b in squares:
                adj[i].add(j)
                adj[j].add(i)
        # Diagonal: a + a = 2a must not be a perfect square
        # If 2*(i+1) is a perfect square, vertex i cannot be selected at all
        if 2 * a in squares:
            # Mark as forbidden
            adj[i].add(-1)  # sentinel for self-conflict

    # Remove forbidden vertices (those where 2a is a perfect square)
    allowed = [i for i in range(n) if -1 not in adj[i]]
    for i in allowed:
        adj[i].discard(-1)

    # Map allowed vertices to compact indices
    idx_map = {v: idx for idx, v in enumerate(allowed)}
    m = len(allowed)
    if m == 0:
        return [1]  # fallback, shouldn't happen

    # Build compact adjacency list with bitmasks for speed
    adj_bits = [0] * m
    for i, v in enumerate(allowed):
        for nb in adj[v]:
            if nb in idx_map:
                adj_bits[i] |= (1 << idx_map[nb])

    # Degree for branching heuristic
    degrees = [bin(adj_bits[i]).count('1') for i in range(m)]

    # Branch and bound for maximum independent set
    best_set = [False] * m  # track best solution
    best_size = 0

    # Use a simple recursive branch and bound
    # State: remaining candidates (as set of indices), current selected set
    # We'll use bitmasks for efficiency

    # Initialize: all vertices are candidates
    full_mask = (1 << m) - 1

    def popcount(x):
        return bin(x).count('1')

    # Greedy initial solution
    def greedy_solution():
        selected = []
        remaining = set(range(m))
        while remaining:
            if time.time() > deadline - 5.0:
                break
            # Pick vertex with minimum degree in remaining subgraph
            best_v = None
            best_deg = float('inf')
            for v in remaining:
                deg = popcount(adj_bits[v] & (full_mask - 1))
                # Only count neighbors that are in remaining
                deg_in_rem = 0
                for nb in range(m):
                    if nb in remaining and (adj_bits[v] >> nb) & 1:
                        deg_in_rem += 1
                if deg_in_rem < best_deg:
                    best_deg = deg_in_rem
                    best_v = v
            if best_v is None:
                break
            selected.append(best_v)
            remaining.discard(best_v)
            # Remove neighbors from remaining
            for nb in range(m):
                if nb in remaining and (adj_bits[best_v] >> nb) & 1:
                    remaining.discard(nb)
        return selected

    # Get initial solution via greedy
    initial = greedy_solution()
    if len(initial) > best_size:
        best_size = len(initial)
        for v in initial:
            best_set[v] = True

    # Branch and bound
    sys.setrecursionlimit(10000)

    def branch_and_bound(candidates_mask, current_mask, depth=0):
        nonlocal best_set, best_size

        if time.time() > deadline - 2.0:
            return

        # Upper bound: current selected + remaining candidates
        upper_bound = popcount(current_mask) + popcount(candidates_mask)
        if upper_bound <= best_size:
            return

        if candidates_mask == 0:
            if popcount(current_mask) > best_size:
                best_size = popcount(current_mask)
                best_set = [False] * m
                for i in range(m):
                    if (current_mask >> i) & 1:
                        best_set[i] = True
            return

        # Choose branching vertex: pick one with maximum degree in candidates
        # to maximize pruning
        best_v = -1
        best_deg = -1
        temp_mask = candidates_mask
        while temp_mask:
            lsb = temp_mask & (-temp_mask)
            v = lsb.bit_length() - 1
            deg = popcount(adj_bits[v] & candidates_mask)
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp_mask ^= lsb

        if best_v == -1:
            return

        # Branch 1: include best_v
        new_current = current_mask | (1 << best_v)
        # Remove best_v and its neighbors from candidates
        new_candidates = candidates_mask & ~(1 << best_v) & ~adj_bits[best_v]
        branch_and_bound(new_candidates, new_current, depth + 1)

        # Branch 2: exclude best_v
        new_candidates = candidates_mask & ~(1 << best_v)
        branch_and_bound(new_candidates, current_mask, depth + 1)

    branch_and_bound(full_mask, 0)

    # Convert back to 1-indexed values
    result = []
    for i in range(m):
        if best_set[i]:
            result.append(allowed[i] + 1)

    if not result:
        result = [1]

    # Verify the result
    square_set = squares
    for idx, a in enumerate(result):
        for b in result[idx:]:
            if a + b in square_set:
                # Invalid, fall back to greedy
                result = [allowed[i] + 1 for i in initial] if initial else [1]
                break
        else:
            continue
        break

    return result

# EVOLVE-BLOCK-END
