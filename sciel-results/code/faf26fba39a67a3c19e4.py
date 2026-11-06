import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    N = 511
    C = 512

    # Sieve for primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    prime_set = set(primes)

    # Weights: w[a] = 1/(512 - a)
    w = [0.0] * (N + 1)
    for a in range(1, N + 1):
        w[a] = 1.0 / (C - a)

    # Prime factors for each number
    pf = [[] for _ in range(N + 1)]
    for p in primes:
        for multiple in range(p, N + 1, p):
            pf[multiple].append(p)

    # Conflict: two numbers conflict if they share a prime factor
    # Precompute conflict as a set of neighbors for each vertex
    # Use a list of sets for efficiency
    neighbors = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        if a == 1:
            continue
        for b in range(a + 1, N + 1):
            # Check if a and b share a prime factor
            shared = False
            for p in pf[a]:
                if p in pf[b]:
                    shared = True
                    break
            if shared:
                neighbors[a].add(b)
                neighbors[b].add(a)

    def get_score(selected):
        return sum(w[a] for a in selected)

    # Greedy initialization with multiple orderings
    def greedy_init(order):
        S = {1}  # 1 is coprime with everything
        for a in order:
            if a in S:
                continue
            can_add = True
            for b in S:
                if a in neighbors[b]:
                    can_add = False
                    break
            if can_add:
                S.add(a)
        return S

    # Try several greedy orderings
    orders = [
        list(range(N, 0, -1)),  # Descending (high weight first)
        list(range(2, N + 1)),  # Ascending
        sorted(range(1, N + 1), key=lambda x: -w[x]),  # By weight descending
    ]

    best_S = None
    best_score = -1
    for order in orders:
        S_greedy = greedy_init(order)
        score = get_score(S_greedy)
        if score > best_score:
            best_score = score
            best_S = S_greedy.copy()

    # Local search: try to add any vertex that fits
    def can_add(S_set, a):
        for b in S_set:
            if a in neighbors[b]:
                return False
        return True

    # Add all possible vertices
    for a in range(1, N + 1):
        if a not in best_S and can_add(best_S, a):
            best_S.add(a)
            best_score += w[a]

    # Local search: try swaps (remove one, add up to two)
    improved = True
    while improved and time.time() < deadline - 30.0:
        improved = False
        # Try single removals and re-additions
        for x in list(best_S):
            if x == 1:
                continue
            old_score = best_score
            best_S.remove(x)
            best_score -= w[x]
            # Try to add back best vertices
            added = []
            for a in range(N, 0, -1):
                if a not in best_S and can_add(best_S, a):
                    best_S.add(a)
                    best_score += w[a]
                    added.append(a)
            if best_score > old_score + 1e-12:
                improved = True
            else:
                # Revert
                for a in added:
                    best_S.remove(a)
                    best_score -= w[a]
                best_S.add(x)
                best_score += w[x]

    # Branch and bound on the full problem
    # Order vertices by weight descending
    vertices = list(range(1, N + 1))
    vertices.sort(key=lambda a: -w[a])

    # Precompute: for each vertex, its neighbors (conflicts)
    # Upper bound: for a set of remaining vertices, the max additional weight
    # is at most the sum of weights of vertices that don't conflict with selected
    # and are mutually compatible (loose bound)
    # Tighter: use a greedy matching bound on the conflict graph

    best_solution = best_S.copy()
    best_val = best_score

    # B&B with time limit
    # State: (selected_set, excluded_set, remaining_list, current_weight)
    # Use a stack to avoid recursion depth issues

    # For efficiency, represent sets as frozensets or use bitmasks
    # Since N=511, we can use Python sets

    def bnb(selected, excluded, remaining, current_weight):
        nonlocal best_solution, best_val

        if time.time() > deadline - 5.0:
            return

        if not remaining:
            if current_weight > best_val:
                best_val = current_weight
                best_solution = selected.copy()
            return

        # Upper bound: current weight + sum of weights of remaining vertices
        # that don't conflict with selected and aren't excluded
        valid = [v for v in remaining if v not in excluded]
        if not valid:
            if current_weight > best_val:
                best_val = current_weight
                best_solution = selected.copy()
            return

        ub = current_weight + sum(w[v] for v in valid)
        if ub <= best_val:
            return

        # Choose next vertex to branch on
        # Pick the vertex with highest weight among valid
        v = max(valid, key=lambda x: w[x])
        rest = [u for u in remaining if u != v]

        # Option 1: Exclude v
        bnb(selected, excluded | {v}, rest, current_weight)

        # Option 2: Include v (if it doesn't conflict with selected)
        if v not in excluded:
            # Check if v conflicts with any selected
            conflicts = False
            for s in selected:
                if v in neighbors[s]:
                    conflicts = True
                    break
            if not conflicts:
                new_excluded = excluded | {v}
                for u in neighbors[v]:
                    new_excluded.add(u)
                bnb(selected | {v}, new_excluded, rest, current_weight + w[v])

    # Start B&B from the current best solution
    # Actually, let's start from scratch but with a good initial bound
    bnb(set(), set(), vertices, 0.0)

    # Final greedy fill
    final_S = best_solution.copy()
    for a in range(N, 0, -1):
        if a not in final_S and can_add(final_S, a):
            final_S.add(a)

    final_score = get_score(final_S)
    if final_score > best_val:
        best_solution = final_S
        best_val = final_score

    return list(best_solution)
