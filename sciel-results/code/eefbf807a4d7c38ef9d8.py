import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit

    N = 180
    all_elements = set(range(1, N + 1))

    # Precompute all forbidden triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    triples = []
    triples_by_a = defaultdict(list)
    for a in range(1, N + 1):
        a_sq = a * a
        for u in range(1, int(a_sq ** 0.5) + 1):
            if a_sq % u == 0:
                v = a_sq // u
                if u == v:
                    continue
                b = a + u
                c = a + v
                if b > N or c > N:
                    continue
                if b == c:
                    continue
                if b < c:
                    triples.append((a, b, c))
                    triples_by_a[a].append((b, c))
                else:
                    triples.append((a, c, b))
                    triples_by_a[a].append((c, b))

    # Precompute witness pairs for fast conflict checking
    witness_pairs = defaultdict(set)
    for (a, b, c) in triples:
        witness_pairs[a].add(frozenset((b, c)))
        witness_pairs[b].add(frozenset((a, c)))
        witness_pairs[c].add(frozenset((a, b)))

    witness_list = {}
    for x in range(1, N + 1):
        witness_list[x] = list(witness_pairs.get(x, set()))

    def can_add(x, S_set):
        """Check if x can be added to S_set without creating a forbidden triple."""
        for pair in witness_list[x]:
            it = iter(pair)
            e1 = next(it)
            e2 = next(it)
            if e1 in S_set and e2 in S_set:
                return False
        return True

    def greedy_solution(order):
        S_set = set()
        for x in order:
            if can_add(x, S_set):
                S_set.add(x)
        return S_set

    # Try multiple greedy orderings
    best_S = set()
    best_size = 0

    # Order 1: descending
    order1 = list(range(N, 0, -1))
    S1 = greedy_solution(order1)
    if len(S1) > best_size:
        best_S = set(S1)
        best_size = len(S1)

    # Order 2: ascending
    order2 = list(range(1, N + 1))
    S2 = greedy_solution(order2)
    if len(S2) > best_size:
        best_S = set(S2)
        best_size = len(S2)

    # Order 3: random shuffles
    for _ in range(30):
        order3 = list(range(1, N + 1))
        random.shuffle(order3)
        S3 = greedy_solution(order3)
        if len(S3) > best_size:
            best_S = set(S3)
            best_size = len(S3)

    # Order 4: most constrained first
    conflict_count = {x: len(witness_list[x]) for x in range(1, N + 1)}
    order4 = sorted(range(1, N + 1), key=lambda x: -conflict_count[x])
    S4 = greedy_solution(order4)
    if len(S4) > best_size:
        best_S = set(S4)
        best_size = len(S4)

    # Local search: try to add elements
    S = set(best_S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for y in range(1, N + 1):
            if y not in S:
                if can_add(y, S):
                    S.add(y)
                    improved = True

    if len(S) > best_size:
        best_S = set(S)
        best_size = len(S)

    # Repair-based local search: try swapping out large elements to allow adding small ones
    # Key insight: small numbers have many constraints, so they're often excluded early.
    # By removing a large "tail" element that blocks a small number, we might net gain.
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        S_list = sorted(S, reverse=True)  # Try removing largest first (least constrained)
        for rem in S_list:
            if time.time() > deadline - 20:
                break
            S_temp = set(S)
            S_temp.discard(rem)
            # Try to add all possible elements
            added_count = 0
            for y in range(1, N + 1):
                if y not in S_temp and y != rem:
                    if can_add(y, S_temp):
                        S_temp.add(y)
                        added_count += 1
            # If we can add more than we removed (net gain), accept
            if added_count > 1:  # We removed 1, need to add > 1 for net gain
                S = S_temp
                improved = True
                break

    if len(S) > best_size:
        best_S = set(S)
        best_size = len(S)

    # Branch and bound for improvement
    lower_bound = best_size

    element_order = sorted(range(1, N + 1), key=lambda x: -len(witness_list[x]))

    best_found = set(best_S)
    best_found_size = best_size
    nodes_explored = 0
    max_nodes = 1000000

    def branch_and_bound(pos, current_set, remaining_count):
        nonlocal best_found, best_found_size, nodes_explored, lower_bound

        if time.time() > deadline - 5:
            return

        if pos >= len(element_order):
            if len(current_set) > best_found_size:
                best_found_size = len(current_set)
                best_found = set(current_set)
                lower_bound = best_found_size
            return

        if len(current_set) + remaining_count <= lower_bound:
            return

        if nodes_explored > max_nodes:
            return

        x = element_order[pos]
        nodes_explored += 1

        if can_add(x, current_set):
            current_set.add(x)
            branch_and_bound(pos + 1, current_set, remaining_count - 1)
            current_set.discard(x)

        branch_and_bound(pos + 1, current_set, remaining_count - 1)

    if time.time() < deadline - 10:
        branch_and_bound(0, set(), N)

    # Final local search on best found
    S = set(best_found)
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for y in range(1, N + 1):
            if y not in S:
                if can_add(y, S):
                    S.add(y)
                    improved = True

    if len(S) > len(best_found):
        best_found = set(S)

    return list(best_found)
