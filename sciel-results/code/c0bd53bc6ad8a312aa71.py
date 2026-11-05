import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s limit

    N = 180
    all_elements = set(range(1, N + 1))

    # Precompute all forbidden triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    # For each a, find all (b, c) pairs where b, c > a
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
                # Ensure a < b < c
                if b < c:
                    triples.append((a, b, c))
                    triples_by_a[a].append((b, c))
                else:
                    triples.append((a, c, b))
                    triples_by_a[a].append((c, b))

    # For each element, compute which triples it participates in
    # An element x is "small" in triples where x is the first element
    # An element x is "large" in triples where x is b or c
    triples_by_element = defaultdict(list)  # element -> list of (triple_idx, role)
    for idx, (a, b, c) in enumerate(triples):
        triples_by_element[a].append((idx, 'small'))
        triples_by_element[b].append((idx, 'large'))
        triples_by_element[c].append((idx, 'large'))

    # Precompute for each pair (x, y), which triples contain both
    # This helps with fast conflict checking
    # For each element x, and potential addition y, check conflicts

    # Greedy initial solution
    def greedy_solution(order):
        S = set()
        S_set = set()
        for x in order:
            # Check if x can be added
            valid = True
            # x as small element
            for b, c in triples_by_a[x]:
                if b in S_set and c in S_set:
                    valid = False
                    break
            if not valid:
                continue
            # x as large element
            for a in S_set:
                if a >= x:
                    continue
                # Check if there exists c such that (a, x, c) or (a, c, x) is a triple
                # 1/a = 1/x + 1/c => c = ax/(x-a)
                den = x - a
                num = a * x
                if num % den != 0:
                    continue
                c = num // den
                if c == a or c == x:
                    continue
                if c in S_set:
                    valid = False
                    break
            if valid:
                S.add(x)
                S_set.add(x)
        return S

    # Try multiple greedy orderings
    best_S = set()
    best_size = 0

    # Order 1: descending (prefer large numbers)
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

    # Order 3: random
    for _ in range(20):
        order3 = list(range(1, N + 1))
        random.shuffle(order3)
        S3 = greedy_solution(order3)
        if len(S3) > best_size:
            best_S = set(S3)
            best_size = len(S3)

    # Local search: try to add elements
    S = set(best_S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for y in range(1, N + 1):
            if y not in S:
                # Check if y can be added
                valid = True
                # y as small element
                for b, c in triples_by_a[y]:
                    if b in S and c in S:
                        valid = False
                        break
                if not valid:
                    continue
                # y as large element
                for a in S:
                    if a >= y:
                        continue
                    den = y - a
                    num = a * y
                    if num % den != 0:
                        continue
                    c = num // den
                    if c == a or c == y:
                        continue
                    if c in S:
                        valid = False
                        break
                if valid:
                    S.add(y)
                    improved = True

    if len(S) > best_size:
        best_S = set(S)
        best_size = len(S)

    # Branch and bound for improvement
    # Use the current best as a lower bound
    lower_bound = best_size

    # Build conflict structure for B&B
    # For each element, list of elements it conflicts with in some triple
    # Two elements x, y conflict if there exists a triple containing both
    # Actually, we need: if we have a partial set P, and we add x, 
    # then no triple can be completed with x and elements from P

    # For B&B, we process elements in some order
    # At each step, decide to include or exclude the next element
    # Prune if |P| + |remaining| <= lower_bound

    # Order elements by "conflict degree" (number of triples they're in)
    conflict_degree = {}
    for x in range(1, N + 1):
        conflict_degree[x] = len(triples_by_element[x])

    # Sort by conflict degree descending (more constrained first)
    element_order = sorted(range(1, N + 1), key=lambda x: -conflict_degree[x])

    # B&B state
    current_set = set()
    best_found = set(best_S)
    best_found_size = best_size
    nodes_explored = 0
    max_nodes = 500000

    def can_add_to_set(x, S_set):
        # Check if x can be added to S_set without creating a triple
        # x as small
        for b, c in triples_by_a[x]:
            if b in S_set and c in S_set:
                return False
        # x as large
        for a in S_set:
            if a >= x:
                continue
            den = x - a
            num = a * x
            if num % den != 0:
                continue
            c = num // den
            if c == a or c == x:
                continue
            if c in S_set:
                return False
        return True

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

        # Upper bound: current size + remaining elements
        if len(current_set) + remaining_count <= lower_bound:
            return

        if nodes_explored > max_nodes:
            return

        x = element_order[pos]
        nodes_explored += 1

        # Try including x
        if can_add_to_set(x, current_set):
            current_set.add(x)
            branch_and_bound(pos + 1, current_set, remaining_count - 1)
            current_set.discard(x)

        # Try excluding x
        branch_and_bound(pos + 1, current_set, remaining_count - 1)

    # Run B&B with time limit
    if time.time() < deadline - 10:
        branch_and_bound(0, set(), N)

    # Final local search on best found
    S = set(best_found)
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for y in range(1, N + 1):
            if y not in S:
                if can_add_to_set(y, S):
                    S.add(y)
                    improved = True

    if len(S) > len(best_found):
        best_found = set(S)

    return list(best_found)
