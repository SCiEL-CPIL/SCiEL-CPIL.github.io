import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    N = 256

    # Weights: w[i] = 1/i
    w = [Fraction(0)] * (N + 1)
    for i in range(1, N + 1):
        w[i] = Fraction(1, i)

    # Precompute: for each element x, the list of elements y (y > x) that conflict with x
    # i.e., there exists z such that {x, y, z} is a 3-AP
    # We'll store conflict[x] as a list of y > x that conflict with x
    conflict = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for y in range(x + 1, N + 1):
            c = False
            # Case 1: x, y are endpoints, middle is (x+y)/2
            s = x + y
            if s % 2 == 0:
                m = s // 2
                if 1 <= m <= N and m != x and m != y:
                    c = True
            # Case 2: x is middle, y and z are endpoints: z = 2x - y
            if not c:
                z = 2 * x - y
                if 1 <= z <= N and z != x and z != y:
                    c = True
            # Case 3: y is middle, x and z are endpoints: z = 2y - x
            if not c:
                z = 2 * y - x
                if 1 <= z <= N and z != x and z != y:
                    c = True
            if c:
                conflict[x].append(y)

    # Also build reverse: for each y, which x < y conflict with y
    # Actually, for branch and bound, we need to know when we pick x,
    # which remaining elements become blocked.
    # blocked[x] = list of y > x that are blocked if x is selected
    # But we also need to handle the case where y < x is already decided.
    # Let's precompute for each x, the set of all y (y != x) that conflict with x
    conflict_set = [set() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for y in conflict[x]:
            conflict_set[x].add(y)
            conflict_set[y].add(x)

    # Greedy initialization: process in order 1, 2, 3, ... (decreasing weight)
    in_S = [False] * (N + 1)
    S = set()

    for x in range(1, N + 1):
        valid = True
        for a in S:
            if a in conflict_set[x]:
                valid = False
                break
        if valid:
            S.add(x)
            in_S[x] = True

    best_score = sum(w[x] for x in S)
    best_set = set(S)

    # Precompute suffix sums of weights for bounding
    # suffix_sum[i] = w[i] + w[i+1] + ... + w[N]
    suffix_sum = [Fraction(0)] * (N + 2)
    for i in range(N, 0, -1):
        suffix_sum[i] = suffix_sum[i + 1] + w[i]

    # Branch and bound DFS
    # State: (idx, S_set, score, blocked_set)
    # idx: next element to consider (1-indexed)
    # We process elements in order 1, 2, ..., N

    def dfs(idx, S_set, score, blocked_set):
        nonlocal best_score, best_set, deadline, start_time

        if time.time() > deadline:
            return

        # Base case: all elements considered
        if idx > N:
            if score > best_score:
                best_score = score
                best_set = set(S_set)
            return

        # Upper bound: current score + sum of weights of remaining elements not blocked
        # For tighter bound, only count elements from idx onwards that are not blocked
        ub = score
        for j in range(idx, N + 1):
            if j not in blocked_set:
                ub += w[j]
                if ub <= best_score:
                    break

        if ub <= best_score:
            return

        # Option 1: Exclude element idx
        dfs(idx + 1, S_set, score, blocked_set)

        if time.time() > deadline:
            return

        # Option 2: Include element idx (if not blocked)
        if idx not in blocked_set:
            # Check if idx conflicts with any element in S_set
            valid = True
            for a in S_set:
                if a in conflict_set[idx]:
                    valid = False
                    break

            if valid:
                new_S = S_set | {idx}
                # Compute new blocked set: add all elements that conflict with idx
                new_blocked = set(blocked_set)
                for y in conflict_set[idx]:
                    if y >= idx:  # Only care about future elements
                        new_blocked.add(y)

                dfs(idx + 1, new_S, score + w[idx], new_blocked)

    # Start DFS from empty set
    dfs(1, set(), Fraction(0), set())

    return list(best_set)
