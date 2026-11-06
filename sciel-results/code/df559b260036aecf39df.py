import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin for 60s limit

    N = 180
    elements = list(range(1, N + 1))

    # 1. Precompute all forbidden triples (a, b, c) with distinct elements in [1, N]
    #    Equation: 1/a = 1/b + 1/c  =>  (b-a)(c-a) = a^2,  a is smallest.
    #    For a < b: c = ab/(b-a), require 1<=c<=N, c distinct from a,b.
    triples = []
    elem_to_triples = {i: [] for i in range(1, N + 1)}

    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    t = (a, b, c)
                    idx = len(triples)
                    triples.append(t)
                    for x in t:
                        elem_to_triples[x].append(idx)

    # For each element x, the set of pairs (b,c) that with x form a forbidden triple.
    # Used for fast "can I add x?" checks.
    conflict_pairs = {x: [] for x in range(1, N + 1)}
    for (a, b, c) in triples:
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))

    def can_add(x, S):
        for (b, c) in conflict_pairs[x]:
            if b in S and c in S:
                return False
        return True

    def greedy_add(order):
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S

    def local_search(S, time_budget):
        """Hill-climb: repeatedly try to add any missing element; if none,
        remove one element at random and try to add back greedily. Accept
        only if size does not decrease (we track best)."""
        S = set(S)
        best_S = set(S)
        best_size = len(S)
        missing = [x for x in range(1, N + 1) if x not in S]

        while time.time() < time_budget:
            # Try to add any missing element
            added = False
            new_missing = []
            for x in missing:
                if can_add(x, S):
                    S.add(x)
                    added = True
                else:
                    new_missing.append(x)
            missing = new_missing
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
            if not added:
                # Stuck: try removing one element to unblock
                if not S:
                    break
                # Pick a random element to remove; prefer ones in many active conflicts
                # Compute a cheap heuristic: count of conflict_pairs[x] where both partners in S
                # But that's O(|S|) per x; N is small so just pick randomly among S
                # with a slight bias toward elements that are "blocking" (in many active triples).
                # Simple: random pick.
                victim = random.choice(list(S))
                S.remove(victim)
                missing.append(victim)
                # After removal, loop back to try adding
                continue
            if not missing:
                # All elements added; done
                break

        return best_S

    # Strategy: many randomized greedy passes, then local search on the best few.
    best_S = set()
    best_size = 0

    # Deterministic pass: add from large to small (keeps big numbers, fills small gaps)
    S = greedy_add(list(range(N, 0, -1)))
    if len(S) > best_size:
        best_size = len(S)
        best_S = set(S)

    # Randomized passes: random order
    rng = random.Random()
    num_random_passes = 2000
    for i in range(num_random_passes):
        if time.time() > deadline - 15:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        S = greedy_add(order)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

    # Local search on the best set found
    S = local_search(best_S, deadline - 2)
    if len(S) > best_size:
        best_size = len(S)
        best_S = set(S)

    # Also local search a couple of other good candidates if time remains
    # Re-run a few more randomized passes and local-search them
    for i in range(200):
        if time.time() > deadline - 5:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        S = greedy_add(order)
        if len(S) >= best_size:  # only polish candidates that are competitive
            S = local_search(S, min(deadline - 5, time.time() + 1.0))
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)

    # Final polish on the best
    if time.time() < deadline - 1:
        S = local_search(best_S, deadline - 1)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

    result = sorted(best_S)
    # Ensure nonempty
    if not result:
        result = [N]
    return result
