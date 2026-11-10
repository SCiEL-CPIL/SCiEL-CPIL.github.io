import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill

    N = 256

    # Precompute conflicts
    # conflicts[i] is a set of j such that i and j conflict
    conflicts = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = math.gcd(i, j)
            x = i // g
            y = j // g
            if g % (x + y) == 0:
                conflicts[i].add(j)
                conflicts[j].add(i)

    # Initial set: all odds
    S = set(range(1, N + 1, 2))

    # List of evens not in S
    evens = [i for i in range(2, N + 1, 2) if i not in S]

    def greedy_add(S, evens, conflicts, deadline):
        """Greedily add any evens that don't conflict with S. Returns updated S, evens."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for e in list(evens):
                if time.time() > deadline:
                    break
                if not conflicts[e].intersection(S):
                    S.add(e)
                    evens.remove(e)
                    changed = True
        return S, evens

    # Initial greedy addition
    S, evens = greedy_add(S, evens, conflicts, deadline)

    def try_single_removal(S, evens, conflicts, deadline):
        """Try 1-for-k swaps: remove one element, add multiple evens."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list = list(S)
            for s_elem in S_list:
                if time.time() > deadline:
                    break
                S_test = S - {s_elem}
                # Find evens that become addable
                candidates = [e for e in evens if not conflicts[e].intersection(S_test)]
                if not candidates:
                    continue
                # Sort by conflict count with S_test (fewer conflicts = easier to add)
                candidates.sort(key=lambda e: len(conflicts[e] & S_test))
                S_test2 = S_test.copy()
                evens_test = list(evens)
                added = 0
                for e in candidates:
                    if time.time() > deadline:
                        break
                    if not conflicts[e].intersection(S_test2):
                        S_test2.add(e)
                        if e in evens_test:
                            evens_test.remove(e)
                        added += 1
                if added > 1:
                    S = S_test2
                    evens = evens_test
                    improved = True
                    break
        return S, evens

    def try_pair_removal(S, evens, conflicts, deadline):
        """Try 2-for-k swaps: remove two elements, add multiple evens."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list = list(S)
            n = len(S_list)
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    s1, s2 = S_list[i], S_list[j]
                    S_test = S - {s1, s2}
                    # Find evens that become addable
                    candidates = [e for e in evens if not conflicts[e].intersection(S_test)]
                    if len(candidates) < 3:
                        continue
                    # Sort by conflict count with S_test
                    candidates.sort(key=lambda e: len(conflicts[e] & S_test))
                    S_test2 = S_test.copy()
                    evens_test = list(evens)
                    added = 0
                    for e in candidates:
                        if time.time() > deadline:
                            break
                        if not conflicts[e].intersection(S_test2):
                            S_test2.add(e)
                            if e in evens_test:
                                evens_test.remove(e)
                            added += 1
                    if added > 2:
                        S = S_test2
                        evens = evens_test
                        improved = True
                        break
                if improved:
                    break
        return S, evens

    # Run local search phases in sequence, repeating until no improvement
    # Phase 1: single removal
    # Phase 2: pair removal
    # Phase 3: greedy add
    # Repeat until no improvement or deadline

    for _ in range(20):  # Max iterations
        if time.time() > deadline:
            break

        old_size = len(S)

        # Phase 1: single removal swaps
        S, evens = try_single_removal(S, evens, conflicts, deadline)
        if time.time() > deadline:
            break

        # Phase 2: pair removal swaps
        S, evens = try_pair_removal(S, evens, conflicts, deadline)
        if time.time() > deadline:
            break

        # Phase 3: greedy add
        S, evens = greedy_add(S, evens, conflicts, deadline)
        if time.time() > deadline:
            break

        if len(S) == old_size:
            break

    # Final greedy pass to ensure no easy additions remain
    S, evens = greedy_add(S, evens, conflicts, deadline)

    return list(S)
# EVOLVE-BLOCK-END
