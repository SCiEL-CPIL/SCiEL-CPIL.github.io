"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    deadline = time.time() + 130.0  # safety margin before 150s kill
    N = 1000
    MAX_REP = 2  # every sum has at most 2 representations a+b with a<=b

    # Greedy baseline: iterate 1..N, add if feasible
    def greedy_greedy():
        chosen = []
        counts = {}  # sum -> count
        for x in range(1, N + 1):
            new_sums = []
            for y in chosen:
                if x <= y:
                    new_sums.append(x + y)
                else:
                    new_sums.append(y + x)
            new_sums.append(2 * x)
            # Check feasibility
            feasible = True
            for s in new_sums:
                if counts.get(s, 0) + 1 > MAX_REP:
                    feasible = False
                    break
            if feasible:
                for s in new_sums:
                    counts[s] = counts.get(s, 0) + 1
                chosen.append(x)
        return sorted(chosen)

    # Local search: start from a set, try swaps to improve size
    def local_search(initial_set, max_iters=5000):
        S = sorted(initial_set)
        S_set = set(S)
        counts = {}
        # Build counts
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                counts[s] = counts.get(s, 0) + 1

        # Check validity
        def is_valid():
            return all(v <= MAX_REP for v in counts.values())

        if not is_valid():
            return S

        best = S[:]
        best_len = len(S)
        rng = random.Random(42)

        for _ in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            # Try to add a random element not in S
            candidates = [x for x in range(1, N + 1) if x not in S_set]
            if not candidates:
                break
            # Try random additions
            for _attempt in range(20):
                x = rng.choice(candidates)
                # Compute new sums
                new_sums = [x + y for y in S if x <= y] + [x + y for y in S if y < x] + [2 * x]
                # Actually simpler: all pairs (x, y) with x <= y
                new_sums = []
                for y in S:
                    if x <= y:
                        new_sums.append(x + y)
                    else:
                        new_sums.append(y + x)
                new_sums.append(2 * x)
                # Check if adding x keeps validity
                ok = True
                for s in new_sums:
                    if counts.get(s, 0) + 1 > MAX_REP:
                        ok = False
                        break
                if ok:
                    # Add x
                    for s in new_sums:
                        counts[s] = counts.get(s, 0) + 1
                    S.append(x)
                    S_set.add(x)
                    S.sort()
                    best = S[:]
                    best_len = len(S)
                    break

            # Try swap: remove one element, add another
            if len(S) > 10:
                for _swap in range(10):
                    i = rng.randrange(len(S))
                    remove_val = S[i]
                    # Compute sums involving remove_val
                    affected_sums = []
                    for j in range(len(S)):
                        if i <= j:
                            s = remove_val + S[j]
                            affected_sums.append((s, 1))
                        else:
                            s = S[j] + remove_val
                            affected_sums.append((s, 1))
                    # Actually, let's just recompute counts after removal
                    # Remove
                    for j in range(len(S)):
                        if i <= j:
                            s = remove_val + S[j]
                            counts[s] -= 1
                        elif j < i:
                            s = S[j] + remove_val
                            counts[s] -= 1
                    S.pop(i)
                    S_set.discard(remove_val)

                    # Try adding a new element
                    candidates = [x for x in range(1, N + 1) if x not in S_set]
                    if candidates:
                        x = rng.choice(candidates)
                        new_sums = []
                        for y in S:
                            if x <= y:
                                new_sums.append(x + y)
                            else:
                                new_sums.append(y + x)
                        new_sums.append(2 * x)
                        ok = True
                        for s in new_sums:
                            if counts.get(s, 0) + 1 > MAX_REP:
                                ok = False
                                break
                        if ok:
                            for s in new_sums:
                                counts[s] = counts.get(s, 0) + 1
                            S.append(x)
                            S_set.add(x)
                            S.sort()
                            if len(S) > best_len:
                                best = S[:]
                                best_len = len(S)
                        else:
                            # Revert: add back remove_val
                            for j in range(len(S)):
                                if remove_val <= S[j]:
                                    s = remove_val + S[j]
                                    counts[s] += 1
                                else:
                                    s = S[j] + remove_val
                                    counts[s] += 1
                            S.append(remove_val)
                            S_set.add(remove_val)
                            S.sort()
                    else:
                        # Revert
                        for j in range(len(S)):
                            if remove_val <= S[j]:
                                s = remove_val + S[j]
                                counts[s] += 1
                            else:
                                s = S[j] + remove_val
                                counts[s] += 1
                        S.append(remove_val)
                        S_set.add(remove_val)
                        S.sort()

        return best

    # Start with greedy
    best = greedy_greedy()
    best_len = len(best)

    # Try multiple random restarts with local search
    rng = random.Random(12345)
    num_restarts = 0
    while time.time() < deadline - 5.0:
        # Create a perturbed initial set
        if num_restarts == 0:
            initial = best[:]
        else:
            # Take best, remove some random elements, then greedy fill
            initial = best[:]
            remove_count = min(len(initial) // 5, 20)
            for _ in range(remove_count):
                idx = rng.randrange(len(initial))
                initial.pop(idx)
            # Greedy fill the gaps
            S_set = set(initial)
            counts = {}
            for i in range(len(initial)):
                for j in range(i, len(initial)):
                    s = initial[i] + initial[j]
                    counts[s] = counts.get(s, 0) + 1
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                new_sums = [x + y for y in initial if x <= y] + [x + y for y in initial if y < x] + [2 * x]
                ok = True
                for s in new_sums:
                    if counts.get(s, 0) + 1 > MAX_REP:
                        ok = False
                        break
                if ok:
                    for s in new_sums:
                        counts[s] = counts.get(s, 0) + 1
                    initial.append(x)
                    S_set.add(x)
            initial.sort()

        result = local_search(initial, max_iters=3000)
        if len(result) > best_len:
            best = result
            best_len = len(result)
        num_restarts += 1

    return sorted(best)

# EVOLVE-BLOCK-END
