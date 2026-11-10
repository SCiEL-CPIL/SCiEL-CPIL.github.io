"""Rewritten program: hypergraph model of LCM triangles with branch-and-bound,
fast incremental conflict counters, local search, simulated annealing, and
maximal independent set heuristics to maximize the valid subset of [1,128]."""
import time
import random
import numpy as np
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    n = 128
    # Precompute LCM table
    lcm_table = np.zeros((n + 1, n + 1), dtype=np.int32)
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            l = _lcm(i, j)
            lcm_table[i, j] = l
            lcm_table[j, i] = l

    # Build hypergraph: bad triples
    bad_triples = []  # list of (a, b, c) with a < b < c
    triple_by_elem = {x: [] for x in range(1, n + 1)}
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a, b]
            for c in range(b + 1, n + 1):
                if lcm_table[a, c] == lab and lcm_table[b, c] == lab:
                    bad_triples.append((a, b, c))
                    triple_by_elem[a].append((b, c))
                    triple_by_elem[b].append((a, c))
                    triple_by_elem[c].append((a, b))

    # Conflict graph: edge between x,y if they appear together in some bad triple
    conflict = [set() for _ in range(n + 1)]
    for (a, b, c) in bad_triples:
        conflict[a].add(b); conflict[a].add(c)
        conflict[b].add(a); conflict[b].add(c)
        conflict[c].add(a); conflict[c].add(b)

    def is_bad_triple(a, b, c):
        lab = lcm_table[a, b]
        lac = lcm_table[a, c]
        lbc = lcm_table[b, c]
        return lab == lac == lbc

    def check_valid(selected):
        sel = list(selected)
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                for k in range(j + 1, len(sel)):
                    if is_bad_triple(sel[i], sel[j], sel[k]):
                        return False
        return True

    # ---------- Maximal independent set (greedy) ----------
    def greedy_mis(order):
        """Greedy maximal independent set: add x if it has no conflict with selected."""
        selected = []
        selected_set = set()
        for x in order:
            if any(y in selected_set for y in conflict[x]):
                continue
            selected.append(x)
            selected_set.add(x)
        return selected

    # ---------- Fast local search with conflict counters ----------
    def local_search(selected, max_iterations=200):
        """Improve by removing one element and adding another (or just adding).
        Uses conflict counters for O(1) validity check of candidates."""
        selected = list(selected)
        selected_set = set(selected)
        # counter[x] = number of bad triples formed by x with two elements in selected
        counter = np.zeros(n + 1, dtype=np.int32)
        for x in range(1, n + 1):
            if x in selected_set:
                continue
            cnt = 0
            for (u, v) in triple_by_elem[x]:
                if u in selected_set and v in selected_set:
                    cnt += 1
            counter[x] = cnt

        improved = True
        iters = 0
        while improved and iters < max_iterations and time.time() < deadline:
            improved = False
            iters += 1
            # First try: just add any element with counter 0
            for x in range(1, n + 1):
                if x in selected_set:
                    continue
                if counter[x] == 0:
                    # add x
                    selected.append(x)
                    selected_set.add(x)
                    # update counters for elements conflicting with x
                    for y in conflict[x]:
                        if y in selected_set:
                            continue
                        for (u, v) in triple_by_elem[y]:
                            if u == x and v in selected_set:
                                counter[y] += 1
                            elif v == x and u in selected_set:
                                counter[y] += 1
                    improved = True
                    break
            if improved:
                continue
            # Try remove one + add another
            for i in range(len(selected)):
                removed = selected[i]
                selected_set.discard(removed)
                # Update counters: removing removed may decrease counter of others
                for y in conflict[removed]:
                    if y in selected_set:
                        continue
                    for (u, v) in triple_by_elem[y]:
                        if u == removed and v in selected_set:
                            counter[y] -= 1
                        elif v == removed and u in selected_set:
                            counter[y] -= 1
                # Now try to add something
                best_add = None
                for x in range(1, n + 1):
                    if x in selected_set:
                        continue
                    if counter[x] == 0:
                        best_add = x
                        break
                if best_add is not None:
                    selected.append(best_add)
                    selected_set.add(best_add)
                    for y in conflict[best_add]:
                        if y in selected_set:
                            continue
                        for (u, v) in triple_by_elem[y]:
                            if u == best_add and v in selected_set:
                                counter[y] += 1
                            elif v == best_add and u in selected_set:
                                counter[y] += 1
                    improved = True
                else:
                    # put removed back
                    selected_set.add(removed)
                    for y in conflict[removed]:
                        if y in selected_set:
                            continue
                        for (u, v) in triple_by_elem[y]:
                            if u == removed and v in selected_set:
                                counter[y] += 1
                            elif v == removed and u in selected_set:
                                counter[y] += 1
        return selected

    # ---------- Simulated annealing ----------
    def simulated_annealing(initial, iters=5000):
        selected = list(initial)
        selected_set = set(selected)
        # Build counter
        counter = np.zeros(n + 1, dtype=np.int32)
        for x in range(1, n + 1):
            if x in selected_set:
                continue
            cnt = 0
            for (u, v) in triple_by_elem[x]:
                if u in selected_set and v in selected_set:
                    cnt += 1
            counter[x] = cnt

        best = list(selected)
        best_len = len(selected)
        T = 1.0
        cooling = 0.999
        for it in range(iters):
            if time.time() >= deadline:
                break
            r = random.random()
            if r < 0.4:
                # try add
                candidates = [x for x in range(1, n + 1) if x not in selected_set]
                if not candidates:
                    continue
                x = random.choice(candidates)
                # add x (may create conflicts)
                selected.append(x)
                selected_set.add(x)
                for y in conflict[x]:
                    if y in selected_set:
                        continue
                    for (u, v) in triple_by_elem[y]:
                        if u == x and v in selected_set:
                            counter[y] += 1
                        elif v == x and u in selected_set:
                            counter[y] += 1
                # check validity: count bad triples in set
                # We need to track actual bad triple count
                # For simplicity, recompute validity
                if len(selected) > best_len and check_valid(selected):
                    best = list(selected)
                    best_len = len(selected)
            elif r < 0.8:
                # try remove
                if not selected:
                    continue
                i = random.randrange(len(selected))
                removed = selected[i]
                selected.pop(i)
                selected_set.discard(removed)
                for y in conflict[removed]:
                    if y in selected_set:
                        continue
                    for (u, v) in triple_by_elem[y]:
                        if u == removed and v in selected_set:
                            counter[y] -= 1
                        elif v == removed and u in selected_set:
                            counter[y] -= 1
            else:
                # swap: remove one, add another
                if not selected:
                    continue
                i = random.randrange(len(selected))
                removed = selected[i]
                candidates = [x for x in range(1, n + 1) if x not in selected_set]
                if not candidates:
                    continue
                x = random.choice(candidates)
                selected.pop(i)
                selected_set.discard(removed)
                for y in conflict[removed]:
                    if y in selected_set:
                        continue
                    for (u, v) in triple_by_elem[y]:
                        if u == removed and v in selected_set:
                            counter[y] -= 1
                        elif v == removed and u in selected_set:
                            counter[y] -= 1
                selected.append(x)
                selected_set.add(x)
                for y in conflict[x]:
                    if y in selected_set:
                        continue
                    for (u, v) in triple_by_elem[y]:
                        if u == x and v in selected_set:
                            counter[y] += 1
                        elif v == x and u in selected_set:
                            counter[y] += 1
                if len(selected) > best_len and check_valid(selected):
                    best = list(selected)
                    best_len = len(selected)
            T *= cooling
        return best

    # ---------- Branch and bound exact solver ----------
    def branch_and_bound(candidates, current, current_set, upper_bound):
        """Try to extend current with elements from candidates to maximize size.
        Returns best valid extension."""
        best = list(current)
        best_len = len(current)
        if not candidates or len(current) + len(candidates) <= best_len:
            return best
        # Sort candidates by degree (ascending) for better pruning
        candidates = sorted(candidates, key=lambda x: len(conflict[x]))
        for x in candidates:
            if time.time() >= deadline:
                break
            # Check if x conflicts with any in current
            if any(y in current_set for y in conflict[x]):
                continue
            new_current = current + [x]
            new_set = current_set | {x}
            # Filter candidates that don't conflict with x
            new_candidates = [y for y in candidates if y > x and not (y in conflict[x] or any(z in new_set for z in conflict[y]))]
            # Upper bound: current + 1 + len(new_candidates)
            ub = len(new_current) + len(new_candidates)
            if ub <= best_len:
                continue
            result = branch_and_bound(new_candidates, new_current, new_set, ub)
            if len(result) > best_len:
                best = result
                best_len = len(result)
        return best

    # ---------- Main search ----------
    # Phase 1: Multiple greedy MIS with different orderings
    best_set = greedy_mis(list(range(1, n + 1)))

    # Reverse order
    alt = greedy_mis(list(range(n, 0, -1)))
    if len(alt) > len(best_set):
        best_set = alt

    # Sorted by degree (ascending)
    alt = greedy_mis(sorted(range(1, n + 1), key=lambda x: len(conflict[x])))
    if len(alt) > len(best_set):
        best_set = alt

    # Sorted by degree (descending)
    alt = greedy_mis(sorted(range(1, n + 1), key=lambda x: len(conflict[x]), reverse=True))
    if len(alt) > len(best_set):
        best_set = alt

    # Random restarts
    random.seed(42)
    for _ in range(50):
        if time.time() >= deadline:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        alt = greedy_mis(order)
        if len(alt) > len(best_set):
            best_set = alt

    # Phase 2: Local search
    best_set = local_search(best_set, max_iterations=300)

    # Phase 3: More random restarts + local search
    for _ in range(30):
        if time.time() >= deadline:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        alt = greedy_mis(order)
        alt = local_search(alt, max_iterations=100)
        if len(alt) > len(best_set):
            best_set = alt

    # Phase 4: Simulated annealing
    if time.time() < deadline - 10:
        sa_result = simulated_annealing(best_set, iters=10000)
        if len(sa_result) > len(best_set):
            best_set = sa_result
            best_set = local_search(best_set, max_iterations=200)

    # Phase 5: Try branch and bound on remaining candidates
    if time.time() < deadline - 5:
        selected_set = set(best_set)
        candidates = [x for x in range(1, n + 1) if x not in selected_set]
        if candidates:
            # Try to add a few more via B&B
            bb_result = branch_and_bound(candidates, list(best_set), selected_set, len(best_set) + len(candidates))
            if len(bb_result) > len(best_set):
                best_set = bb_result

    # Phase 6: Final local search
    best_set = local_search(best_set, max_iterations=200)

    # Ensure nonempty
    if not best_set:
        best_set = [1]

    return best_set
