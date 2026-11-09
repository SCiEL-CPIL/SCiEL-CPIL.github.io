"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from random import Random

def construct():
    deadline = time.time() + 143.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def build_conflict_graph(values):
        """adj[i] = list of j != i such that values[i]+values[j] is in values.
        Sum-free <=> chosen set is an independent set in this graph (a+b=c with
        a,b,c in S means an edge between the two summands; a=b handled by 2a=c)."""
        val_set = set(values)
        n = len(values)
        adj = [[] for _ in range(n)]
        for i in range(n):
            vi = values[i]
            for j in range(n):
                if i == j:
                    continue
                if (vi + values[j]) in val_set:
                    adj[i].append(j)
        for i in range(n):
            adj[i].sort()
        return adj

    def greedy_init(values, adj, order):
        """Pick vertices in `order`, keep if it has no neighbor already chosen."""
        n = len(values)
        chosen = [False] * n
        for v in order:
            if any(chosen[nb] for nb in adj[v]):
                continue
            chosen[v] = True
        return chosen

    def polish(chosen, adj, values, order, deadline):
        """Repeated add-only pass in `order` until a full pass adds nothing."""
        n = len(values)
        cur = chosen[:]
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for v in order:
                if cur[v]:
                    continue
                if any(cur[nb] for nb in adj[v]):
                    continue
                cur[v] = True
                improved = True
        return cur

    def local_search(values, adj, cur, deadline, rng):
        """Add-only polish, then swap moves (remove one chosen, add one free)."""
        n = len(values)
        best = cur[:]
        best_size = sum(cur)

        # add-only polish first
        cur = polish(cur, adj, values, range(n), deadline)
        cur_size = sum(cur)
        if cur_size > best_size:
            best_size = cur_size
            best = cur[:]

        while time.time() < deadline:
            chosen = [i for i in range(n) if cur[i]]
            free = [i for i in range(n) if not cur[i]]
            if not free:
                break

            # shuffle for diversity
            rng.shuffle(chosen)
            rng.shuffle(free)

            moved = False
            for v_in in chosen:
                if time.time() > deadline:
                    break
                if not cur[v_in]:
                    continue
                for v_out in free:
                    if cur[v_out]:
                        continue
                    # v_out must be non-conflicting with all chosen except v_in
                    ok = True
                    for nb in adj[v_out]:
                        if cur[nb] and nb != v_in:
                            ok = False
                            break
                    if ok:
                        cur[v_in] = False
                        cur[v_out] = True
                        cur_size += 0  # size unchanged on a swap
                        moved = True
                        break
                if moved:
                    break

            if not moved:
                # no improving swap; try a few random 1-for-1 swaps to escape
                escaped = False
                for _ in range(min(10, len(chosen))):
                    v_in = rng.choice(chosen)
                    if not cur[v_in]:
                        continue
                    cur[v_in] = False
                    cur_size -= 1
                    # add any free that now fits
                    added = False
                    for v_out in free:
                        if cur[v_out]:
                            continue
                        if any(cur[nb] for nb in adj[v_out]):
                            continue
                        cur[v_out] = True
                        cur_size += 1
                        added = True
                        break
                    if not added:
                        # put v_in back
                        cur[v_in] = True
                        cur_size += 1
                        continue
                    escaped = True
                    break
                if not escaped:
                    # stuck: polish once more then stop
                    cur = polish(cur, adj, values, range(n), deadline)
                    cur_size = sum(cur)
                    if cur_size > best_size:
                        best_size = cur_size
                        best = cur[:]
                    break

            # track best
            if cur_size > best_size:
                best_size = cur_size
                best = cur[:]

        # final polish on best
        best = polish(best, adj, values, range(n), deadline)
        return best

    def solve(values, time_budget, deadline):
        n = len(values)
        adj = build_conflict_graph(values)

        # proportional: more restarts for larger / longer-budget instances
        base = 8
        restarts = max(4, int(base * (time_budget / 8.0) ** 0.5))
        restarts = min(restarts, 40)

        per = time_budget / restarts
        best = None
        best_size = -1

        for r in range(restarts):
            if time.time() > deadline:
                break
            rng = Random(1000 + r)
            order = list(range(n))
            if r % 3 == 0:
                rng.shuffle(order)
            elif r % 3 == 1:
                # lowest conflict degree first
                order = sorted(order, key=lambda i: len(adj[i]))
            else:
                # highest conflict degree first
                order = sorted(order, key=lambda i: -len(adj[i]))

            cur = greedy_init(values, adj, order)
            sub_deadline = min(deadline, time.time() + per)
            cur = local_search(values, adj, cur, sub_deadline, rng)
            sz = sum(cur)
            if sz > best_size:
                best_size = sz
                best = cur[:]

        # final multi-order polish on the best solution
        for o in (range(n), sorted(range(n), key=lambda i: len(adj[i]))):
            best = polish(best, adj, values, o, deadline)

        return [values[i] for i in range(n) if best[i]]

    # proportional time allocation: weight by size so the 240-instance gets the most
    specs = ((60, 801), (120, 802), (240, 803))
    weights = [s for s, _ in specs]
    total_w = sum(weights)
    total_time = deadline - time.time() - 5.0  # reserve 5s for final work
    if total_time < 3.0:
        total_time = 3.0

    results = []
    for idx, (size, seed) in enumerate(specs):
        elems = instance(size, seed)
        budget = total_time * weights[idx] / total_w
        if time.time() > deadline - 3.0:
            # not enough time; use a quick odd-number fallback
            results.append([x for x in elems if x % 2])
            continue
        subset = solve(elems, budget, deadline)
        results.append(subset)

    while len(results) < 3:
        size, seed = specs[len(results)]
        elems = instance(size, seed)
        results.append([x for x in elems if x % 2])

    return results
# EVOLVE-BLOCK-END
