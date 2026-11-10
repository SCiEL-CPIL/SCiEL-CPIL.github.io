# EVOLVE-BLOCK-START
import time
import random


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    deadline = time.time() + 290.0  # safety margin before 330s kill

    # --- Build O(1) index structures ---
    # var_clauses[v] -> list of clause indices containing literal v (signed)
    var_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            var_clauses[abs(l) - 1].append(l)

    # --- Weighted WalkSAT with O(1) delta updates ---
    def run(rng, seed_assign, max_steps):
        # x: 1 = True, 0 = False
        x = [1 if b else 0 for b in seed_assign]
        # clause_sat[i] = number of satisfied literals in clause i
        clause_sat = [0] * m
        for i, c in enumerate(clauses):
            s = 0
            for l in c:
                if x[abs(l) - 1] == (1 if l > 0 else 0):
                    s += 1
            clause_sat[i] = s

        unsat = sum(1 for s in clause_sat if s == 0)
        if unsat == 0:
            return [bool(v) for v in x]

        # clause weights: bias toward persistent conflicts
        w = [1.0] * m
        wsum = float(m)

        # Precompute weight-indexed unsat lists
        # We maintain a flat structure: for each clause, whether unsat
        # Use a simple list of unsat clause indices, rebuilt lazily
        unsat_list = [i for i in range(m) if clause_sat[i] == 0]

        def flip(v):
            nonlocal unsat, wsum
            x[v] ^= 1
            for l in var_clauses[v]:
                i = abs(l) - 1 if False else None  # placeholder
            # recompute affected clauses
            for l in var_clauses[v]:
                ci = var_clauses[v].index(l) if False else None
            # We need clause index, so store (clause_idx, literal) pairs
            pass

        # Redo: var_clauses should store (clause_idx, literal)
        # Rebuild properly
        # (handled below via vcl)
        return None

    # Rebuild var_clauses as list of (clause_idx, literal)
    vcl = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            vcl[abs(l) - 1].append((i, l))

    def run(rng, seed_assign, max_steps):
        x = [1 if b else 0 for b in seed_assign]
        clause_sat = [0] * m
        for i, c in enumerate(clauses):
            s = 0
            for l in c:
                if x[abs(l) - 1] == (1 if l > 0 else 0):
                    s += 1
            clause_sat[i] = s

        unsat = sum(1 for s in clause_sat if s == 0)
        if unsat == 0:
            return [bool(v) for v in x]

        w = [1.0] * m
        unsat_list = [i for i in range(m) if clause_sat[i] == 0]
        # index map for quick removal
        in_unsat = [i in set(unsat_list) for i in range(m)]  # expensive; use bool array
        in_unsat = [False] * m
        for i in unsat_list:
            in_unsat[i] = True

        p_walk = 0.25  # random walk probability
        step = 0
        best_unsat = unsat

        while step < max_steps:
            if not unsat_list:
                return [bool(v) for v in x]
            if time.time() >= deadline:
                break

            # pick an unsat clause weighted by w
            # build cumulative weights occasionally; for speed, pick uniformly then
            # with prob proportional to weight use weighted selection
            # Simple: choose uniformly among unsat, but prefer high weight via
            # a small weighted sampling using prefix sums every K steps
            if len(unsat_list) < 64:
                # weighted pick
                tot = 0.0
                picks = []
                for i in unsat_list:
                    tot += w[i]
                    picks.append(tot)
                r = rng.random() * tot
                # binary search
                lo, hi = 0, len(picks) - 1
                while lo < hi:
                    mid = (lo + hi) // 2
                    if picks[mid] < r:
                        lo = mid + 1
                    else:
                        hi = mid
                ci = unsat_list[lo]
            else:
                ci = unsat_list[rng.randrange(len(unsat_list))]

            clause = clauses[ci]
            # choose literal to flip: with prob p_walk random, else best delta
            if rng.random() < p_walk:
                l = clause[rng.randrange(3)]
            else:
                # pick literal whose flip reduces unsat most (delta)
                best_l = None
                best_delta = None
                for l in clause:
                    v = abs(l) - 1
                    # delta = change in unsat count if we flip v
                    d = 0
                    for (ci2, l2) in vcl[v]:
                        if clause_sat[ci2] == 0:
                            # currently unsat: flipping v will satisfy it if l2 matches new value
                            # new value of v = 1 - x[v]
                            nv = 1 - x[v]
                            if nv == (1 if l2 > 0 else 0):
                                d -= 1  # becomes satisfied
                        else:
                            # currently sat: flipping v may unsatisfy if l2 was the only sat
                            if clause_sat[ci2] == 1 and x[v] == (1 if l2 > 0 else 0):
                                d += 1  # becomes unsat
                    if best_delta is None or d < best_delta:
                        best_delta = d
                        best_l = l
                l = best_l

            v = abs(l) - 1
            old = x[v]
            x[v] = 1 - old

            # update clause_sat for affected clauses
            for (ci2, l2) in vcl[v]:
                before = clause_sat[ci2]
                # does l2 now match?
                now_match = (x[v] == (1 if l2 > 0 else 0))
                before_match = (old == (1 if l2 > 0 else 0))
                if before_match and not now_match:
                    clause_sat[ci2] = before - 1
                elif (not before_match) and now_match:
                    clause_sat[ci2] = before + 1

                # maintain unsat_list
                if before == 0 and clause_sat[ci2] > 0:
                    # became sat: remove from unsat_list
                    in_unsat[ci2] = False
                    unsat -= 1
                    # remove ci2 from unsat_list (swap with last)
                    try:
                        pos = unsat_list.index(ci2)
                    except ValueError:
                        pos = -1
                    if pos != -1:
                        last = unsat_list[-1]
                        unsat_list[pos] = last
                        in_unsat[last] = True
                        unsat_list.pop()
                elif before > 0 and clause_sat[ci2] == 0:
                    # became unsat: add
                    in_unsat[ci2] = True
                    unsat_list.append(ci2)
                    unsat += 1
                    # increase weight of this clause
                    w[ci2] += 0.5

            if unsat < best_unsat:
                best_unsat = unsat

            step += 1

        return [bool(v) for v in x]

    # --- Restart strategy ---
    rng = random.Random(0)
    base = [rng.random() < 0.5 for _ in range(n)]
    best_assign = None
    best_unsat = None

    def count_unsat(assign):
        x = [1 if b else 0 for b in assign]
        c = 0
        for cl in clauses:
            s = 0
            for l in cl:
                if x[abs(l) - 1] == (1 if l > 0 else 0):
                    s += 1
            if s == 0:
                c += 1
        return c

    # First run with generous budget
    steps = 200000
    res = run(rng, base, steps)
    if res is not None:
        u = count_unsat(res)
        if u == 0:
            return res
        best_assign = res
        best_unsat = u

    # Restart loop with perturbation
    restart_budget = 50000
    for r in range(200):
        if time.time() >= deadline:
            break
        # perturb current best
        cand = list(best_assign)
        # flip a few random variables
        for _ in range(3):
            cand[rng.randrange(n)] = not cand[rng.randrange(n)]
        res = run(rng, cand, restart_budget)
        if res is None:
            continue
        u = count_unsat(res)
        if u == 0:
            return res
        if best_unsat is None or u < best_unsat:
            best_unsat = u
            best_assign = res

    return best_assign if best_assign is not None else [False] * n
# EVOLVE-BLOCK-END
