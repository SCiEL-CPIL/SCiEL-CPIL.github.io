"""Planted 3-SAT solver (50 vars) - WalkSAT family with O(1) unsat tracking,
weighted selection, restarts, deadline guard, improved polish and perturbation."""
import time
import random


def _now():
    return time.perf_counter()


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    deadline = _now() + 285.0  # safety margin well under the 330s kill

    # Deterministic seed from instance shape for reproducibility.
    base_seed = (n * 1000003 + m * 7919) & 0xFFFFFFFF
    rng = random.Random(base_seed)

    # Precompute, for each clause, the literal positions and the variable index.
    # lit_pos[c] = list of (var_idx, sign) where sign is +1 if literal positive.
    lit_pos = []
    for c in clauses:
        lit_pos.append([(abs(l) - 1, 1 if l > 0 else -1) for l in c])

    # Variable occurrence count (for weighted selection).
    var_occ = [0] * n
    for c in lit_pos:
        for v, _ in c:
            var_occ[v] += 1

    # Precompute: for each variable, list of (clause_idx, sign) it appears in.
    var_clauses = [[] for _ in range(n)]
    for c in range(m):
        for v, s in lit_pos[c]:
            var_clauses[v].append((c, s))

    best_assignment = None
    best_unsat = m + 1

    def build_state(assignment):
        # sat_count[c] = number of satisfied literals in clause c
        sat_count = [0] * m
        for c in range(m):
            cnt = 0
            for v, s in lit_pos[c]:
                if assignment[v] == (s > 0):
                    cnt += 1
            sat_count[c] = cnt
        unsat = sum(1 for x in sat_count if x == 0)
        return sat_count, unsat

    def try_polish(assignment, sat_count, unsat):
        """Greedy targeted flips to reduce unsat; returns (assignment, sat_count, unsat)."""
        if unsat == 0:
            return assignment, sat_count, 0
        # Multiple rounds of best-possible single flips.
        for _ in range(200):
            if unsat == 0:
                break
            # Find an unsat clause
            target = None
            for c in range(m):
                if sat_count[c] == 0:
                    target = c
                    break
            if target is None:
                break
            best_gain = 0
            best_var = None
            for v, s in lit_pos[target]:
                cur = assignment[v]
                desired = (s > 0)
                if cur == desired:
                    continue
                # simulate flip: only need to check clauses containing v
                gain = 0
                for cc, ss in var_clauses[v]:
                    before = sat_count[cc]
                    if cur == (ss > 0):
                        newcnt = before - 1
                    else:
                        newcnt = before + 1
                    if before == 0 and newcnt > 0:
                        gain += 1
                    elif before > 0 and newcnt == 0:
                        gain -= 1
                if gain > best_gain:
                    best_gain = gain
                    best_var = v
            if best_var is None or best_gain <= 0:
                break
            # apply flip
            v = best_var
            cur = assignment[v]
            assignment[v] = not cur
            for cc, ss in var_clauses[v]:
                if cur == (ss > 0):
                    sat_count[cc] -= 1
                    if sat_count[cc] == 0:
                        unsat += 1
                else:
                    sat_count[cc] += 1
                    if sat_count[cc] == 1:
                        unsat -= 1
        return assignment, sat_count, unsat

    def run_walksat(assignment, sat_count, unsat, max_flips, noise):
        """Run WalkSAT-style search with O(1) unsat tracking; returns (assignment, sat_count, unsat)."""
        flips = 0
        while flips < max_flips:
            if unsat == 0:
                break
            # Time guard: check every 1000 flips
            if flips > 0 and flips % 1000 == 0 and _now() > deadline - 3.0:
                break
            # pick a random unsat clause
            c = rng.randrange(m)
            while sat_count[c] != 0:
                c = rng.randrange(m)
            # pick a literal: with prob noise pick uniformly, else weighted by var_occ
            if rng.random() < noise:
                v, s = lit_pos[c][rng.randrange(len(lit_pos[c]))]
            else:
                # weighted by 1/var_occ (prefer less-occurring vars)
                weights = [1.0 / max(1, var_occ[v]) for v, _ in lit_pos[c]]
                total = sum(weights)
                r = rng.random() * total
                acc = 0.0
                chosen = 0
                for i, w in enumerate(weights):
                    acc += w
                    if r <= acc:
                        chosen = i
                        break
                v, s = lit_pos[c][chosen]
            # flip v with O(1) unsat update
            cur = assignment[v]
            assignment[v] = not cur
            for cc, ss in var_clauses[v]:
                if cur == (ss > 0):
                    sat_count[cc] -= 1
                    if sat_count[cc] == 0:
                        unsat += 1
                else:
                    sat_count[cc] += 1
                    if sat_count[cc] == 1:
                        unsat -= 1
            flips += 1
        return assignment, sat_count, unsat

    # Track best across restarts.
    restart = 0
    while _now() < deadline:
        # Choose initial assignment: random for first few, then perturb best
        if restart < 3 or best_assignment is None:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        else:
            # Perturb best: flip a few random variables
            assignment = list(best_assignment)
            n_flips = min(3 + rng.randint(0, 3), n)
            for _ in range(n_flips):
                assignment[rng.randrange(n)] = not assignment[rng.randrange(n)]

        sat_count, unsat = build_state(assignment)
        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = list(assignment)
        if unsat == 0:
            return assignment

        # Adaptive budget per restart
        base_flips = 5000 + restart * 1500
        batch = min(base_flips, 8000)
        noise = 0.40 + 0.05 * min(restart, 6)
        assignment, sat_count, unsat = run_walksat(assignment, sat_count, unsat, batch, noise)

        # polish
        assignment, sat_count, unsat = try_polish(assignment, sat_count, unsat)
        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = list(assignment)
        if unsat == 0:
            return assignment

        # Continue with more flips if time remains
        if _now() < deadline - 10.0:
            assignment, sat_count, unsat = run_walksat(assignment, sat_count, unsat, 5000, noise)
            assignment, sat_count, unsat = try_polish(assignment, sat_count, unsat)
            if unsat < best_unsat:
                best_unsat = unsat
                best_assignment = list(assignment)
            if unsat == 0:
                return assignment

        restart += 1
        if restart > 500:
            break

    # Final: return best assignment found (polish it once more if it's not already good).
    if best_assignment is not None:
        sc, u = build_state(best_assignment)
        best_assignment, sc, u = try_polish(best_assignment, sc, u)
        if u == 0:
            return best_assignment
    # Fallback
    if best_assignment is None:
        best_assignment = [rng.random() < 0.5 for _ in range(n)]
    return best_assignment
