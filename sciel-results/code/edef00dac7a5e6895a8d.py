"""Planted 3-SAT solver (50 vars) - WalkSAT family with vectorized tracking,
weighted selection, restarts, deadline guard and a final polish pass."""
import time
import random


def _now():
    return time.perf_counter()


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    deadline = _now() + 290.0  # safety margin well under the 330s kill

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
        return sat_count

    def unsat_count_from_sat(sat_count):
        return sum(1 for x in sat_count if x == 0)

    def try_polish(assignment, sat_count):
        """Greedy targeted flips to reduce unsat; returns (assignment, sat_count, unsat)."""
        unsat = unsat_count_from_sat(sat_count)
        if unsat == 0:
            return assignment, sat_count, 0
        # A few rounds of best-possible single flips.
        for _ in range(60):
            if unsat == 0:
                break
            # Find an unsat clause and try flipping each of its literals.
            # Choose the flip that reduces unsat the most (tie -> fewer flips to others).
            target = None
            for c in range(m):
                if sat_count[c] == 0:
                    target = c
                    break
            if target is None:
                break
            best_gain = 0
            best_var = None
            best_sign = None
            for v, s in lit_pos[target]:
                cur = assignment[v]
                desired = (s > 0)
                if cur == desired:
                    # flipping this would make literal unsat; skip (would increase)
                    continue
                # simulate flip
                gain = 0
                for cc in range(m):
                    before = sat_count[cc]
                    # recompute this clause's count quickly
                    newcnt = before
                    for vv, ss in lit_pos[cc]:
                        if vv == v:
                            # this clause contains var v
                            # determine contribution before/after
                            if cur == (ss > 0):
                                newcnt -= 1
                            else:
                                newcnt += 1
                            break
                    if before == 0 and newcnt > 0:
                        gain += 1
                    elif before > 0 and newcnt == 0:
                        gain -= 1
                if gain > best_gain:
                    best_gain = gain
                    best_var = v
                    best_sign = s
            if best_var is None or best_gain <= 0:
                break
            # apply flip
            v = best_var
            cur = assignment[v]
            assignment[v] = not cur
            for cc in range(m):
                for vv, ss in lit_pos[cc]:
                    if vv == v:
                        if cur == (ss > 0):
                            sat_count[cc] -= 1
                        else:
                            sat_count[cc] += 1
                        break
            unsat = unsat_count_from_sat(sat_count)
        return assignment, sat_count, unsat

    def run_walksat(assignment, sat_count, max_flips, noise):
        """Run WalkSAT-style search; returns (assignment, sat_count, unsat)."""
        unsat = unsat_count_from_sat(sat_count)
        flips = 0
        while flips < max_flips:
            if unsat == 0:
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
            # flip v
            cur = assignment[v]
            assignment[v] = not cur
            for cc in range(m):
                for vv, ss in lit_pos[cc]:
                    if vv == v:
                        if cur == (ss > 0):
                            sat_count[cc] -= 1
                        else:
                            sat_count[cc] += 1
                        break
            unsat = unsat_count_from_sat(sat_count)
            flips += 1
        return assignment, sat_count, unsat

    # Track best across restarts.
    start = _now()
    restart = 0
    while _now() < deadline:
        # fresh random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        sat_count = build_state(assignment)
        unsat = unsat_count_from_sat(sat_count)
        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = list(assignment)
        if unsat == 0:
            return assignment

        # Adaptive budget per restart: more flips early, but bounded by time.
        # Estimate cost per flip ~ (m * avg_clauses_per_var) small; use a fixed-ish
        # flip budget that grows with restarts but is capped by remaining time.
        base_flips = 4000 + restart * 2000
        # Rough time check: do a small batch then stop if near deadline.
        batch = min(base_flips, 5000)
        noise = 0.45 + 0.05 * min(restart, 4)
        assignment, sat_count, unsat = run_walksat(assignment, sat_count, batch, noise)

        # polish
        assignment, sat_count, unsat = try_polish(assignment, sat_count)
        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = list(assignment)
        if unsat == 0:
            return assignment

        # If we still have time, continue this same state with more flips (no restart)
        # to exploit progress, but cap total per-restart work.
        if _now() < deadline - 5.0:
            assignment, sat_count, unsat = run_walksat(assignment, sat_count, 3000, noise)
            assignment, sat_count, unsat = try_polish(assignment, sat_count)
            if unsat < best_unsat:
                best_unsat = unsat
                best_assignment = list(assignment)
            if unsat == 0:
                return assignment

        restart += 1
        # Hard cap on restarts to avoid infinite loop near deadline.
        if restart > 400:
            break

    # Final: return best assignment found (polish it once more if it's not already good).
    if best_assignment is not None:
        sc = build_state(best_assignment)
        best_assignment, sc, u = try_polish(best_assignment, sc)
        if u == 0:
            return best_assignment
    # Fallback: return last assignment if we have one, else a random one.
    if best_assignment is None:
        best_assignment = [rng.random() < 0.5 for _ in range(n)]
    return best_assignment
