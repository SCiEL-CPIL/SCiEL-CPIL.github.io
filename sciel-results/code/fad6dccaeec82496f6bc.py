"""Planted 3-SAT solver: incremental WalkSAT with dynamic weights + no-improvement restarts + SA escapes + polish."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 290.0  # safety margin vs 330s kill

    # Pre-index: for each variable, list of (clause_idx, literal)
    var_clauses = [[] for _ in range(n)]
    for ci, cl in enumerate(clauses):
        for lit in cl:
            var_clauses[abs(lit) - 1].append((ci, lit))

    # Incremental state tracking
    def init_state(a):
        """Initialize clause satisfaction counts and unsat list from assignment a."""
        clause_sat_count = [0] * m
        unsat = []
        for ci, cl in enumerate(clauses):
            cnt = 0
            for lit in cl:
                if a[abs(lit) - 1] == (lit > 0):
                    cnt += 1
            clause_sat_count[ci] = cnt
            if cnt == 0:
                unsat.append(ci)
        return clause_sat_count, unsat

    def update_state(v, old_val, new_val, clause_sat_count, unsat, unsat_set):
        """Update clause state after flipping variable v from old_val to new_val."""
        for (ci, lit) in var_clauses[v]:
            old_sat = (old_val == (lit > 0))
            new_sat = (new_val == (lit > 0))
            if old_sat == new_sat:
                continue
            old_cnt = clause_sat_count[ci]
            new_cnt = old_cnt + (1 if new_sat else -1)
            clause_sat_count[ci] = new_cnt
            if old_cnt == 0 and new_cnt > 0:
                if ci in unsat_set:
                    unsat_set.discard(ci)
            elif old_cnt > 0 and new_cnt == 0:
                unsat.append(ci)
                unsat_set.add(ci)
        # Clean up unsat list
        cleaned = []
        for ci in unsat:
            if clause_sat_count[ci] == 0:
                cleaned.append(ci)
        unsat[:] = cleaned

    best_assignment = None
    best_unsat = m + 1

    def update_best(a):
        nonlocal best_assignment, best_unsat
        clause_sat_count, unsat = init_state(a)
        u = len(unsat)
        if u < best_unsat:
            best_unsat = u
            best_assignment = a[:]
        return u

    rng = random.Random(12345)

    def walk_sat(a, max_steps, no_improve_limit=2000):
        """
        WalkSAT with dynamic clause weights and no-improvement restart.
        Returns (assignment, unsat_count, converged).
        """
        clause_weights = [1.0] * m
        clause_sat_count, unsat = init_state(a)
        unsat_set = set(unsat)
        cur_unsat = len(unsat)
        best_local_unsat = cur_unsat
        steps_since_improve = 0
        converged = False

        for step in range(max_steps):
            if time.time() > deadline:
                return a, cur_unsat, converged

            # Lazy cleanup
            if len(unsat) != len(unsat_set):
                unsat[:] = [ci for ci in unsat if clause_sat_count[ci] == 0]
                unsat_set = set(unsat)
                cur_unsat = len(unsat)

            if not unsat:
                return a, 0, True

            # Track improvement
            if cur_unsat < best_local_unsat:
                best_local_unsat = cur_unsat
                steps_since_improve = 0
            else:
                steps_since_improve += 1

            # No-improvement restart: perturb and reset weights
            if steps_since_improve >= no_improve_limit:
                # Perturb: flip a few random variables in unsat clauses
                for _ in range(min(3, len(unsat))):
                    ci = rng.choice(unsat)
                    v = abs(rng.choice(clauses[ci])) - 1
                    old_val = a[v]
                    a[v] = not old_val
                    update_state(v, old_val, a[v], clause_sat_count, unsat, unsat_set)
                # Reset weights
                clause_weights = [1.0] * m
                cur_unsat = len(unsat_set)
                steps_since_improve = 0
                continue

            # Pick clause weighted by clause_weights
            total_w = 0.0
            for ci in unsat:
                total_w += clause_weights[ci]
            if total_w <= 0:
                ci = rng.choice(unsat)
            else:
                r = rng.random() * total_w
                acc = 0.0
                ci = unsat[-1]
                for c in unsat:
                    acc += clause_weights[c]
                    if r <= acc:
                        ci = c
                        break
            cl = clauses[ci]

            # Score each literal in the clause
            best_var = None
            best_score = -1e18
            best_val = None
            for lit in cl:
                v = abs(lit) - 1
                desired = (lit > 0)
                if a[v] == desired:
                    continue
                gain = 0.0
                loss = 0.0
                for (ci2, lit2) in var_clauses[v]:
                    old_sat = (a[v] == (lit2 > 0))
                    new_sat = (desired == (lit2 > 0))
                    if old_sat == new_sat:
                        continue
                    old_cnt = clause_sat_count[ci2]
                    new_cnt = old_cnt + (1 if new_sat else -1)
                    if old_cnt == 0 and new_cnt > 0:
                        gain += clause_weights[ci2]
                    elif old_cnt > 0 and new_cnt == 0:
                        loss += clause_weights[ci2]
                score = gain - 1.0 * loss
                if score > best_score:
                    best_score = score
                    best_var = v
                    best_val = desired

            if best_var is None:
                v = abs(cl[0]) - 1
                old_val = a[v]
                new_val = not old_val
                a[v] = new_val
                update_state(v, old_val, new_val, clause_sat_count, unsat, unsat_set)
                continue

            old_val = a[best_var]
            a[best_var] = best_val
            update_state(best_var, old_val, best_val, clause_sat_count, unsat, unsat_set)

            # Update clause weights: increase weight of selected unsat clause, decay others slightly
            clause_weights[ci] *= 1.08
            # Periodic renormalization to prevent overflow
            if step % 300 == 0:
                max_w = 0.0
                for w in clause_weights:
                    if w > max_w:
                        max_w = w
                if max_w > 0:
                    clause_weights = [w / max_w for w in clause_weights]

        return a, len(unsat_set), False

    def sa_escape(a, steps, t0):
        """Simulated annealing escape with incremental updates."""
        clause_sat_count, unsat = init_state(a)
        unsat_set = set(unsat)

        for step in range(steps):
            if time.time() > deadline:
                return a

            if not unsat:
                return a

            ci = rng.choice(unsat)
            cl = clauses[ci]
            v = abs(rng.choice(cl)) - 1
            new_val = not a[v]
            old_val = a[v]

            old_unsat_count = len(unsat_set)
            a[v] = new_val
            update_state(v, old_val, new_val, clause_sat_count, unsat, unsat_set)
            new_unsat_count = len(unsat_set)
            delta = new_unsat_count - old_unsat_count

            if delta <= 0 or rng.random() < math.exp(-delta / max(t0, 1e-9)):
                pass  # accept
            else:
                a[v] = old_val
                update_state(v, new_val, old_val, clause_sat_count, unsat, unsat_set)

            t0 *= 0.9995
        return a

    def polish(a, max_iters=200):
        """
        High-precision polish: iteratively find the single flip that most reduces
        the soft objective (sum of min(1, #unsat_lits) over clauses).
        Uses a smooth surrogate: for each clause, cost = min(1.0, unsat_count/3.0)
        which is differentiable-ish and encourages partial progress.
        """
        clause_sat_count, unsat = init_state(a)
        unsat_set = set(unsat)
        cur_unsat = len(unsat_set)

        for it in range(max_iters):
            if time.time() > deadline:
                return a
            if not unsat:
                return a

            # Try each variable flip, pick the one that minimizes soft objective
            best_v = -1
            best_soft = None
            best_val = None

            # Candidate variables: those appearing in unsat clauses
            candidate_vars = set()
            for ci in unsat:
                for lit in clauses[ci]:
                    candidate_vars.add(abs(lit) - 1)
            if not candidate_vars:
                break

            # Limit candidates for speed
            candidates = list(candidate_vars)
            if len(candidates) > 60:
                rng.shuffle(candidates)
                candidates = candidates[:60]

            for v in candidates:
                for val in (True, False):
                    if a[v] == val:
                        continue
                    old_val = a[v]
                    a[v] = val
                    update_state(v, old_val, val, clause_sat_count, unsat, unsat_set)
                    u = len(unsat_set)
                    # Soft objective: sum over unsat clauses of min(1, unsat_lits/3)
                    soft = 0.0
                    for ci in unsat:
                        cnt = clause_sat_count[ci]
                        if cnt == 0:
                            soft += 1.0
                        else:
                            soft += min(1.0, (3 - cnt) / 3.0)
                    # Revert
                    a[v] = old_val
                    update_state(v, val, old_val, clause_sat_count, unsat, unsat_set)

                    if best_soft is None or soft < best_soft:
                        best_soft = soft
                        best_v = v
                        best_val = val

            if best_v >= 0:
                old_val = a[best_v]
                a[best_v] = best_val
                update_state(best_v, old_val, best_val, clause_sat_count, unsat, unsat_set)
                cur_unsat = len(unsat_set)
                if cur_unsat == 0:
                    return a
            else:
                break

        return a

    # Main loop with restarts
    restart = 0
    base_steps = 8000
    no_improve_limit = 2000
    while time.time() < deadline:
        a = [rng.random() < 0.5 for _ in range(n)]
        max_steps = base_steps + restart * 500
        a, u, converged = walk_sat(a, max_steps, no_improve_limit)
        if u < best_unsat:
            best_unsat = u
            best_assignment = a[:]
        if u == 0:
            return a

        # If not converged, try SA escape
        if time.time() < deadline - 10.0:
            a2 = a[:]
            a2 = sa_escape(a2, 3000, 5.0)
            u2 = update_best(a2)
            if u2 == 0:
                return a2

        restart += 1
        if restart > 400:
            break

    # Final: polish from best assignment
    if best_assignment is not None and time.time() < deadline - 2.0:
        a = best_assignment[:]
        a = polish(a, max_iters=300)
        u = update_best(a)
        if u == 0:
            return a

    # Fallback: greedy repair
    if best_assignment is not None and time.time() < deadline - 1.0:
        a = best_assignment[:]
        clause_sat_count, unsat = init_state(a)
        unsat_set = set(unsat)
        for _ in range(200):
            if time.time() > deadline:
                break
            if not unsat:
                return a
            best_v = -1
            best_gain = 0
            best_val = None
            candidates = list(range(n))
            rng.shuffle(candidates)
            for v in candidates[:100]:
                for val in (True, False):
                    if a[v] == val:
                        continue
                    old_val = a[v]
                    a[v] = val
                    update_state(v, old_val, val, clause_sat_count, unsat, unsat_set)
                    u = len(unsat_set)
                    gain = best_unsat - u
                    a[v] = old_val
                    update_state(v, val, old_val, clause_sat_count, unsat, unsat_set)
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                        best_val = val
            if best_v >= 0:
                old_val = a[best_v]
                a[best_v] = best_val
                update_state(best_v, old_val, best_val, clause_sat_count, unsat, unsat_set)
                update_best(a)
            else:
                break

    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
