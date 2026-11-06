"""Synthetic planted 3-SAT solver (150 vars) — refined local search."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Deadline guard: leave a safety margin before the 300s wall-clock kill.
    start = time.time()
    DEADLINE = 300.0 - 15.0  # 15s safety margin

    # Precompute clause literals as (var_index, polarity_bool) 0-based.
    # polarity True means the literal is positive (var must be True).
    cl = []
    for c in clauses:
        cl.append([(abs(l) - 1, l > 0) for l in c])

    # ---- Polarity-based initialization ----
    # Count positive vs negative occurrences per variable.
    pos = [0] * n
    neg = [0] * n
    for c in cl:
        for (v, p) in c:
            if p:
                pos[v] += 1
            else:
                neg[v] += 1
    # Set each variable to its majority polarity; ties broken randomly.
    rng = random.Random(0)
    assignment = []
    for v in range(n):
        if pos[v] > neg[v]:
            assignment.append(True)
        elif neg[v] > pos[v]:
            assignment.append(False)
        else:
            assignment.append(rng.random() < 0.5)

    # ---- Fast evaluation helpers ----
    def count_unsat(assn):
        cnt = 0
        for c in cl:
            ok = False
            for (v, p) in c:
                if assn[v] == p:
                    ok = True
                    break
            if not ok:
                cnt += 1
        return cnt

    def is_satisfied(assn):
        for c in cl:
            ok = False
            for (v, p) in c:
                if assn[v] == p:
                    ok = True
                    break
            if not ok:
                return False
        return True

    # ---- Targeted repair with variable scoring ----
    # For an unsatisfied clause, the best flip is the variable whose flip
    # satisfies the most currently-unsatisfied clauses (i.e., resolves the
    # most conflicts). We compute, for each variable, the number of unsat
    # clauses that would be satisfied if that variable were set to satisfy
    # them, weighted by clause frequency.
    def best_flip_score(assn):
        # Build list of unsat clauses.
        unsat = []
        for c in cl:
            ok = False
            for (v, p) in c:
                if assn[v] == p:
                    ok = True
                    break
            if not ok:
                unsat.append(c)
        if not unsat:
            return None
        # For each variable, count how many unsat clauses it appears in with
        # the polarity that would satisfy them (i.e., the literal is currently
        # False, so flipping the var to match polarity satisfies the clause).
        score = [0] * n
        for c in unsat:
            for (v, p) in c:
                # In an unsat clause, no literal is satisfied, so assn[v] != p
                # for every literal. Flipping v to p would satisfy this clause.
                score[v] += 1
        # Choose variable with max score; random tie-break via rng.
        best = -1
        best_val = -1
        for v in range(n):
            s = score[v]
            if s > best_val:
                best_val = s
                best = v
        return best

    # ---- Main search loop with adaptive restarts ----
    best_assign = assignment[:]
    best_unsat = count_unsat(best_assign)
    if best_unsat == 0:
        return best_assign

    no_progress = 0
    max_no_progress = 3000
    steps = 0
    max_steps = 400000

    while True:
        if time.time() > DEADLINE:
            break
        if steps >= max_steps:
            break

        # Evaluate current.
        cur_unsat = count_unsat(assignment)
        if cur_unsat == 0:
            return assignment[:]
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assign = assignment[:]
            no_progress = 0
        else:
            no_progress += 1

        if no_progress >= max_no_progress:
            # Adaptive restart: perturb from best, not full random.
            assignment = best_assign[:]
            # Flip a small random fraction to escape the local optimum.
            k = max(1, n // 20)
            for _ in range(k):
                v = rng.randrange(n)
                assignment[v] = not assignment[v]
            no_progress = 0
            continue

        # Choose a flip via targeted scoring.
        # With some probability use pure random (WalkSAT noise) to escape
        # plateaus; otherwise use the scoring heuristic.
        if rng.random() < 0.15:
            # Random walk: pick a random unsat clause, flip a random literal.
            # Find a random unsat clause quickly.
            # We'll just pick a random clause and, if unsat, flip random var.
            for _ in range(3):
                c = cl[rng.randrange(m)]
                ok = False
                for (v, p) in c:
                    if assignment[v] == p:
                        ok = True
                        break
                if not ok:
                    v = c[rng.randrange(len(c))][0]
                    assignment[v] = not assignment[v]
                    break
            else:
                # Fallback: flip random var.
                v = rng.randrange(n)
                assignment[v] = not assignment[v]
        else:
            v = best_flip_score(assignment)
            if v is None:
                return assignment[:]
            # Flip the variable. To satisfy the clauses it appears in, set it
            # to the polarity that satisfies the most; but simplest: flip.
            assignment[v] = not assignment[v]

        steps += 1

    # Return best found.
    return best_assign
# EVOLVE-BLOCK-END
