"""Improved planted 3-SAT solver: WalkSAT with weighted flips, adaptive noise,
and an explicit wall-clock deadline guard that returns the best assignment found."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]

    # Wall-clock deadline with a safety margin (evaluator kills at 330s).
    start = time.time()
    deadline = start + 280.0  # 50s safety margin

    rng = random.Random(0)
    assignment = [rng.random() < 0.5 for _ in range(n)]

    # Precompute clause literal indices for speed.
    # For each clause, store list of (var_index, is_positive).
    clause_vars = []
    for c in clauses:
        clause_vars.append([(abs(l) - 1, l > 0) for l in c])

    # Count how many times each variable appears in each polarity across clauses.
    # pos_count[v] = number of clauses where +v appears
    # neg_count[v] = number of clauses where -v appears
    pos_count = [0] * n
    neg_count = [0] * n
    for c in clause_vars:
        for v, pos in c:
            if pos:
                pos_count[v] += 1
            else:
                neg_count[v] += 1

    def is_satisfied_clause(ci, assign):
        c = clause_vars[ci]
        for v, pos in c:
            if assign[v] == pos:
                return True
        return False

    def count_unsat(assign):
        cnt = 0
        for ci in range(len(clause_vars)):
            if not is_satisfied_clause(ci, assign):
                cnt += 1
        return cnt

    best_assign = assignment[:]
    best_unsat = count_unsat(assignment)
    if best_unsat == 0:
        return best_assign

    # WalkSAT main loop with adaptive noise and weighted literal selection.
    max_steps = 200000
    steps = 0
    noise = 0.2  # probability of random flip; otherwise weighted
    min_noise = 0.05
    max_noise = 0.5
    patience = 0
    best_patience = 2000

    while steps < max_steps:
        # Check deadline
        if time.time() >= deadline:
            break

        # Find all unsatisfied clauses
        unsat = []
        for ci in range(len(clause_vars)):
            if not is_satisfied_clause(ci, assignment):
                unsat.append(ci)

        if not unsat:
            best_assign = assignment[:]
            best_unsat = 0
            break

        # Track best
        cur_unsat = len(unsat)
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assign = assignment[:]
            patience = 0
        else:
            patience += 1

        # Adaptive noise: increase when stuck, decrease when improving
        if patience > best_patience:
            noise = min(noise + 0.05, max_noise)
            patience = 0
        elif cur_unsat < best_unsat:
            noise = max(noise - 0.05, min_noise)

        # Pick a random unsatisfied clause
        ci = unsat[rng.randrange(len(unsat))]
        c = clause_vars[ci]

        # Weighted literal selection: weight = number of clauses the literal
        # would satisfy if flipped (i.e., clauses currently falsified that contain
        # the opposite literal). We approximate by counting clauses where flipping
        # this literal makes it true and the clause is currently unsat.
        # For speed, use a simpler heuristic: weight proportional to how many
        # times this literal appears in unsat clauses (precomputed per variable).
        # But we need per-literal-in-unsat-clause count. Compute on the fly for
        # the 3 literals of this clause.
        weights = []
        for v, pos in c:
            # Flipping v: if currently assign[v] == pos, flipping makes it False
            # (bad, since we need it True to satisfy). If assign[v] != pos,
            # flipping makes it True (good).
            # Weight: number of other unsat clauses that contain the literal
            # (v, pos) — i.e., clauses that would become satisfied.
            if assign[v] == pos:
                # Flipping would make this literal false; weight is low.
                w = 1.0
            else:
                # Flipping would make this literal true; count how many unsat
                # clauses contain (v, pos).
                w = 1.0
                for uj in unsat:
                    if uj == ci:
                        continue
                    for vv, pp in clause_vars[uj]:
                        if vv == v and pp == pos:
                            w += 1.0
                            break
            weights.append(w)

        # Choose literal with probability proportional to weight
        total_w = sum(weights)
        r = rng.random() * total_w
        cum = 0.0
        chosen = 0
        for i, w in enumerate(weights):
            cum += w
            if r <= cum:
                chosen = i
                break
        chosen = chosen % len(c)

        # With probability noise, flip a random literal instead
        if rng.random() < noise:
            chosen = rng.randrange(len(c))

        v, pos = c[chosen]
        assignment[v] = not assignment[v]
        steps += 1

    # Final check
    if best_unsat > 0:
        # One more quick check on best_assign
        fu = count_unsat(best_assign)
        if fu == 0:
            return best_assign

    return best_assign
# EVOLVE-BLOCK-END
