"""Rewritten solver: efficient bookkeeping WalkSAT with weighted flips + restarts."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    rng = random.Random(12345)
    start_time = time.monotonic()
    time_limit = 290.0  # stay safely under 300 s

    # Precompute for each variable the list of (clause_index, sign) it appears in.
    # sign: +1 means the clause is satisfied by var being True, -1 by var being False.
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_clauses[v].append((ci, sign))

    m = len(clauses)

    def make_initial_assign():
        # Phase-biased random init.
        return [rng.random() < 0.5 for _ in range(n)]

    def build_false_counts(assignment):
        """Return list of false-count per clause (number of literals false)."""
        fc = [0] * m
        for ci, clause in enumerate(clauses):
            c = 0
            for lit in clause:
                v = abs(lit) - 1
                if assignment[v] != (lit > 0):
                    c += 1
            fc[ci] = c
        return fc

    def flip(var, assignment, fc):
        """Flip variable `var`, update fc in place. Returns old value."""
        old = assignment[var]
        new = not old
        assignment[var] = new
        for ci, sign in var_clauses[var]:
            # If sign==+1: clause satisfied when var True. Flipping changes its contribution.
            # old contribution to false: (old != (sign>0)) i.e. old != True if sign>0
            # We can just recompute the delta:
            # For sign>0: literal is var. False when var is False.
            #   old false? old == False -> if old is False, was false; new is True -> not false. delta -1
            #   if old True -> was not false; new False -> false. delta +1
            # For sign<0: literal is not var. False when var is True.
            #   old True -> was false; new False -> not false. delta -1
            #   old False -> was not false; new True -> false. delta +1
            if sign > 0:
                if old:
                    fc[ci] += 1
                else:
                    fc[ci] -= 1
            else:
                if old:
                    fc[ci] -= 1
                else:
                    fc[ci] += 1

    def walksat_once(assignment, max_iters):
        fc = build_false_counts(assignment)
        # Track unsatisfied clause indices for quick access.
        unsat = [ci for ci in range(m) if fc[ci] == 3]
        # WalkSAT parameters
        noise = 0.55  # probability of random pick among the clause's literals
        for it in range(max_iters):
            if not unsat:
                return assignment, True
            if time.monotonic() - start_time > time_limit:
                return assignment, False
            # Pick an unsatisfied clause, bias toward those with more "activity" (just random for simplicity)
            ci = unsat[rng.randrange(len(unsat))]
            clause = clauses[ci]
            # Score each literal in the clause by how many other currently-false clauses it would satisfy.
            best_score = -1
            best_lit = None
            scored = []
            for lit in clause:
                v = abs(lit) - 1
                # Compute gain: number of clauses (other than ci) that are currently false
                # and would become satisfied if we set var to the value lit demands.
                gain = 0
                target = lit > 0
                for c2, sign in var_clauses[v]:
                    if c2 == ci:
                        continue
                    if fc[c2] > 0:  # currently not fully satisfied
                        # Would this flip satisfy c2? c2 is satisfied by this var if (assignment matches sign)
                        # After flip, var = target. c2's literal on var is satisfied iff target == (sign>0).
                        if target == (sign > 0):
                            gain += 1
                scored.append((gain, lit))
            if rng.random() < noise:
                _, lit = scored[rng.randrange(len(scored))]
            else:
                # Pick max gain (break ties randomly)
                maxg = max(g for g, _ in scored)
                candidates = [l for g, l in scored if g == maxg]
                lit = candidates[rng.randrange(len(candidates))]
            v = abs(lit) - 1
            # Flip and update fc and unsat list
            # Determine which clauses change status
            old = assignment[v]
            new = not old
            # Update fc via flip
            flip(v, assignment, fc)
            # Rebuild unsat incrementally: remove ci if now satisfied, add any that became unsat
            # Simpler: rebuild unsat occasionally; but for correctness we need accurate unsat.
            # We'll do a lightweight update: scan var_clauses[v] to update membership.
            # To keep it simple and correct, we rebuild unsat when it's small relative to m,
            # else do a targeted update. Given n=150, m is moderate; rebuild is cheap enough.
            unsat = [c for c in range(m) if fc[c] == 3]
        return assignment, (len(unsat) == 0)

    # Multi-restart loop
    best_assignment = None
    restart_budget_iters = 15000
    max_restarts = 60
    for r in range(max_restarts):
        if time.monotonic() - start_time > time_limit:
            break
        assignment = make_initial_assign()
        assignment, ok = walksat_once(assignment, restart_budget_iters)
        if ok:
            return assignment
        best_assignment = assignment
        # Slightly vary noise per restart for diversity
        noise = 0.4 + 0.3 * rng.random()
    return best_assignment
# EVOLVE-BLOCK-END
