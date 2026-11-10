"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    # Improved WalkSAT with:
    # 1) Time-bounded loop with explicit deadline guard
    # 2) Better variable selection: pick the literal in the most-conflicting clause
    #    that appears in the most unsatisfied clauses (min-ones heuristic variant)
    # 3) Periodic restarts with random re-initialization to escape local minima
    # 4) Track best (fewest unsat) assignment seen and return it if not solved
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 280  # safety margin before 330s kill

    # Precompute clause sizes for speed
    rng = random.Random(0)

    def satisfied_fast(assignment, clause):
        for l in clause:
            if assignment[abs(l) - 1] == (l > 0):
                return True
        return False

    def count_unsat(assignment, clauses):
        cnt = 0
        for c in clauses:
            ok = False
            for l in c:
                if assignment[abs(l) - 1] == (l > 0):
                    ok = True
                    break
            if not ok:
                cnt += 1
        return cnt

    best_assignment = None
    best_unsat = float('inf')

    max_restarts = 500
    steps_per_restart = 5000

    for restart in range(max_restarts):
        if time.time() >= deadline:
            break

        assignment = [rng.random() < 0.5 for _ in range(n)]
        unsat_count = count_unsat(assignment, clauses)

        if unsat_count == 0:
            return assignment

        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]

        for step in range(steps_per_restart):
            if time.time() >= deadline:
                break

            # Find unsatisfied clauses
            unsat_clauses = []
            for c in clauses:
                ok = False
                for l in c:
                    if assignment[abs(l) - 1] == (l > 0):
                        ok = True
                        break
                if not ok:
                    unsat_clauses.append(c)

            if not unsat_clauses:
                return assignment

            if len(unsat_clauses) < best_unsat:
                best_unsat = len(unsat_clauses)
                best_assignment = assignment[:]

            # Pick a random unsatisfied clause
            clause = unsat_clauses[rng.randrange(len(unsat_clauses))]

            # For each literal in the clause, count how many unsat clauses it appears in
            # Choose the literal with the highest count (min-ones style: flip it to satisfy this clause
            # while breaking as few others as possible)
            best_lit = None
            best_score = -1
            for l in clause:
                var = abs(l) - 1
                # Count how many unsat clauses contain this variable
                cnt = 0
                for uc in unsat_clauses:
                    for ul in uc:
                        if abs(ul) - 1 == var:
                            cnt += 1
                            break
                # We want to flip the variable that appears in the fewest unsat clauses
                # (so we break the least), but also prefer literals that satisfy this clause
                # Since all literals in 'clause' are currently unsatisfied, flipping any will satisfy it.
                # So pick the one appearing in fewest other unsat clauses.
                score = -cnt  # higher is better (fewer conflicts)
                if score > best_score:
                    best_score = score
                    best_lit = l

            if best_lit is None:
                best_lit = rng.choice(clause)

            # With some probability, do a random flip instead (to escape plateaus)
            if rng.random() < 0.1:
                # Pick random literal from the clause
                best_lit = rng.choice(clause)

            assignment[abs(best_lit) - 1] = not assignment[abs(best_lit) - 1]

    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
