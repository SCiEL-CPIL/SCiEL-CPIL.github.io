"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 330s kill

    # Preprocess: unit propagation
    assignment = [None] * n  # None = unassigned
    unit_queue = []

    def propagate():
        """Unit propagation: assign literals forced by unit clauses."""
        changed = True
        while changed:
            changed = False
            # Build map of literal -> clauses containing it
            for clause in clauses:
                unassigned = [l for l in clause if assignment[abs(l) - 1] is None]
                satisfied = any(assignment[abs(l) - 1] == (l > 0) for l in clause)
                if satisfied:
                    continue
                if len(unassigned) == 0:
                    return False  # conflict
                if len(unassigned) == 1:
                    l = unassigned[0]
                    if assignment[abs(l) - 1] is None:
                        assignment[abs(l) - 1] = (l > 0)
                        changed = True
        return True

    if not propagate():
        # Conflict during propagation; reset and proceed with random init
        assignment = [rng.random() < 0.5 for _ in range(n)]
    else:
        # Fill unassigned variables randomly
        for i in range(n):
            if assignment[i] is None:
                assignment[i] = rng.random() < 0.5

    def is_satisfied(clause):
        return any(assignment[abs(l) - 1] == (l > 0) for l in clause)

    def count_unsat():
        return sum(1 for c in clauses if not is_satisfied(c))

    # Track unsatisfied clauses for efficiency
    def get_unsat_clauses():
        return [c for c in clauses if not is_satisfied(c)]

    # WalkSAT with gain/loss heuristic
    p = 0.55  # probability of using gain heuristic
    max_steps = 50000
    restarts = 0
    max_restarts = 50

    while time.time() < deadline and restarts < max_restarts:
        unsat = get_unsat_clauses()
        if not unsat:
            break

        for step in range(max_steps):
            if time.time() >= deadline:
                break
            unsat = get_unsat_clauses()
            if not unsat:
                break

            # Pick a random unsatisfied clause
            clause = rng.choice(unsat)

            # Compute gain/loss for each literal in clause
            best_var = -1
            best_gain = -1000
            best_is_gain = False

            for l in clause:
                var = abs(l) - 1
                # Count how many unsatisfied clauses would be satisfied
                # and how many satisfied clauses would become unsatisfied
                gain = 0
                loss = 0
                for c in clauses:
                    if is_satisfied(c):
                        # Would this flip make it unsatisfied?
                        # Only if the literal l is the only satisfied literal in c
                        has_l = any(abs(ll) - 1 == var for ll in c)
                        if has_l:
                            # Check if l is the only satisfied literal
                            only_sat = all(
                                (assignment[abs(ll) - 1] != (ll > 0)) or (abs(ll) - 1 == var)
                                for ll in c
                            )
                            if only_sat:
                                loss += 1
                    else:
                        # Would this flip satisfy it?
                        if any(abs(ll) - 1 == var and (ll > 0) == (not assignment[var]) for ll in c):
                            gain += 1

                net = gain - loss
                if net > best_gain:
                    best_gain = net
                    best_var = var
                    best_is_gain = gain > 0

            # With probability p, use the best gain variable; otherwise random
            if best_var >= 0 and (rng.random() < p or best_gain > 0):
                var_to_flip = best_var
            else:
                # Random variable from the clause
                l = rng.choice(clause)
                var_to_flip = abs(l) - 1

            assignment[var_to_flip] = not assignment[var_to_flip]

        # If we exhausted steps without solving, restart
        unsat = get_unsat_clauses()
        if not unsat:
            break
        restarts += 1
        # Reinitialize
        assignment = [rng.random() < 0.5 for _ in range(n)]
        # Try unit propagation again
        assignment = [None] * n
        if propagate():
            for i in range(n):
                if assignment[i] is None:
                    assignment[i] = rng.random() < 0.5
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]

    return assignment
# EVOLVE-BLOCK-END
