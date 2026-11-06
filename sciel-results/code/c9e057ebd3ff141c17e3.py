"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280  # 300s limit, keep 20s safety margin

    # Preprocess: build clause literals as (var, is_positive) 0-indexed
    # clauses[i] is list of signed 1-based ints
    # We'll work with 0-indexed variables internally

    # Convert clauses to (var_idx, sign) where sign is +1 or -1
    # var_idx is 0-based
    cls = []
    for c in clauses:
        cls.append([(abs(l) - 1, 1 if l > 0 else -1) for l in c])

    num_clauses = len(cls)

    # Build variable-to-clauses index for efficiency
    var_clauses = [[] for _ in range(n)]
    for i, c in enumerate(cls):
        for var, sign in c:
            var_clauses[var].append((i, sign))

    # Weighted WalkSAT with restarts
    # Use a global random seed for reproducibility
    rng = random.Random(42)

    def evaluate(assignment):
        """Return number of unsatisfied clauses."""
        unsat_count = 0
        for c in cls:
            sat = False
            for var, sign in c:
                if assignment[var] == (sign == 1):
                    sat = True
                    break
            if not sat:
                unsat_count += 1
        return unsat_count

    def is_satisfied(assignment):
        for c in cls:
            sat = False
            for var, sign in c:
                if assignment[var] == (sign == 1):
                    sat = True
                    break
            if not sat:
                return False
        return True

    def walksat_run(assignment, max_steps):
        """Run WalkSAT from current assignment for up to max_steps.
        Returns (assignment, solved)."""
        # Compute initial unsat count
        unsat_vars = {}  # var -> count of unsatisfied clauses containing var
        # Build unsat clause set
        unsat_clauses = []
        for i, c in enumerate(cls):
            sat = False
            for var, sign in c:
                if assignment[var] == (sign == 1):
                    sat = True
                    break
            if not sat:
                unsat_clauses.append(i)

        if not unsat_clauses:
            return assignment, True

        # Compute weight for each variable: number of unsat clauses it appears in
        weight = [0] * n
        for ci in unsat_clauses:
            for var, sign in cls[ci]:
                weight[var] += 1

        # Track unsat clause count for fast updates
        unsat_count = len(unsat_clauses)

        # Use a set for fast membership
        unsat_set = set(unsat_clauses)

        p = 0.55  # probability to flip a literal in the clause vs random

        for step in range(max_steps):
            if time.time() > deadline - 5:
                break

            if unsat_count == 0:
                return assignment, True

            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            clause = cls[ci]

            if p < rng.random():
                # Flip a literal in the clause (weighted by weight)
                total_w = sum(weight[var] for var, sign in clause)
                if total_w == 0:
                    var, sign = rng.choice(clause)
                else:
                    r = rng.random() * total_w
                    cum = 0
                    var, sign = clause[0]
                    for v, s in clause:
                        cum += weight[v]
                        if cum >= r:
                            var, sign = v, s
                            break
            else:
                # Flip a random variable
                var, sign = rng.choice(clause)

            # Flip variable
            old_val = assignment[var]
            assignment[var] = not old_val

            # Update unsat status for clauses containing this variable
            for (ci2, sign2) in var_clauses[var]:
                clause2 = cls[ci2]
                # Check if clause2 is now satisfied
                sat = False
                for var2, sign3 in clause2:
                    if assignment[var2] == (sign3 == 1):
                        sat = True
                        break

                was_unsat = ci2 in unsat_set
                is_unsat = not sat

                if was_unsat and not is_unsat:
                    # Became satisfied
                    unsat_set.discard(ci2)
                    unsat_count -= 1
                    # Remove from unsat_clauses list (lazy: we'll filter later)
                    for var2, s2 in clause2:
                        weight[var2] -= 1
                elif not was_unsat and is_unsat:
                    # Became unsatisfied
                    unsat_set.add(ci2)
                    unsat_count += 1
                    for var2, s2 in clause2:
                        weight[var2] += 1

            # Periodically clean up unsat_clauses list to match unsat_set
            if step % 100 == 0:
                unsat_clauses = [c for c in unsat_clauses if c in unsat_set]

        return assignment, unsat_count == 0

    # Try multiple restarts with different strategies
    best_assignment = None
    best_unsat = num_clauses + 1

    # First: try a simple heuristic - unit propagation style initial assignment
    # Set variables to satisfy as many clauses as possible initially
    assignment = [False] * n
    # For each variable, count how many clauses would be satisfied by True vs False
    for v in range(n):
        true_count = 0
        false_count = 0
        for (ci, sign) in var_clauses[v]:
            if sign == 1:
                true_count += 1
            else:
                false_count += 1
        assignment[v] = true_count >= false_count

    if is_satisfied(assignment):
        return assignment

    # Now run WalkSAT with restarts
    max_steps_per_run = 5000
    num_restarts = 50

    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break

        # New random assignment for each restart
        if restart == 0:
            # Use the heuristic assignment
            current = assignment[:]
        else:
            current = [rng.random() < 0.5 for _ in range(n)]

        # Run WalkSAT
        current, solved = walksat_run(current, max_steps_per_run)

        if solved:
            return current

        # Track best
        u = evaluate(current)
        if u < best_unsat:
            best_unsat = u
            best_assignment = current[:]

    # If we have a best assignment, try a final polish
    if best_assignment is not None and best_unsat < num_clauses:
        if time.time() < deadline - 5:
            best_assignment, solved = walksat_run(best_assignment, 20000)
            if solved:
                return best_assignment

    # Return best found or last attempt
    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
