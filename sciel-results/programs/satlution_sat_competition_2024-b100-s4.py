"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    # Improved WalkSAT with min-conflicts heuristic, restarts, and deadline guard.
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280.0  # 50s safety margin before 330s kill

    # Pre-extract clause literals as (var_idx, sign) for speed
    # var_idx is 0-based, sign is +1 for positive, -1 for negative
    clause_data = []
    for clause in clauses:
        clause_data.append([(abs(l) - 1, 1 if l > 0 else -1) for l in clause])

    num_clauses = len(clause_data)

    def count_unsat(assignment):
        """Count number of unsatisfied clauses."""
        count = 0
        for clause in clause_data:
            sat = False
            for var_idx, sign in clause:
                if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def is_satisfied(assignment):
        """Check if all clauses are satisfied."""
        for clause in clause_data:
            sat = False
            for var_idx, sign in clause:
                if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                    sat = True
                    break
            if not sat:
                return False
        return True

    # Min-conflicts heuristic: for an unsatisfied clause, find the literal whose
    # flip reduces the total number of unsatisfied clauses the most.
    # We approximate by counting how many clauses would be affected.
    # For speed, we use a simpler heuristic: pick the literal that appears in
    # the fewest unsatisfied clauses (or most, depending on direction).

    def walk_one(assignment, rng, max_steps):
        """Run WalkSAT from current assignment for up to max_steps."""
        # Build clause -> list of clause indices mapping for variable occurrences
        # For efficiency, we'll recompute unsat clauses each step but use a
        # scoring heuristic for literal selection.

        for step in range(max_steps):
            if time.time() > deadline:
                return assignment, False

            # Find unsatisfied clauses
            unsat_clauses = []
            for i, clause in enumerate(clause_data):
                sat = False
                for var_idx, sign in clause:
                    if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(i)

            if not unsat_clauses:
                return assignment, True

            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            clause = clause_data[ci]

            # Score each literal: prefer flipping the one that satisfies the most
            # other unsatisfied clauses (or minimizes new unsats).
            # Simple heuristic: for each literal in the clause, count how many
            # currently unsatisfied clauses would be satisfied if we flip that variable.
            # But that's expensive. Instead, use a weighted random choice where
            # literals that appear in more unsatisfied clauses are more likely to help.

            # Count occurrences of each variable in unsatisfied clauses
            var_unsat_count = {}
            for uc in unsat_clauses:
                for var_idx, sign in clause_data[uc]:
                    var_unsat_count[var_idx] = var_unsat_count.get(var_idx, 0) + 1

            # Weighted selection: higher weight for variables in more unsat clauses
            weights = []
            for var_idx, sign in clause:
                w = var_unsat_count.get(var_idx, 0) + 1
                weights.append(w)

            # Normalize and pick
            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0
            chosen_idx = 0
            for idx, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    chosen_idx = idx
                    break

            var_idx, sign = clause[chosen_idx]
            assignment[var_idx] = not assignment[var_idx]

        return assignment, False

    # Multi-restart strategy
    rng = random.Random(0)
    best_assignment = None
    best_unsat_count = float('inf')

    # Parameters
    max_steps_per_restart = 10000
    num_restarts = 500  # max restarts, will stop at deadline

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Try to solve
        assignment, solved = walk_one(assignment, rng, max_steps_per_restart)

        if solved:
            return assignment

        # Track best
        uc = count_unsat(assignment)
        if uc < best_unsat_count:
            best_unsat_count = uc
            best_assignment = assignment[:]

    # Return best assignment found
    if best_assignment is not None:
        return best_assignment

    # Fallback: all False
    return [False] * n

# EVOLVE-BLOCK-END
