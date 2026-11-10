"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 290  # 290s safety margin under 330s limit

    # Precompute clause structure
    # For each clause, store the variable indices and signs
    # clause_vars[c] = list of (var_idx, sign) where sign is +1 or -1
    # clause_satisfied_count[c] = number of literals satisfied in clause c
    # var_occurrences[v] = list of (clause_idx, sign) for variable v

    clause_vars = []
    var_occurrences = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        lits = []
        for l in clause:
            vi = abs(l) - 1
            sign = 1 if l > 0 else -1
            lits.append((vi, sign))
            var_occurrences[vi].append((ci, sign))
        clause_vars.append(lits)

    def random_assignment():
        return [random.random() < 0.5 for _ in range(n)]

    def compute_clause_counts(assignment):
        """Compute for each clause how many literals are satisfied."""
        counts = [0] * m
        for ci in range(m):
            cnt = 0
            for vi, sign in clause_vars[ci]:
                if assignment[vi] == (sign == 1):
                    cnt += 1
            counts[ci] = cnt
        return counts

    def flip_variable(assignment, vi, counts):
        """Flip variable vi and update clause counts incrementally."""
        old_val = assignment[vi]
        new_val = not old_val
        assignment[vi] = new_val
        for ci, sign in var_occurrences[vi]:
            # Before flip: literal satisfied if assignment[vi] == (sign==1)
            # After flip: literal satisfied if new_val == (sign==1)
            was_sat = old_val == (sign == 1)
            now_sat = new_val == (sign == 1)
            if was_sat and not now_sat:
                counts[ci] -= 1
            elif not was_sat and now_sat:
                counts[ci] += 1

    def is_satisfied(counts):
        return all(c >= 1 for c in counts)

    def walk_sat(assignment, counts, max_steps, rng):
        """WalkSAT with incremental tracking. Returns (success, assignment, counts)."""
        for step in range(max_steps):
            if time.time() > deadline:
                return False, assignment, counts
            # Find unsatisfied clauses
            unsat = [ci for ci in range(m) if counts[ci] == 0]
            if not unsat:
                return True, assignment, counts

            # Pick a random unsatisfied clause
            ci = rng.choice(unsat)

            # Count how many literals in this clause would become satisfied if we flip each variable
            # We want to flip a variable that makes this clause satisfied
            # without breaking too many other clauses
            best_vi = None
            best_score = None
            lits_in_clause = clause_vars[ci]

            # For each literal in the clause, check if flipping its variable satisfies the clause
            # and count how many other clauses would be broken
            for vi, sign in lits_in_clause:
                # Flipping vi will satisfy this literal (since clause is unsat, all lits are false)
                # Count how many other clauses containing vi would be broken
                break_count = 0
                for oci, osign in var_occurrences[vi]:
                    if oci == ci:
                        continue
                    # Current state of literal in clause oci
                    cur_sat = assignment[vi] == (osign == 1)
                    if cur_sat:
                        # This literal is currently satisfied in oci
                        # If we flip vi, it becomes unsatisfied
                        # Only breaks oci if it was the only satisfied literal
                        if counts[oci] == 1:
                            break_count += 1
                # Score: minimize break_count, with some randomness
                score = break_count
                if best_score is None or score < best_score or (score == best_score and rng.random() < 0.5):
                    best_score = score
                    best_vi = vi

            if best_vi is not None:
                flip_variable(assignment, best_vi, counts)

        return False, assignment, counts

    def local_search(assignment, counts, max_steps, rng):
        """Local search with different heuristic: pick variable that satisfies most unsat clauses."""
        for step in range(max_steps):
            if time.time() > deadline:
                return False, assignment, counts
            unsat = [ci for ci in range(m) if counts[ci] == 0]
            if not unsat:
                return True, assignment, counts

            # For each variable, count how many unsat clauses it could satisfy
            var_score = [0] * n
            for ci in unsat:
                for vi, sign in clause_vars[ci]:
                    var_score[vi] += 1

            # Pick the variable with highest score (with some randomness)
            max_score = max(var_score) if var_score else 0
            if max_score == 0:
                # No variable can satisfy any unsat clause directly; random flip
                vi = rng.randrange(n)
            else:
                # Pick randomly from top-scoring variables
                candidates = [vi for vi in range(n) if var_score[vi] == max_score]
                vi = rng.choice(candidates)

            flip_variable(assignment, vi, counts)

        return False, assignment, counts

    # Main loop with restarts
    best_assignment = None
    best_unsat_count = m + 1

    for attempt in range(200):
        if time.time() > deadline:
            break

        rng = random.Random(attempt * 7919 + 13)
        assignment = random_assignment()
        counts = compute_clause_counts(assignment)

        if is_satisfied(counts):
            return assignment

        cur_unsat = sum(1 for c in counts if c == 0)
        if cur_unsat < best_unsat_count:
            best_unsat_count = cur_unsat
            best_assignment = assignment[:]

        # Alternate between WalkSAT and local search
        if attempt % 2 == 0:
            success, assignment, counts = walk_sat(assignment, counts, 5000, rng)
        else:
            success, assignment, counts = local_search(assignment, counts, 5000, rng)

        if success:
            return assignment

        cur_unsat = sum(1 for c in counts if c == 0)
        if cur_unsat < best_unsat_count:
            best_unsat_count = cur_unsat
            best_assignment = assignment[:]

    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
