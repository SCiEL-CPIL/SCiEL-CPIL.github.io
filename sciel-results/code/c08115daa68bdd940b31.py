"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 280  # safety margin before 300s kill

    # Precompute clause literal info: for each clause, list of (var_index, sign)
    # sign: +1 if literal is positive, -1 if negative
    clause_info = []
    for c in clauses:
        lits = [(abs(l) - 1, 1 if l > 0 else -1) for l in c]
        clause_info.append(lits)

    # For incremental tracking: for each variable, list of (clause_index, sign)
    # where sign is +1 if the variable appears positively in that clause, -1 if negatively
    var_to_clauses = [[] for _ in range(n)]
    for ci, lits in enumerate(clause_info):
        for var_idx, sign in lits:
            var_to_clauses[var_idx].append((ci, sign))

    # Track: for each clause, how many literals are currently satisfied
    # satisfied_count[ci] = number of satisfied literals in clause ci

    def init_satisfied_count(assignment):
        counts = [0] * m
        for ci, lits in enumerate(clause_info):
            cnt = 0
            for var_idx, sign in lits:
                if assignment[var_idx] == (sign == 1):
                    cnt += 1
            counts[ci] = cnt
        return counts

    def flip_var(var_idx, assignment, sat_counts):
        """Flip variable var_idx and update sat_counts incrementally."""
        old_val = assignment[var_idx]
        new_val = not old_val
        for ci, sign in var_to_clauses[var_idx]:
            # Before flip: was this literal satisfied?
            was_sat = old_val == (sign == 1)
            # After flip: will this literal be satisfied?
            is_sat = new_val == (sign == 1)
            if was_sat and not is_sat:
                sat_counts[ci] -= 1
            elif not was_sat and is_sat:
                sat_counts[ci] += 1
        assignment[var_idx] = new_val

    def count_unsat(sat_counts):
        return sum(1 for c in sat_counts if c == 0)

    def get_unsat_clauses(sat_counts):
        return [i for i in range(m) if sat_counts[i] == 0]

    def is_solved(sat_counts):
        return all(c > 0 for c in sat_counts)

    best_assignment = None
    best_unsat_count = m + 1

    # Multiple restarts with adaptive step budgets
    max_restarts = 50
    base_steps = 2000
    rng = random.Random(42)

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Initialize assignment: random, biased slightly
        assignment = [rng.random() < 0.5 for _ in range(n)]
        sat_counts = init_satisfied_count(assignment)
        unsat = count_unsat(sat_counts)

        if unsat == 0:
            return assignment

        if unsat < best_unsat_count:
            best_unsat_count = unsat
            best_assignment = assignment[:]

        # Adaptive step budget: more steps for harder instances
        step_budget = base_steps + restart * 500
        step_budget = min(step_budget, 50000)

        # WalkSAT with damage-minimizing flip selection
        for step in range(step_budget):
            if time.time() > deadline:
                break

            unsat_list = get_unsat_clauses(sat_counts)
            if not unsat_list:
                return assignment

            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_list)
            lits = clause_info[ci]

            # Damage-minimizing: for each literal in the clause, compute the damage
            # (number of satisfied clauses that would become unsatisfied if we flip this var)
            best_var = -1
            best_damage = float('inf')
            best_gain = -1

            for var_idx, sign in lits:
                # Compute damage: how many currently-satisfied clauses contain this var
                # with the same polarity (i.e., would become unsatisfied if flipped)
                damage = 0
                for cj, s in var_to_clauses[var_idx]:
                    if cj == ci:
                        continue
                    if sat_counts[cj] == 1:  # only one satisfied literal in that clause
                        # Check if this variable is the one satisfying it
                        if assignment[var_idx] == (s == 1):
                            damage += 1
                # Gain: number of currently-unsatisfied clauses this flip would satisfy
                gain = 0
                for cj, s in var_to_clauses[var_idx]:
                    if cj == ci:
                        continue
                    if sat_counts[cj] == 0:
                        if not assignment[var_idx] == (s == 1):
                            gain += 1

                # Score: minimize damage, maximize gain
                score = damage - gain
                if score < best_damage or (score == best_damage and gain > best_gain):
                    best_damage = score
                    best_gain = gain
                    best_var = var_idx

            if best_var >= 0:
                flip_var(best_var, assignment, sat_counts)

            # Check if solved
            # Instead of full scan, track unsat count incrementally
            # But for simplicity, do periodic checks
            if step % 10 == 0:
                cur_unsat = count_unsat(sat_counts)
                if cur_unsat == 0:
                    return assignment
                if cur_unsat < best_unsat_count:
                    best_unsat_count = cur_unsat
                    best_assignment = assignment[:]

    # Final check
    if best_assignment is not None:
        sat_counts = init_satisfied_count(best_assignment)
        if is_solved(sat_counts):
            return best_assignment

    return best_assignment if best_assignment is not None else [False] * n
# EVOLVE-BLOCK-END
