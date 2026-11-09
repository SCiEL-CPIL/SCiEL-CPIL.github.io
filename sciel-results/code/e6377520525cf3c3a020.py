"""WalkSAT + DLS hybrid with adaptive restarts for planted 3-SAT."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 280  # safety margin before 300s kill

    # Precompute: for each variable, which clauses contain it (and with what sign)
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var_clauses[abs(lit) - 1].append((ci, lit > 0))

    def make_satisfied_array(assignment):
        """Return a boolean array: satisfied[ci] = True if clause ci is satisfied."""
        sat = [False] * m
        for ci, clause in enumerate(clauses):
            for lit in clause:
                if assignment[abs(lit) - 1] == (lit > 0):
                    sat[ci] = True
                    break
        return sat

    def count_unsat(sat):
        return sum(1 for s in sat if not s)

    def flip_var(assignment, var_idx, sat):
        """Flip variable var_idx and update sat array incrementally."""
        assignment[var_idx] = not assignment[var_idx]
        for ci, is_pos in var_clauses[var_idx]:
            if is_pos == assignment[var_idx]:
                # This literal now satisfies the clause
                sat[ci] = True
            else:
                # This literal no longer satisfies; check if others do
                clause = clauses[ci]
                still_sat = False
                for lit in clause:
                    if lit == var_idx + 1 or lit == -(var_idx + 1):
                        continue
                    if assignment[abs(lit) - 1] == (lit > 0):
                        still_sat = True
                        break
                sat[ci] = still_sat

    def walksat_step(assignment, sat, rng, p_walk):
        """One WalkSAT step: pick an unsat clause, flip a variable.
        With probability p_walk, flip the variable in the clause that
        maximizes gains (WalkSAT heuristic); otherwise flip randomly.
        Returns number of newly satisfied clauses (gains)."""
        unsat = [ci for ci in range(m) if not sat[ci]]
        if not unsat:
            return 0
        ci = rng.choice(unsat)
        clause = clauses[ci]
        if rng.random() < p_walk:
            # WalkSAT: pick variable with best gain/loss ratio
            best_var = -1
            best_score = -1
            for lit in clause:
                v = abs(lit) - 1
                if assignment[v] == (lit > 0):
                    # Flipping would break this clause's only satisfied literal
                    # (if it's the only one) - compute net effect
                    gains = 0
                    losses = 0
                    for c2, is_pos2 in var_clauses[v]:
                        if c2 == ci:
                            continue
                        if is_pos2 == assignment[v]:
                            # Currently satisfies c2; flipping loses it
                            # Check if c2 has other satisfied literals
                            clause2 = clauses[c2]
                            other_sat = False
                            for lit2 in clause2:
                                if lit2 == lit or lit2 == -lit:
                                    continue
                                if assignment[abs(lit2) - 1] == (lit2 > 0):
                                    other_sat = True
                                    break
                            if not other_sat:
                                losses += 1
                        else:
                            # Flipping would satisfy c2
                            gains += 1
                    score = gains - losses
                    if score > best_score:
                        best_score = score
                        best_var = v
                else:
                    # Variable doesn't currently satisfy this clause
                    gains = 0
                    losses = 0
                    for c2, is_pos2 in var_clauses[v]:
                        if c2 == ci:
                            continue
                        if is_pos2 == assignment[v]:
                            # Currently satisfies c2; flipping loses it
                            clause2 = clauses[c2]
                            other_sat = False
                            for lit2 in clause2:
                                if lit2 == lit or lit2 == -lit:
                                    continue
                                if assignment[abs(lit2) - 1] == (lit2 > 0):
                                    other_sat = True
                                    break
                            if not other_sat:
                                losses += 1
                        else:
                            # Flipping would satisfy c2
                            gains += 1
                    score = gains - losses
                    if score > best_score:
                        best_score = score
                        best_var = v
            if best_var >= 0:
                flip_var(assignment, best_var, sat)
            else:
                lit = rng.choice(clause)
                flip_var(assignment, abs(lit) - 1, sat)
        else:
            lit = rng.choice(clause)
            flip_var(assignment, abs(lit) - 1, sat)
        return 0

    def dls_step(assignment, sat, rng):
        """DLS step: pick the shortest unsatisfied clause, flip a random var in it."""
        unsat = [ci for ci in range(m) if not sat[ci]]
        if not unsat:
            return
        # Find shortest unsat clause
        min_len = 4
        best_ci = unsat[0]
        for ci in unsat:
            l = len(clauses[ci])
            if l < min_len:
                min_len = l
                best_ci = ci
                if l <= 1:
                    break
        clause = clauses[best_ci]
        lit = rng.choice(clause)
        flip_var(assignment, abs(lit) - 1, sat)

    best_assignment = None
    best_unsat = m + 1

    for restart in range(200):
        if time.time() > deadline:
            break
        rng = random.Random(restart * 7919 + 13)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        sat = make_satisfied_array(assignment)
        cur_unsat = count_unsat(sat)
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
        if cur_unsat == 0:
            return assignment

        steps_in_restart = 0
        max_steps = 5000
        dls_interval = 100  # do DLS every 100 steps
        stall_count = 0
        prev_unsat = cur_unsat
        p_walk = 0.6

        for step in range(max_steps):
            if time.time() > deadline:
                if cur_unsat < best_unsat:
                    best_unsat = cur_unsat
                    best_assignment = assignment[:]
                return best_assignment if best_assignment else assignment

            # WalkSAT step
            walksat_step(assignment, sat, rng, p_walk)
            steps_in_restart += 1

            # Periodic DLS for escape
            if steps_in_restart % dls_interval == 0:
                dls_step(assignment, sat, rng)

            # Check progress
            if step % 50 == 0:
                cur_unsat = count_unsat(sat)
                if cur_unsat == 0:
                    return assignment
                if cur_unsat < best_unsat:
                    best_unsat = cur_unsat
                    best_assignment = assignment[:]
                if cur_unsat < prev_unsat:
                    stall_count = 0
                    # Slightly decrease p_walk when making progress
                    p_walk = max(0.3, p_walk - 0.02)
                else:
                    stall_count += 1
                prev_unsat = cur_unsat

            # Adaptive restart within this restart attempt
            if stall_count > 20:
                break

    return best_assignment if best_assignment else [False] * n
# EVOLVE-BLOCK-END
