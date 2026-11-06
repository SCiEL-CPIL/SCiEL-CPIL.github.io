"""Stronger planted 3-SAT solver: vectorized adaptive WalkSAT with incremental updates."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def _build_structures(n, clauses):
    """Precompute literal arrays for fast vectorized evaluation and variable-to-clause mapping."""
    m = len(clauses)
    if m == 0:
        return None, None, None
    lit_arr = np.empty((m, 3), dtype=np.int8)
    for i, c in enumerate(clauses):
        lit_arr[i, 0] = c[0]
        lit_arr[i, 1] = c[1]
        lit_arr[i, 2] = c[2]
    var_arr = np.abs(lit_arr) - 1  # 0-based variable index

    # Build var_to_clauses: for each variable, list of (clause_idx, position)
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for pos in range(3):
            v = var_arr[i, pos]
            var_to_clauses[v].append((i, pos))

    return lit_arr, var_arr, var_to_clauses


def _eval_clauses(assignment, lit_arr, var_arr):
    """Return boolean array: is each clause satisfied?"""
    a = assignment[var_arr]  # shape (m, 3), bool
    pos = lit_arr > 0        # shape (m, 3), bool
    sat_literal = a == pos   # shape (m, 3)
    return sat_literal.any(axis=1)  # shape (m,)


def _update_sat_mask(assignment, sat_mask, lit_arr, var_arr, var_to_clauses, v):
    """Incrementally update sat_mask after flipping variable v. Returns new unsat_count."""
    # For each clause containing v, re-evaluate whether it's satisfied
    for ci, pos in var_to_clauses[v]:
        # Check if clause ci is satisfied by any literal
        # We need to check all 3 literals of clause ci
        # But we only changed v, so we can check: was it satisfied before? 
        # If it was satisfied and the satisfying literal was v, it might now be unsat
        # If it was unsat, flipping v might satisfy it
        
        # Simple approach: re-evaluate clause ci fully (3 literals)
        clause_satisfied = False
        for p in range(3):
            var = var_arr[ci, p]
            lit = lit_arr[ci, p]
            if assignment[var] == (lit > 0):
                clause_satisfied = True
                break
        sat_mask[ci] = clause_satisfied
    
    return int((~sat_mask).sum())


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [False] * n

    start_time = time.time()
    DEADLINE = 295.0  # safety margin before 330s kill

    lit_arr, var_arr, var_to_clauses = _build_structures(n, clauses)

    best_assignment = None
    best_unsat = m + 1

    rng = random.Random(12345)

    restart = 0
    while time.time() - start_time < DEADLINE:
        restart += 1

        # Random initial assignment
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)

        # Evaluate
        sat_mask = _eval_clauses(assignment, lit_arr, var_arr)
        unsat_count = int((~sat_mask).sum())

        if unsat_count == 0:
            return assignment.tolist()

        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment.copy()

        # Adaptive WalkSAT with phase-change schedule
        max_steps = 50000  # More steps due to faster incremental updates
        p_min = 0.15
        p_max = 0.9

        unsat_indices = np.where(~sat_mask)[0]

        for step in range(max_steps):
            if time.time() - start_time > DEADLINE:
                break

            if unsat_count == 0:
                return assignment.tolist()

            # Adaptive p: ramp up based on step
            progress = step / max_steps
            p = p_min + (p_max - p_min) * progress

            # Pick an unsatisfied clause
            ci = unsat_indices[rng.randrange(len(unsat_indices))]
            clause_vars = var_arr[ci]

            if rng.random() < p:
                pos = rng.randrange(3)
            else:
                # Heuristic: score each candidate by net impact
                # For each variable v in the clause, estimate:
                #   gain = number of unsat clauses that would become sat (containing v with correct polarity)
                #   loss = number of sat clauses that would become unsat (containing v with wrong polarity, where v is the only satisfying literal)
                # Use a smooth approximation: score = sum over clauses containing v of
                #   +1 if clause is unsat and flipping v would satisfy it
                #   -0.5 if clause is sat and flipping v would unsatisfy it (approximate)
                
                best_pos = 0
                best_score = float('-inf')
                for pos in range(3):
                    v = clause_vars[pos]
                    lit_v = lit_arr[ci, pos]
                    # Current state of v
                    v_is_true = assignment[v]
                    # After flip, v will be not v_is_true
                    # A clause containing v is satisfied by v if:
                    #   (lit_v > 0 and v_is_true) or (lit_v < 0 and not v_is_true)
                    # i.e., v satisfies its literal in this clause when assignment[v] == (lit_v > 0)
                    
                    # Score: count impact on all clauses containing v
                    score = 0.0
                    for cj, _ in var_to_clauses[v]:
                        if sat_mask[cj]:
                            # Clause is currently satisfied. Check if flipping v would unsatisfy it.
                            # This happens if v is the ONLY satisfying literal in clause cj.
                            # Check if any other literal in cj is satisfied
                            other_sat = False
                            for p2 in range(3):
                                if p2 == _:
                                    continue
                                var2 = var_arr[cj, p2]
                                lit2 = lit_arr[cj, p2]
                                if assignment[var2] == (lit2 > 0):
                                    other_sat = True
                                    break
                            if not other_sat:
                                score -= 1.0  # Loss: this clause would become unsat
                        else:
                            # Clause is currently unsat. Check if flipping v would satisfy it.
                            # v satisfies its literal in cj if after flip: (not v_is_true) == (lit2 > 0) where lit2 is v's literal in cj
                            # Find v's literal in cj
                            for p2 in range(3):
                                if var_arr[cj, p2] == v:
                                    lit2 = lit_arr[cj, p2]
                                    # After flip, v is not v_is_true
                                    if (not v_is_true) == (lit2 > 0):
                                        score += 1.0  # Gain: this clause would become sat
                                    break
                    
                    if score > best_score:
                        best_score = score
                        best_pos = pos
                pos = best_pos

            v = clause_vars[pos]
            assignment[v] = not assignment[v]

            # Incremental update
            unsat_count = _update_sat_mask(assignment, sat_mask, lit_arr, var_arr, var_to_clauses, v)
            unsat_indices = np.where(~sat_mask)[0]

            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment.copy()

        if best_unsat == 0:
            return best_assignment.tolist()

    if best_assignment is not None:
        return best_assignment.tolist()
    return [False] * n
# EVOLVE-BLOCK-END
