"""Stronger planted 3-SAT solver: vectorized adaptive WalkSAT with exact gain/loss scoring and smart init."""
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
    for ci, pos in var_to_clauses[v]:
        clause_satisfied = False
        for p in range(3):
            var = var_arr[ci, p]
            lit = lit_arr[ci, p]
            if assignment[var] == (lit > 0):
                clause_satisfied = True
                break
        sat_mask[ci] = clause_satisfied
    
    return int((~sat_mask).sum())


def _exact_flip_score(assignment, sat_mask, lit_arr, var_arr, var_to_clauses, v):
    """
    Compute exact gain/loss score for flipping variable v.
    Returns: (gain - loss) where gain = # unsat clauses that become sat,
             loss = # sat clauses that become unsat.
    """
    v_is_true = assignment[v]
    new_val = not v_is_true
    
    score = 0.0
    for cj, cpos in var_to_clauses[v]:
        lit_v_cj = lit_arr[cj, cpos]
        v_satisfies_after_flip = (new_val == (lit_v_cj > 0))
        
        if sat_mask[cj]:
            v_satisfies_now = (v_is_true == (lit_v_cj > 0))
            if v_satisfies_now and not v_satisfies_after_flip:
                other_sat = False
                for p2 in range(3):
                    if p2 == cpos:
                        continue
                    var2 = var_arr[cj, p2]
                    lit2 = lit_arr[cj, p2]
                    if assignment[var2] == (lit2 > 0):
                        other_sat = True
                        break
                if not other_sat:
                    score -= 1.0
        else:
            if v_satisfies_after_flip:
                score += 1.0
    
    return score


def _smart_init(n, clauses, lit_arr, var_arr, var_to_clauses, rng, max_iter=300):
    """
    Smart initialization: start random, then do rounds of best-improvement
    flips to get closer to a local minimum before full WalkSAT.
    """
    m = len(clauses)
    assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
    sat_mask = _eval_clauses(assignment, lit_arr, var_arr)
    unsat_count = int((~sat_mask).sum())
    
    if unsat_count == 0:
        return assignment, sat_mask, unsat_count
    
    for _ in range(max_iter):
        if unsat_count == 0:
            break
        best_var = -1
        best_score = float('-inf')
        
        # Sample up to 100 random variables to evaluate (for speed)
        candidates = rng.sample(range(n), min(100, n))
        for v in candidates:
            in_unsat = False
            for ci, _ in var_to_clauses[v]:
                if not sat_mask[ci]:
                    in_unsat = True
                    break
            if not in_unsat:
                continue
            
            v_is_true = assignment[v]
            new_val = not v_is_true
            
            gain = 0
            loss = 0
            for ci, cpos in var_to_clauses[v]:
                lit_v = lit_arr[ci, cpos]
                v_satisfies_after = (new_val == (lit_v > 0))
                
                if sat_mask[ci]:
                    v_satisfies_now = (v_is_true == (lit_v > 0))
                    if v_satisfies_now and not v_satisfies_after:
                        other_sat = False
                        for p2 in range(3):
                            if p2 == cpos:
                                continue
                            var2 = var_arr[ci, p2]
                            lit2 = lit_arr[ci, p2]
                            if assignment[var2] == (lit2 > 0):
                                other_sat = True
                                break
                        if not other_sat:
                            loss += 1
                else:
                    if v_satisfies_after:
                        gain += 1
            
            net = gain - loss
            if net > best_score:
                best_score = net
                best_var = v
        
        if best_var == -1 or best_score <= 0:
            break
        
        v = best_var
        assignment[v] = not assignment[v]
        unsat_count = _update_sat_mask(assignment, sat_mask, lit_arr, var_arr, var_to_clauses, v)
    
    return assignment, sat_mask, unsat_count


def _local_search_polish(assignment, sat_mask, lit_arr, var_arr, var_to_clauses, rng, max_iter=500):
    """
    Final polish: aggressive best-improvement local search to close in on a solution.
    """
    m = len(clauses)
    unsat_count = int((~sat_mask).sum())
    
    if unsat_count == 0:
        return assignment, sat_mask, unsat_count
    
    for _ in range(max_iter):
        if unsat_count == 0:
            break
        
        best_var = -1
        best_score = float('-inf')
        
        # Evaluate all variables that appear in unsatisfied clauses
        unsat_vars = set()
        unsat_indices = np.where(~sat_mask)[0]
        for ci in unsat_indices:
            for p in range(3):
                unsat_vars.add(var_arr[ci, p])
        
        for v in unsat_vars:
            v_is_true = assignment[v]
            new_val = not v_is_true
            
            gain = 0
            loss = 0
            for ci, cpos in var_to_clauses[v]:
                lit_v = lit_arr[ci, cpos]
                v_satisfies_after = (new_val == (lit_v > 0))
                
                if sat_mask[ci]:
                    v_satisfies_now = (v_is_true == (lit_v > 0))
                    if v_satisfies_now and not v_satisfies_after:
                        other_sat = False
                        for p2 in range(3):
                            if p2 == cpos:
                                continue
                            var2 = var_arr[ci, p2]
                            lit2 = lit_arr[ci, p2]
                            if assignment[var2] == (lit2 > 0):
                                other_sat = True
                                break
                        if not other_sat:
                            loss += 1
                else:
                    if v_satisfies_after:
                        gain += 1
            
            net = gain - loss
            if net > best_score:
                best_score = net
                best_var = v
        
        if best_var == -1 or best_score <= 0:
            break
        
        v = best_var
        assignment[v] = not assignment[v]
        unsat_count = _update_sat_mask(assignment, sat_mask, lit_arr, var_arr, var_to_clauses, v)
    
    return assignment, sat_mask, unsat_count


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

        # Smart initial assignment
        assignment, sat_mask, unsat_count = _smart_init(n, clauses, lit_arr, var_arr, var_to_clauses, rng)

        if unsat_count == 0:
            return assignment.tolist()

        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment.copy()

        # Adaptive WalkSAT with phase-change schedule
        max_steps = 50000
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
                # Exact gain/loss scoring for each variable in the clause
                best_pos = 0
                best_score = float('-inf')
                for pos in range(3):
                    v = clause_vars[pos]
                    score = _exact_flip_score(assignment, sat_mask, lit_arr, var_arr, var_to_clauses, v)
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

        # Final polish: aggressive local search on the best assignment found in this restart
        if time.time() - start_time < DEADLINE - 10:
            assignment, sat_mask, unsat_count = _local_search_polish(
                assignment, sat_mask, lit_arr, var_arr, var_to_clauses, rng, max_iter=200
            )
            if unsat_count == 0:
                return assignment.tolist()
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment.copy()

        if best_unsat == 0:
            return best_assignment.tolist()

    if best_assignment is not None:
        return best_assignment.tolist()
    return [False] * n
# EVOLVE-BLOCK-END
