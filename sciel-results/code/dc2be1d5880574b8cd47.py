# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # 1. Preprocessing: Pure Literal Elimination and Unit Propagation
    assignment = [None] * n
    
    var_clauses = defaultdict(list)
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var = abs(lit) - 1
            var_clauses[var].append(ci)
    
    changed = True
    while changed:
        changed = False
        
        # Identify active clauses (not satisfied by current assignment)
        active_clause_indices = []
        for ci in range(len(clauses)):
            clause = clauses[ci]
            sat = False
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] is not None:
                    if assignment[var] == (lit > 0):
                        sat = True
                        break
            if not sat:
                active_clause_indices.append(ci)
        
        if not active_clause_indices:
            break
            
        # Pure Literal Elimination
        var_pos_count = defaultdict(int)
        var_neg_count = defaultdict(int)
        for ci in active_clause_indices:
            clause = clauses[ci]
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] is None:
                    if lit > 0:
                        var_pos_count[var] += 1
                    else:
                        var_neg_count[var] += 1
        
        for var in range(n):
            if assignment[var] is not None:
                continue
            pos = var_pos_count.get(var, 0)
            neg = var_neg_count.get(var, 0)
            
            if pos > 0 and neg == 0:
                assignment[var] = True
                changed = True
            elif neg > 0 and pos == 0:
                assignment[var] = False
                changed = True
        
        # Unit Propagation
        for ci in active_clause_indices:
            clause = clauses[ci]
            unassigned_lits = []
            sat = False
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] is not None:
                    if assignment[var] == (lit > 0):
                        sat = True
                        break
                else:
                    unassigned_lits.append(lit)
            
            if sat:
                continue
                
            if len(unassigned_lits) == 0:
                # Conflict, fallback to WalkSAT
                return _walksat_optimized(n, clauses, assignment, rng_seed=123)
            
            if len(unassigned_lits) == 1:
                lit = unassigned_lits[0]
                var = abs(lit) - 1
                if assignment[var] is None:
                    assignment[var] = (lit > 0)
                    changed = True
    
    # Check if all clauses are satisfied
    all_sat = True
    for clause in clauses:
        sat = False
        for lit in clause:
            var = abs(lit) - 1
            if assignment[var] is not None and assignment[var] == (lit > 0):
                sat = True
                break
        if not sat:
            all_sat = False
            break
    
    if all_sat:
        rng = random.Random(0)
        final_assignment = []
        for var in range(n):
            if assignment[var] is None:
                final_assignment.append(rng.random() < 0.5)
            else:
                final_assignment.append(assignment[var])
        return final_assignment
    
    # Fall back to optimized WalkSAT
    return _walksat_optimized(n, clauses, assignment, rng_seed=42)


def _walksat_optimized(n, clauses, initial_assignment, rng_seed=42):
    """
    Optimized WalkSAT with delta tracking and restarts.
    """
    clause_vars = []
    for clause in clauses:
        cv = []
        for lit in clause:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else 0
            cv.append((var, sign))
        clause_vars.append(cv)
    
    m = len(clause_vars)
    
    var_to_clauses = defaultdict(list)
    for ci, cv in enumerate(clause_vars):
        for var, sign in cv:
            var_to_clauses[var].append(ci)
    
    def is_satisfied(ci, assignment):
        cv = clause_vars[ci]
        for var, sign in cv:
            if assignment[var] == (sign == 1):
                return True
        return False
        
    def count_unsat(assignment):
        count = 0
        for ci in range(m):
            if not is_satisfied(ci, assignment):
                count += 1
        return count

    best_assignment = None
    best_unsat = float('inf')
    
    max_restarts = 10
    max_flips_per_restart = 10000
    noise_start = 0.25
    noise_end = 0.05
    
    for restart in range(max_restarts):
        if restart == 0 and initial_assignment:
            assignment = [False] * n
            for i in range(n):
                if initial_assignment[i] is not None:
                    assignment[i] = initial_assignment[i]
                else:
                    assignment[i] = (random.Random(rng_seed + restart).random() < 0.5)
        else:
            rng_init = random.Random(rng_seed + restart * 100)
            assignment = [rng_init.random() < 0.5 for _ in range(n)]
        
        unsat_count = count_unsat(assignment)
        
        if unsat_count == 0:
            return assignment
            
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
            
        if best_unsat == 0:
            return best_assignment

        rng = random.Random(rng_seed + restart)
        
        for flip in range(max_flips_per_restart):
            if unsat_count == 0:
                return assignment
                
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
                
            progress = flip / max_flips_per_restart
            noise = noise_start + (noise_end - noise_start) * progress
            
            # Select an unsatisfied clause
            chosen_ci = None
            for _ in range(10):
                ci = rng.randrange(m)
                if not is_satisfied(ci, assignment):
                    chosen_ci = ci
                    break
            
            if chosen_ci is None:
                for ci in range(m):
                    if not is_satisfied(ci, assignment):
                        chosen_ci = ci
                        break
                if chosen_ci is None:
                    break
            
            clause = clause_vars[chosen_ci]
            
            # Choose variable to flip
            if rng.random() < noise:
                var, sign = clause[rng.randrange(len(clause))]
            else:
                best_gain = -1
                best_var = clause[0][0]
                for var, sign in clause:
                    current_val = assignment[var]
                    new_val = not current_val
                    
                    gain = 0
                    for ci2 in var_to_clauses[var]:
                        if is_satisfied(ci2, assignment):
                            continue
                        for v2, s2 in clause_vars[ci2]:
                            if v2 == var:
                                if (s2 == 1) == new_val:
                                    gain += 1
                                break
                    if gain > best_gain:
                        best_gain = gain
                        best_var = var
                        if best_gain == len(var_to_clauses[var]):
                            break
                            
                var = best_var
            
            # Delta Update
            affected_cis = var_to_clauses[var]
            was_sat = [is_satisfied(ci, assignment) for ci in affected_cis]
            
            assignment[var] = not assignment[var]
            
            for idx, ci in enumerate(affected_cis):
                is_sat = is_satisfied(ci, assignment)
                if was_sat[idx] and not is_sat:
                    unsat_count += 1
                elif not was_sat[idx] and is_sat:
                    unsat_count -= 1

    return best_assignment

# EVOLVE-BLOCK-END
