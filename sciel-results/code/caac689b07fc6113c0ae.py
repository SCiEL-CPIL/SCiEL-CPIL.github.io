# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict
import random


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Convert clauses to 0-based signed literals
    # literal l (1-based): variable = abs(l)-1, sign = l > 0
    # We'll work with a structure that allows efficient updates
    
    # Step 1: Preprocessing - pure literal elimination and unit propagation
    # Build variable -> set of literals appearing in clauses
    # For pure literal: if a variable only appears in one polarity, set it to satisfy those
    
    var_pos = defaultdict(list)  # var -> list of clause indices where it appears positive
    var_neg = defaultdict(list)  # var -> list of clause indices where it appears negative
    var_clauses = defaultdict(list)  # var -> list of clause indices containing it
    
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var = abs(lit) - 1
            var_clauses[var].append(ci)
            if lit > 0:
                var_pos[var].append(ci)
            else:
                var_neg[var].append(ci)
    
    assignment = [None] * n  # None means unassigned
    
    # Pure literal elimination
    changed = True
    while changed:
        changed = False
        for var in range(n):
            if assignment[var] is not None:
                continue
            pos_count = len(var_pos[var])
            neg_count = len(var_neg[var])
            if pos_count == 0 and neg_count > 0:
                # Only appears negative, set to True to satisfy
                assignment[var] = True
                changed = True
            elif neg_count == 0 and pos_count > 0:
                # Only appears positive, set to False to satisfy
                assignment[var] = False
                changed = True
    
    # Now propagate: remove satisfied clauses, simplify others
    # Track which clauses are still unsatisfied
    active_clauses = []
    for ci, clause in enumerate(clauses):
        sat = False
        new_clause = []
        for lit in clause:
            var = abs(lit) - 1
            val = assignment[var]
            if val is not None:
                is_sat = (val == (lit > 0))
                if is_sat:
                    sat = True
                    break
            else:
                new_clause.append(lit)
        if not sat:
            if len(new_clause) == 0:
                # Empty clause - unsatisfiable, but problem is planted SAT so this shouldn't happen
                # If it does, we have a problem
                pass
            elif len(new_clause) == 1:
                # Unit clause
                lit = new_clause[0]
                var = abs(lit) - 1
                if assignment[var] is None:
                    assignment[var] = (lit > 0)
                    # Need to re-propagate, so restart
                    active_clauses = None
                    break
            else:
                active_clauses.append(new_clause)
    
    if active_clauses is None:
        # Restart with propagation
        return _walksat_reduced(n, clauses, assignment, rng_seed=42)
    
    # Check if all clauses are satisfied
    all_sat = True
    for ci, clause in enumerate(clauses):
        sat = False
        for lit in clause:
            var = abs(lit) - 1
            val = assignment[var]
            if val is not None and val == (lit > 0):
                sat = True
                break
        if not sat:
            all_sat = False
            break
    
    if all_sat:
        # Fill in remaining unassigned variables randomly
        final_assignment = []
        rng = random.Random(0)
        for var in range(n):
            if assignment[var] is None:
                final_assignment.append(rng.random() < 0.5)
            else:
                final_assignment.append(assignment[var])
        return final_assignment
    
    # Fall back to WalkSAT on the reduced problem
    return _walksat_reduced(n, clauses, assignment, rng_seed=42)


def _walksat_reduced(n, clauses, partial_assignment, rng_seed=42):
    """
    WalkSAT with clause-weighted selection and dynamic noise.
    partial_assignment: list of True/False/None for each variable
    """
    rng = random.Random(rng_seed)
    
    # Initialize assignment for unassigned variables
    assignment = [None] * n
    for i in range(n):
        if partial_assignment[i] is not None:
            assignment[i] = partial_assignment[i]
        else:
            assignment[i] = rng.random() < 0.5
    
    # Precompute clause structure
    # For each clause, store list of (var, sign) where sign = 1 for positive, 0 for negative
    clause_vars = []
    for clause in clauses:
        cv = []
        for lit in clause:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else 0
            cv.append((var, sign))
        clause_vars.append(cv)
    
    m = len(clause_vars)
    
    def is_satisfied(ci):
        cv = clause_vars[ci]
        for var, sign in cv:
            if assignment[var] == (sign == 1):
                return True
        return False
    
    # Count unsatisfied clauses initially
    unsat_count = sum(1 for ci in range(m) if not is_satisfied(ci))
    
    max_flips = 50000
    noise_start = 0.25
    noise_end = 0.05
    flip_budget_per_clause = 2
    
    for flip in range(max_flips):
        if unsat_count == 0:
            break
        
        # Find all unsatisfied clauses
        unsat_clauses = [ci for ci in range(m) if not is_satisfied(ci)]
        if not unsat_clauses:
            break
        
        # Dynamic noise: decrease over time
        progress = flip / max_flips
        noise = noise_start + (noise_end - noise_start) * progress
        
        # Choose a clause: weight by how many literals could be satisfied (all unsat clauses have 0 satisfied)
        # Use random selection from unsat clauses, but prefer shorter clauses (easier to satisfy)
        # Weight = 1 / len(clause) to prefer shorter
        weights = [1.0 / len(clause_vars[ci]) for ci in unsat_clauses]
        total_w = sum(weights)
        r = rng.random() * total_w
        cumulative = 0
        chosen_ci = unsat_clauses[0]
        for idx, ci in enumerate(unsat_clauses):
            cumulative += weights[idx]
            if r <= cumulative:
                chosen_ci = ci
                break
        
        clause = clause_vars[chosen_ci]
        
        # Choose a literal to flip: with probability (1-noise), pick the one that maximizes gain
        # Gain = number of other clauses that would become satisfied by flipping this variable
        # With probability noise, pick randomly
        
        if rng.random() < noise:
            # Random flip
            var, sign = clause[rng.randrange(len(clause))]
        else:
            # Greedy: pick variable with max gain
            best_gain = -1
            best_var = clause[0][0]
            for var, sign in clause:
                # Calculate gain if we flip var
                current_val = assignment[var]
                new_val = not current_val
                gain = 0
                # Count clauses that contain var and would become satisfied
                for ci2 in range(m):
                    if is_satisfied(ci2):
                        continue
                    cv2 = clause_vars[ci2]
                    # Check if flipping var satisfies this clause
                    for v2, s2 in cv2:
                        if v2 == var and new_val == (s2 == 1):
                            gain += 1
                            break
                if gain > best_gain:
                    best_gain = gain
                    best_var = var
            
            var = best_var
        
        # Flip the variable
        assignment[var] = not assignment[var]
        
        # Update unsat_count: recompute for efficiency (could be optimized with delta tracking)
        # For simplicity, recompute
        new_unsat = 0
        for ci in range(m):
            if not is_satisfied(ci):
                new_unsat += 1
        unsat_count = new_unsat
    
    return assignment


# EVOLVE-BLOCK-END
