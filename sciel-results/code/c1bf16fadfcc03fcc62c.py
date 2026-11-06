# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque
import random


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Convert to 0-based literals
    # We will work with a list of lists of 0-based signed literals
    # But for efficiency in preprocessing, let's keep them as is or convert to (var, sign)
    
    # Preprocessing
    assignment = [None] * n
    active_clauses = [list(c) for c in clauses]
    
    # Build indices for pure literal detection
    # var -> set of clauses where it appears pos/neg
    var_pos_clauses = [set() for _ in range(n)]
    var_neg_clauses = [set() for _ in range(n)]
    
    for ci, clause in enumerate(active_clauses):
        for lit in clause:
            var = abs(lit) - 1
            if lit > 0:
                var_pos_clauses[var].add(ci)
            else:
                var_neg_clauses[var].add(ci)
                
    # Queue for unit clauses
    unit_queue = deque()
    
    # Initial pure literal check
    # If a var only appears in pos, set to True.
    # If a var only appears in neg, set to False.
    changed = True
    while changed:
        changed = False
        for var in range(n):
            if assignment[var] is not None:
                continue
            if len(var_pos_clauses[var]) == 0 and len(var_neg_clauses[var]) > 0:
                assignment[var] = False # Neg literal is true if var is False? No.
                # Literal -var is true if var is False.
                # If var only appears as -var, setting var=False satisfies -var.
                # So assignment[var] = False.
                changed = True
            elif len(var_neg_clauses[var]) == 0 and len(var_pos_clauses[var]) > 0:
                # Var only appears as +var. Setting var=True satisfies +var.
                assignment[var] = True
                changed = True
                
    # Propagate assignments
    # We need a loop that processes assignments and simplifies clauses
    # We'll use a queue of newly assigned variables
    # Or just iterate until stable
    
    # Let's do a standard unit propagation loop
    # We need to track which clauses are active (not satisfied, not empty)
    # And which literals are still in the clause
    
    # This is getting complex to do perfectly in one pass without rebuilding indices.
    # Given n=150, a simple iterative simplification is fast enough.
    
    def simplify_and_propagate():
        nonlocal assignment, active_clauses, var_pos_clauses, var_neg_clauses
        changed = True
        while changed:
            changed = False
            # 1. Pure Literal Elimination
            for var in range(n):
                if assignment[var] is not None:
                    continue
                if len(var_pos_clauses[var]) == 0 and len(var_neg_clauses[var]) > 0:
                    assignment[var] = False
                    changed = True
                elif len(var_neg_clauses[var]) == 0 and len(var_pos_clauses[var]) > 0:
                    assignment[var] = True
                    changed = True
            
            # 2. Simplify Clauses
            new_active_clauses = []
            new_var_pos = [set() for _ in range(n)]
            new_var_neg = [set() for _ in range(n)]
            
            for ci, clause in enumerate(active_clauses):
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
                
                if sat:
                    # Clause satisfied, remove from active
                    continue
                
                if len(new_clause) == 0:
                    # Empty clause -> UNSAT (should not happen)
                    return False, assignment, []
                
                # Add to new active clauses
                new_ci = len(new_active_clauses)
                new_active_clauses.append(new_clause)
                
                for lit in new_clause:
                    var = abs(lit) - 1
                    if lit > 0:
                        new_var_pos[var].add(new_ci)
                    else:
                        new_var_neg[var].add(new_ci)
                        
                if len(new_clause) == 1:
                    # Unit clause
                    lit = new_clause[0]
                    var = abs(lit) - 1
                    if assignment[var] is None:
                        assignment[var] = (lit > 0)
                        changed = True
                        # We don't add to queue, we just restart loop
                        # But we must be careful: we are iterating over active_clauses
                        # and building new ones. If we change assignment, subsequent clauses in this iteration
                        # might be affected.
                        # To be safe, if we find a unit, we can break and restart the while loop.
                        # Or we can just mark changed=True and let the next iteration handle it.
                        # However, the current `new_active_clauses` might contain clauses that are now satisfied
                        # by this new assignment.
                        # So it's better to break out of the clause loop and restart the while loop.
                        pass 
            
            # If we didn't find a unit clause that changed assignment, we can update state
            # But if we did find one, `changed` is True, so we will loop again.
            # The `new_active_clauses` built so far is correct for the assignments made *before* the unit change?
            # No, if we assigned a var in the middle of the loop, the subsequent clauses in `active_clauses` 
            # were not simplified with the new assignment.
            # So if `changed` becomes True due to a unit clause, we should discard `new_active_clauses` 
            # and restart the `while changed` loop.
            
            if changed:
                # We made an assignment change. 
                # We need to re-evaluate everything.
                # But wait, if we just set `changed=True`, the loop continues.
                # We haven't updated `active_clauses` yet.
                # So we just continue the while loop.
                continue
            
            # If no changes, update structures
            active_clauses = new_active_clauses
            var_pos_clauses = new_var_pos
            var_neg_clauses = new_var_neg
            
        return True, assignment, active_clauses

    success, assignment, active_clauses = simplify_and_propagate()
    
    if not success:
        # Should not happen for planted SAT
        return [True] * n
        
    # Check if all clauses are satisfied
    if not active_clauses:
        # All satisfied
        final_assignment = []
        rng = random.Random(0)
        for var in range(n):
            if assignment[var] is None:
                final_assignment.append(rng.random() < 0.5)
            else:
                final_assignment.append(assignment[var])
        return final_assignment
        
    # WalkSAT on the remaining problem
    return _walksat_fast(n, active_clauses, assignment, rng_seed=42)


def _walksat_fast(n, clauses, partial_assignment, rng_seed=42):
    """
    Fast WalkSAT with delta tracking.
    """
    rng = random.Random(rng_seed)
    
    # Initialize assignment
    assignment = [None] * n
    for i in range(n):
        if partial_assignment[i] is not None:
            assignment[i] = partial_assignment[i]
        else:
            assignment[i] = rng.random() < 0.5
            
    m = len(clauses)
    if m == 0:
        return assignment
        
    # Precompute clause structure
    # clause_vars: list of lists of (var, sign) where sign is 1 for True, 0 for False
    clause_vars = []
    var_clauses = defaultdict(list) # var -> list of (clause_idx, lit_sign)
    
    for ci, clause in enumerate(clauses):
        cv = []
        for lit in clause:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else 0
            cv.append((var, sign))
            var_clauses[var].append((ci, sign))
        clause_vars.append(cv)
        
    # Initial satisfaction check
    clause_sat = [False] * m
    unsat_count = 0
    
    for ci, cv in enumerate(clause_vars):
        sat = False
        for var, sign in cv:
            if assignment[var] == (sign == 1):
                sat = True
                break
        clause_sat[ci] = sat
        if not sat:
            unsat_count += 1
            
    if unsat_count == 0:
        return assignment
        
    max_flips = 100000
    noise_start = 0.25
    noise_end = 0.05
    
    # List of unsatisfied clause indices for fast selection
    unsat_clauses = [ci for ci in range(m) if not clause_sat[ci]]
    
    for flip in range(max_flips):
        if unsat_count == 0:
            break
            
        # Dynamic noise
        progress = flip / max_flips
        noise = noise_start + (noise_end - noise_start) * progress
        
        # Select unsatisfied clause
        if not unsat_clauses:
            break
        chosen_ci = unsat_clauses[rng.randrange(len(unsat_clauses))]
        chosen_clause_vars = clause_vars[chosen_ci]
        
        # Select variable to flip
        best_var = None
        best_delta = -100
        best_noise_var = None
        
        # Calculate delta for each variable in the clause
        for var, sign in chosen_clause_vars:
            delta = 1 # The chosen clause becomes satisfied
            
            # Check other clauses containing this variable
            for ci2, sign2 in var_clauses[var]:
                if ci2 == chosen_ci:
                    continue
                    
                old_val = assignment[var]
                new_val = not old_val
                
                if clause_sat[ci2]:
                    # Clause is satisfied. Check if it becomes unsatisfied.
                    # It becomes unsatisfied if the literal (var, sign2) was the ONLY true literal.
                    # Since we are flipping var, the literal (var, sign2) changes value.
                    # If old_val == (sign2 == 1), the literal was true.
                    # If new_val != (sign2 == 1), the literal becomes false.
                    # We need to check if any OTHER literal in clause ci2 is true.
                    if old_val == (sign2 == 1):
                        # The literal was true. Now it becomes false.
                        # Check other literals
                        other_sat = False
                        for v2, s2 in clause_vars[ci2]:
                            if v2 == var:
                                continue
                            if assignment[v2] == (s2 == 1):
                                other_sat = True
                                break
                        if not other_sat:
                            delta -= 1 # Clause becomes unsat
                else:
                    # Clause is unsatisfied. Check if it becomes satisfied.
                    # It becomes satisfied if the new value of var satisfies the literal.
                    if new_val == (sign2 == 1):
                        delta += 1 # Clause becomes sat
                        
            if rng.random() < noise:
                # Random selection
                if best_noise_var is None:
                    best_noise_var = var
            else:
                if delta > best_delta:
                    best_delta = delta
                    best_var = var
                    
        if best_var is None:
            best_var = best_noise_var
            if best_var is None:
                # Fallback to random var in clause
                best_var = chosen_clause_vars[0][0]
                
        # Flip the variable
        old_val = assignment[best_var]
        assignment[best_var] = not old_val
        
        # Update clause_sat and unsat_count
        # Only clauses containing best_var are affected
        for ci2, sign2 in var_clauses[best_var]:
            old_sat = clause_sat[ci2]
            # Check new satisfaction
            new_sat = False
            for v2, s2 in clause_vars[ci2]:
                if assignment[v2] == (s2 == 1):
                    new_sat = True
                    break
            clause_sat[ci2] = new_sat
            
            if old_sat and not new_sat:
                unsat_count += 1
                # Add to unsat_clauses list? 
                # We are using a list `unsat_clauses`. 
                # Removing from a list is O(n). 
                # It's better to just rebuild `unsat_clauses` occasionally or use a set.
                # Given m is small (after preprocessing), rebuilding is fine.
                # Or we can just scan for unsat clauses in the selection step.
                # Let's just rebuild `unsat_clauses` if it gets too messy or every K steps.
                # For simplicity, we'll just find a random unsat clause by scanning if list is empty or invalid.
                # Actually, let's maintain `unsat_clauses` as a list and clean it up.
            elif not old_sat and new_sat:
                unsat_count -= 1
                
        # Clean up unsat_clauses list
        # Remove satisfied clauses, add newly unsatisfied ones
        # This is O(m). Since m is small, it's okay.
        if flip % 10 == 0: # Every 10 flips
            unsat_clauses = [ci for ci in range(m) if not clause_sat[ci]]
        else:
            # Quick check: if the chosen clause is now sat, remove it.
            # If any other clause became unsat, we don't know which one immediately without scanning.
            # So we rely on the periodic rebuild or just scan if list is empty.
            # To be safe, if `unsat_clauses` is empty but `unsat_count > 0`, rebuild.
            if not unsat_clauses and unsat_count > 0:
                unsat_clauses = [ci for ci in range(m) if not clause_sat[ci]]
            else:
                # Filter out satisfied ones
                # This is inefficient if done every time.
                # Let's just rebuild every 5 flips.
                pass
                
        # Rebuild every 5 flips to keep list accurate
        if flip % 5 == 0:
            unsat_clauses = [ci for ci in range(m) if not clause_sat[ci]]

    return assignment


# EVOLVE-BLOCK-END
