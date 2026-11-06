# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Convert clauses to 0-based signed literals
    # literal l (1-based): variable = abs(l)-1, sign = l > 0
    # We'll work with a structure that allows efficient updates
    
    # Step 1: Preprocessing - pure literal elimination and unit propagation
    # We will use a robust unit propagation loop.
    
    # Initialize assignment
    assignment = [None] * n
    
    # Build index structures for efficient propagation
    # var_to_clauses[var] = list of (clause_idx, sign)
    # clause_literals[clause_idx] = list of (var, sign)
    
    var_to_clauses = defaultdict(list)
    clause_literals = []
    
    for ci, clause in enumerate(clauses):
        lit_list = []
        for lit in clause:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else 0
            lit_list.append((var, sign))
            var_to_clauses[var].append((ci, sign))
        clause_literals.append(lit_list)
        
    m = len(clauses)
    
    # Track status of clauses
    # sat_count[ci] = number of literals in clause ci that are currently satisfied
    # unassigned_count[ci] = number of literals in clause ci that are unassigned
    # If sat_count > 0, clause is satisfied.
    # If sat_count == 0 and unassigned_count == 0, clause is violated (empty).
    # If sat_count == 0 and unassigned_count == 1, clause is unit.
    
    sat_count = [0] * m
    unassigned_count = [0] * m
    
    # Initialize counts
    for ci in range(m):
        unassigned_count[ci] = len(clause_literals[ci])
        
    # Queue for unit clauses
    # A clause becomes unit if unassigned_count == 1 and sat_count == 0
    # We need to identify which variable is left.
    
    # Helper to get the single unassigned literal in a clause
    def get_unit_literal(ci):
        for var, sign in clause_literals[ci]:
            if assignment[var] is None:
                return var, sign
        return None, None # Should not happen if unassigned_count==1
        
    # Unit Propagation
    # We'll use a queue of clause indices that are currently unit or empty
    # Initially, find all unit clauses
    unit_queue = []
    for ci in range(m):
        if unassigned_count[ci] == 1 and sat_count[ci] == 0:
            unit_queue.append(ci)
            
    # Process queue
    # We need to be careful about re-processing.
    # If we assign a variable, we update all clauses containing it.
    # This might create new unit clauses.
    
    # To avoid infinite loops or redundant work, we can just loop until no changes.
    # Given n=150, this is small enough for a simple loop with queue.
    
    changed = True
    while changed:
        changed = False
        while unit_queue:
            ci = unit_queue.pop()
            # Check if it's still a unit clause that needs action
            # It might have become satisfied or empty due to other assignments
            if sat_count[ci] > 0:
                continue
            if unassigned_count[ci] == 0:
                # Empty clause -> UNSAT (should not happen for planted SAT)
                # But if it does, we break
                return [False] * n # Fallback, shouldn't happen
            if unassigned_count[ci] == 1:
                var, sign = get_unit_literal(ci)
                if assignment[var] is not None:
                    # Already assigned, check consistency
                    if assignment[var] != (sign == 1):
                         # Contradiction
                         return [False] * n
                    else:
                         # Already satisfied, do nothing
                         continue
                else:
                    # Assign variable
                    assignment[var] = (sign == 1)
                    changed = True
                    
                    # Update all clauses containing this variable
                    for ci2, sign2 in var_to_clauses[var]:
                        # If clause is already satisfied, skip? No, we need to update counts correctly.
                        # Actually, if sat_count[ci2] > 0, it's satisfied. Flipping a var in it doesn't change sat status (it stays sat).
                        # But we need to update unassigned_count.
                        
                        # Was this literal satisfied before?
                        # assignment[var] was None, so it wasn't satisfied.
                        # Now assignment[var] is set.
                        # Does it satisfy the clause?
                        is_sat = (assignment[var] == (sign2 == 1))
                        
                        if is_sat:
                            sat_count[ci2] += 1
                        # unassigned_count decreases by 1 regardless
                        unassigned_count[ci2] -= 1
                        
                        # Check if this clause becomes unit
                        if sat_count[ci2] == 0 and unassigned_count[ci2] == 1:
                            unit_queue.append(ci2)
                        
                        # Check if empty (UNSAT)
                        if sat_count[ci2] == 0 and unassigned_count[ci2] == 0:
                            # Contradiction
                            return [False] * n

    # After propagation, check if all clauses are satisfied
    all_sat = True
    for ci in range(m):
        if sat_count[ci] == 0:
            # If unassigned_count > 0, it's not satisfied yet
            # If unassigned_count == 0, it's empty (UNSAT)
            all_sat = False
            break
            
    if all_sat:
        # Fill remaining unassigned variables randomly
        rng = random.Random(0)
        final_assignment = []
        for var in range(n):
            if assignment[var] is None:
                final_assignment.append(rng.random() < 0.5)
            else:
                final_assignment.append(assignment[var])
        return final_assignment
        
    # Fall back to WalkSAT on the reduced problem
    # We pass the current assignment state to WalkSAT
    
    return _walksat_efficient(n, clauses, assignment)


def _walksat_efficient(n, clauses, initial_assignment):
    """
    Efficient WalkSAT with delta tracking.
    """
    rng = random.Random(42)
    
    # Initialize assignment
    assignment = [None] * n
    for i in range(n):
        if initial_assignment[i] is not None:
            assignment[i] = initial_assignment[i]
        else:
            assignment[i] = rng.random() < 0.5
            
    # Build structures
    m = len(clauses)
    clause_literals = []
    var_to_clauses = defaultdict(list)
    
    for ci, clause in enumerate(clauses):
        lit_list = []
        for lit in clause:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else 0
            lit_list.append((var, sign))
            var_to_clauses[var].append((ci, sign))
        clause_literals.append(lit_list)
        
    # Calculate initial sat_count and unsat_clauses
    sat_count = [0] * m
    unsat_clauses = set()
    
    for ci in range(m):
        sat = 0
        for var, sign in clause_literals[ci]:
            if assignment[var] == (sign == 1):
                sat += 1
        sat_count[ci] = sat
        if sat == 0:
            unsat_clauses.add(ci)
            
    unsat_count = len(unsat_clauses)
    
    max_flips = 100000
    noise_start = 0.25
    noise_end = 0.05
    
    # Time limit check (optional, but good practice)
    start_time = time.time()
    time_limit = 250 # Leave buffer for Python overhead
    
    for flip in range(max_flips):
        if unsat_count == 0:
            break
            
        if time.time() - start_time > time_limit:
            break
        
        progress = flip / max_flips
        noise = noise_start + (noise_end - noise_start) * progress
        
        # Choose a clause
        # Randomly pick from unsat_clauses
        # For efficiency, we can just pick a random one from the set
        # If set is large, list conversion is costly. But m is small (likely < 1000).
        # Let's keep it as a set for O(1) add/remove, but pick randomly.
        if not unsat_clauses:
            break
        chosen_ci = rng.choice(list(unsat_clauses))
        
        clause = clause_literals[chosen_ci]
        
        # Choose variable to flip
        if rng.random() < noise:
            # Random flip
            var, sign = clause[rng.randrange(len(clause))]
        else:
            # Greedy: pick variable with max gain
            best_gain = -1
            best_var = clause[0][0]
            
            for var, sign in clause:
                # Calculate gain
                # Gain = (clauses satisfied) - (clauses unsatisfied)
                # We only need to check clauses containing 'var'
                
                # Current state of 'var': assignment[var]
                # New state: not assignment[var]
                
                # For each clause containing var:
                # 1. If clause is currently satisfied:
                #    - If this literal is the ONLY satisfied literal, flipping makes it unsat (loss 1)
                #    - Else, flipping keeps it sat (no change)
                # 2. If clause is currently unsatisfied:
                #    - If flipping satisfies this literal, clause becomes sat (gain 1)
                #    - Else, clause remains unsat (no change)
                
                gain = 0
                for ci2, sign2 in var_to_clauses[var]:
                    # Check if current assignment satisfies ci2
                    # We need to know if this specific literal is the only one satisfying ci2
                    # This requires knowing sat_count[ci2] and if this literal is satisfied.
                    
                    is_lit_sat = (assignment[var] == (sign2 == 1))
                    
                    if sat_count[ci2] > 0:
                        # Clause is satisfied
                        if is_lit_sat:
                            # This literal is contributing to satisfaction
                            if sat_count[ci2] == 1:
                                # It's the only one, so flipping will make clause unsat
                                gain -= 1
                            # else: flipping doesn't affect sat status (other lit still sat)
                    else:
                        # Clause is unsatisfied
                        # If flipping satisfies this literal, it becomes sat
                        # New value is not assignment[var]
                        new_val = not assignment[var]
                        if new_val == (sign2 == 1):
                            gain += 1
                            
                if gain > best_gain:
                    best_gain = gain
                    best_var = var
                    
            var = best_var
            
        # Flip the variable
        old_val = assignment[var]
        new_val = not old_val
        assignment[var] = new_val
        
        # Update sat_count and unsat_clauses
        for ci2, sign2 in var_to_clauses[var]:
            # Determine if the literal (var, sign2) is now satisfied
            # It is satisfied if new_val == (sign2 == 1)
            is_lit_sat_now = (new_val == (sign2 == 1))
            is_lit_sat_before = (old_val == (sign2 == 1))
            
            if is_lit_sat_now and not is_lit_sat_before:
                # Literal became satisfied
                sat_count[ci2] += 1
                if sat_count[ci2] == 1:
                    # Clause became satisfied
                    unsat_clauses.discard(ci2)
                    unsat_count -= 1
            elif not is_lit_sat_now and is_lit_sat_before:
                # Literal became unsatisfied
                sat_count[ci2] -= 1
                if sat_count[ci2] == 0:
                    # Clause became unsatisfied
                    unsat_clauses.add(ci2)
                    unsat_count += 1
            
    return assignment

# EVOLVE-BLOCK-END
