"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from collections import defaultdict


def solve(instance):
    """
    Solve a planted 3-SAT instance with n variables and clauses.
    Returns a list of booleans (length n) that satisfies all clauses if possible.
    """
    n, clauses = instance["n"], instance["clauses"]
    num_clauses = len(clauses)
    
    # Deadline: leave a safety margin before the 330s kill
    deadline = time.time() + 325.0
    
    # Pre-process clauses for faster access
    # Each clause is a list of signed literals (1-based).
    # Convert to 0-based indices and signs.
    fast_clauses = []
    for clause in clauses:
        fast_clause = []
        for lit in clause:
            var_idx = abs(lit) - 1
            required = (lit > 0)
            fast_clause.append((var_idx, required))
        fast_clauses.append(fast_clause)
    
    # Build variable-to-clauses index for efficient updates
    var_to_clauses = defaultdict(list)
    for i, fc in enumerate(fast_clauses):
        for var_idx, _ in fc:
            var_to_clauses[var_idx].append(i)
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Adaptive parameters
    base_walk_prob = 0.20
    max_walk_prob = 0.50
    progress_threshold = 500
    
    for restart in range(500):
        if time.time() > deadline:
            break
        
        # Initialize assignment randomly
        assignment = [random.random() < 0.5 for _ in range(n)]
        
        # Initialize clause status
        # sat_count[c]: number of satisfied literals in clause c
        sat_count = [0] * num_clauses
        # unsat_set: set of indices of unsatisfied clauses
        unsat_set = set()
        
        for i in range(num_clauses):
            count = 0
            for var_idx, required in fast_clauses[i]:
                if assignment[var_idx] == required:
                    count += 1
            sat_count[i] = count
            if count == 0:
                unsat_set.add(i)
        
        # break_count[v]: number of satisfied clauses where v is the ONLY satisfied literal
        # If we flip v, these clauses will become unsatisfied.
        break_count = [0] * n
        
        for i in range(num_clauses):
            if sat_count[i] == 1:
                # Find the unique satisfied variable
                for var_idx, required in fast_clauses[i]:
                    if assignment[var_idx] == required:
                        break_count[var_idx] += 1
                        break
        
        current_unsat = len(unsat_set)
        
        if current_unsat == 0:
            return assignment
        
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        # Adaptive walk probability for this restart
        walk_prob = base_walk_prob
        steps_since_improvement = 0
        local_best_unsat = current_unsat
        max_steps = 20000 # Increase steps since heuristic is faster
        
        # WalkSAT loop with clause status tracking
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if not unsat_set:
                return assignment
            
            if len(unsat_set) < best_unsat_count:
                best_unsat_count = len(unsat_set)
                best_assignment = assignment[:]
            
            if len(unsat_set) < local_best_unsat:
                local_best_unsat = len(unsat_set)
                steps_since_improvement = 0
                walk_prob = base_walk_prob
            else:
                steps_since_improvement += 1
                if steps_since_improvement > progress_threshold:
                    walk_prob = min(max_walk_prob, base_walk_prob + (steps_since_improvement - progress_threshold) * 0.001)
            
            # Choose a clause from unsat_set
            clause_idx = random.choice(list(unsat_set))
            clause = fast_clauses[clause_idx]
            
            # Select variable to flip
            if random.random() < walk_prob:
                var_idx, required = random.choice(clause)
            else:
                # Min-Conflicts: choose var with minimum break_count
                min_break = float('inf')
                candidates = []
                for var_idx, required in clause:
                    bc = break_count[var_idx]
                    if bc < min_break:
                        min_break = bc
                        candidates = [var_idx]
                    elif bc == min_break:
                        candidates.append(var_idx)
                
                var_idx = random.choice(candidates)
                # Find the required polarity for this var in this clause
                for v, r in clause:
                    if v == var_idx:
                        required = r
                        break
            
            # Flip the variable
            old_val = assignment[var_idx]
            assignment[var_idx] = not assignment[var_idx]
            
            # Update sat_count and break_count incrementally
            for c_idx in var_to_clauses[var_idx]:
                fc = fast_clauses[c_idx]
                
                # Determine old and new satisfaction status for this clause
                old_sat_val = old_val
                new_sat_val = not old_val
                
                # Find the literal for var_idx in this clause
                var_req = None
                for v, r in fc:
                    if v == var_idx:
                        var_req = r
                        break
                
                # Was the literal satisfied before?
                lit_was_sat = (old_sat_val == var_req)
                # Is the literal satisfied after?
                lit_is_sat = (new_sat_val == var_req)
                
                old_count = sat_count[c_idx]
                
                if lit_was_sat and not lit_is_sat:
                    new_count = old_count - 1
                elif not lit_was_sat and lit_is_sat:
                    new_count = old_count + 1
                else:
                    new_count = old_count
                
                # Identify unique satisfied variable before and after
                old_unique_var = None
                if old_count == 1:
                    if lit_was_sat:
                        old_unique_var = var_idx
                    else:
                        for v, r in fc:
                            if v != var_idx and assignment[v] == r:
                                old_unique_var = v
                                break
                
                new_unique_var = None
                if new_count == 1:
                    if lit_is_sat:
                        new_unique_var = var_idx
                    else:
                        for v, r in fc:
                            if v != var_idx and assignment[v] == r:
                                new_unique_var = v
                                break
                
                # Update break_count
                if old_unique_var is not None:
                    break_count[old_unique_var] -= 1
                if new_unique_var is not None:
                    break_count[new_unique_var] += 1
                
                # Update sat_count and unsat_set
                sat_count[c_idx] = new_count
                
                was_unsat = (old_count == 0)
                is_unsat = (new_count == 0)
                
                if was_unsat and not is_unsat:
                    unsat_set.discard(c_idx)
                elif not was_unsat and is_unsat:
                    unsat_set.add(c_idx)
            
            # Early exit if solved
            if not unsat_set:
                return assignment
        
        # End of restart
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    else:
        # Fallback: return a random assignment
        return [random.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
