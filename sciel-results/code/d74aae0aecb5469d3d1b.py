import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    # Deadline: 325 seconds to be safe against 330s kill
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Parameters
    max_steps_per_run = 5000
    
    # Pre-process: for each variable, store list of (clause_idx, literal_value)
    var_occurrences = [[] for _ in range(n)]
    
    # Initialize
    rng = random.Random(42)
    
    # Build index
    for i, c in enumerate(clauses):
        for l in c:
            v = abs(l) - 1
            var_occurrences[v].append((i, l))
    
    # Helper: compute clause_sat_count from scratch
    def compute_clause_sat_count(assignment):
        counts = [0] * len(clauses)
        for i, c in enumerate(clauses):
            for l in c:
                v = abs(l) - 1
                if (l > 0 and assignment[v]) or (l < 0 and not assignment[v]):
                    counts[i] += 1
        return counts
    
    # Helper: check if all clauses satisfied
    def is_satisfied(clause_sat_count):
        for count in clause_sat_count:
            if count == 0:
                return False
        return True
    
    # Helper: get unsat clause indices
    def get_unsat_indices(clause_sat_count):
        return [i for i, count in enumerate(clause_sat_count) if count == 0]
    
    # Helper: check if a literal is satisfied by current assignment
    def is_literal_satisfied(l, assignment):
        v = abs(l) - 1
        return (l > 0 and assignment[v]) or (l < 0 and not assignment[v])
    
    # Helper: apply a flip to variable v, updating clause_sat_count
    def apply_flip(v, assignment, clause_sat_count):
        assignment[v] = not assignment[v]
        for ci, lit in var_occurrences[v]:
            if is_literal_satisfied(lit, assignment):
                clause_sat_count[ci] += 1
            else:
                clause_sat_count[ci] -= 1
    
    # Helper: undo a flip (same as applying it again)
    def undo_flip(v, assignment, clause_sat_count):
        # Same as apply_flip since flipping twice returns to original
        apply_flip(v, assignment, clause_sat_count)
    
    # WalkSAT step with adaptive noise
    def walksat_step(assignment, clause_sat_count, rng, noise):
        unsat_indices = get_unsat_indices(clause_sat_count)
        if not unsat_indices:
            return True
        
        # Pick random unsat clause
        clause_idx = rng.choice(unsat_indices)
        clause = clauses[clause_idx]
        
        # With probability noise, pick a random literal from the clause
        if rng.random() < noise:
            l = rng.choice(clause)
            v = abs(l) - 1
            apply_flip(v, assignment, clause_sat_count)
            return False
        
        # Otherwise, pick the literal that minimizes newly unsatisfied clauses
        best_var_idx = None
        best_newly_unsat = float('inf')
        best_newly_sat = 0
        
        for l in clause:
            v = abs(l) - 1
            
            # Calculate the effect of flipping v
            newly_unsat = 0
            newly_sat = 0
            
            for ci, lit in var_occurrences[v]:
                old_count = clause_sat_count[ci]
                # Determine if flipping v changes satisfaction of this clause
                # If lit is currently satisfied, flipping v makes it unsatisfied (count -1)
                # If lit is currently unsatisfied, flipping v makes it satisfied (count +1)
                currently_sat = is_literal_satisfied(lit, assignment)
                
                if currently_sat:
                    # After flip, this literal becomes unsatisfied
                    new_count = old_count - 1
                    if old_count == 1 and new_count == 0:
                        newly_unsat += 1
                else:
                    # After flip, this literal becomes satisfied
                    new_count = old_count + 1
                    if old_count == 0 and new_count == 1:
                        newly_sat += 1
            
            # Prioritize minimizing newly_unsat, then maximize newly_sat
            if newly_unsat < best_newly_unsat or (newly_unsat == best_newly_unsat and newly_sat > best_newly_sat):
                best_newly_unsat = newly_unsat
                best_newly_sat = newly_sat
                best_var_idx = v
        
        # Apply the best flip
        if best_var_idx is not None:
            apply_flip(best_var_idx, assignment, clause_sat_count)
        
        return False
    
    # Track best assignment found so far
    best_assignment = None
    best_unsat_count = float('inf')
    restart_count = 0
    
    # Main loop
    while time.time() < deadline:
        # Adaptive noise: start high, decrease over restarts
        base_noise = 0.3
        noise = max(0.1, base_noise * (1.0 - 0.1 * min(restart_count, 10)))
        
        # For some restarts, use the best assignment as a starting point
        if best_assignment is not None and rng.random() < 0.3:
            assignment = best_assignment[:]
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        
        clause_sat_count = compute_clause_sat_count(assignment)
        
        # Check if already satisfied
        if is_satisfied(clause_sat_count):
            return assignment
        
        # Update best if this is better
        current_unsat = sum(1 for c in clause_sat_count if c == 0)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        steps = 0
        while steps < max_steps_per_run:
            if time.time() >= deadline:
                return best_assignment if best_assignment is not None else assignment
            
            if is_satisfied(clause_sat_count):
                return assignment
            
            walksat_step(assignment, clause_sat_count, rng, noise)
            steps += 1
        
        # After each run, update best if improved
        current_unsat = sum(1 for c in clause_sat_count if c == 0)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        restart_count += 1
    
    return best_assignment if best_assignment is not None else assignment
