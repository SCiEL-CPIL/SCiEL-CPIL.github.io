import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Parameters
    max_steps_per_run = 10000
    base_noise = 0.25
    max_noise = 0.8
    
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
                currently_sat = is_literal_satisfied(lit, assignment)
                
                if currently_sat:
                    if old_count == 1:
                        newly_unsat += 1
                else:
                    if old_count == 0:
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
    
    # Main loop with adaptive noise and better restarts
    while time.time() < deadline:
        # Initialize assignment and clause_sat_count
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_sat_count = compute_clause_sat_count(assignment)
        
        # Check if already satisfied
        if is_satisfied(clause_sat_count):
            return assignment
        
        steps = 0
        # Adaptive noise: start low, increase if stuck
        current_noise = base_noise
        
        while steps < max_steps_per_run:
            if time.time() >= deadline:
                return assignment
            
            if is_satisfied(clause_sat_count):
                return assignment
            
            # Check for stagnation: if no progress in last 200 steps, increase noise
            if steps > 0 and steps % 200 == 0:
                current_noise = min(max_noise, current_noise + 0.05)
            
            walksat_step(assignment, clause_sat_count, rng, current_noise)
            steps += 1
    
    return assignment
