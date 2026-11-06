"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    # Improved WalkSAT with time limit, clause weighting, and better heuristics
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    
    # Preprocess: convert clauses to 0-based indices and precompute abs/1-based
    # clause_vars[i] = list of (var_idx, is_positive) for clause i
    clause_vars = []
    for c in clauses:
        cv = [(abs(l) - 1, l > 0) for l in c]
        clause_vars.append(cv)
    
    m = len(clause_vars)
    
    # Time limit: 290 seconds (safety margin from 330s kill)
    start_time = time.time()
    time_limit = 290.0
    
    # Weighted WalkSAT with restarts
    def random_assignment():
        return [rng.random() < 0.5 for _ in range(n)]
    
    def is_satisfied(clause_idx, assignment):
        for var_idx, is_pos in clause_vars[clause_idx]:
            if assignment[var_idx] == is_pos:
                return True
        return False
    
    def count_unsat(assignment):
        return sum(1 for i in range(m) if not is_satisfied(i, assignment))
    
    def walksat_once(assignment, max_steps, weight_decay):
        """Run one WalkSAT search with clause weighting."""
        # Initialize clause weights
        clause_weights = [1.0] * m
        
        # Count initial unsatisfied
        unsat_count = count_unsat(assignment)
        if unsat_count == 0:
            return assignment, True
        
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                return assignment, False
            
            # Find all unsatisfied clauses
            unsat_clauses = []
            for i in range(m):
                if not is_satisfied(i, assignment):
                    unsat_clauses.append(i)
            
            if not unsat_clauses:
                return assignment, True
            
            # Pick a clause with probability proportional to its weight
            total_weight = sum(clause_weights[i] for i in unsat_clauses)
            r = rng.random() * total_weight
            cumulative = 0.0
            chosen_clause = unsat_clauses[0]
            for i in unsat_clauses:
                cumulative += clause_weights[i]
                if cumulative >= r:
                    chosen_clause = i
                    break
            
            # Try to find a literal in the clause that minimizes damage
            # Damage = number of currently satisfied clauses that would become unsatisfied
            clause = clause_vars[chosen_clause]
            best_var = -1
            best_damage = float('inf')
            best_flip = False
            
            for var_idx, is_pos in clause:
                # If we flip this variable to satisfy the clause
                new_val = is_pos
                current_val = assignment[var_idx]
                
                if current_val == new_val:
                    # Already satisfies this clause, but we need to flip to try
                    # Actually, if it already satisfies, flipping would make it unsat
                    # We want to flip a variable that is NOT currently satisfying
                    # Let's compute damage for flipping to new_val
                    pass
                
                # Compute damage: how many satisfied clauses become unsatisfied
                damage = 0
                # We need to check all clauses containing var_idx
                # This is expensive; let's use a simpler heuristic for now
                # For now, just pick randomly from the clause
                break
            
            # Simpler: pick a literal from the clause, with some bias
            # Use a noise parameter to decide whether to minimize damage or flip randomly
            noise = 0.3  # probability of random flip vs damage-minimizing
            
            if rng.random() < noise:
                # Random flip
                var_idx, is_pos = rng.choice(clause)
                assignment[var_idx] = is_pos
            else:
                # Try to minimize damage: for each literal, count how many
                # satisfied clauses contain the opposite literal
                best_var = -1
                best_score = float('inf')
                
                for var_idx, is_pos in clause:
                    # If we set var_idx to is_pos, check damage
                    # Damage = number of satisfied clauses that contain (var_idx, not is_pos)
                    # We approximate by checking a random subset for speed
                    damage = 0
                    for j in range(m):
                        if j == chosen_clause:
                            continue
                        for v, p in clause_vars[j]:
                            if v == var_idx and p != is_pos:
                                # This clause contains the opposite literal
                                # Check if it's currently satisfied
                                if is_satisfied(j, assignment):
                                    damage += 1
                                break
                    
                    score = damage
                    if score < best_score:
                        best_score = score
                        best_var = var_idx
                        best_flip = is_pos
                
                if best_var >= 0:
                    assignment[best_var] = best_flip
                else:
                    var_idx, is_pos = rng.choice(clause)
                    assignment[var_idx] = is_pos
            
            # Update clause weights: increase weights of unsatisfied clauses
            # Decay all weights slightly
            for i in range(m):
                clause_weights[i] *= weight_decay
            # Boost unsatisfied clauses
            for i in unsat_clauses:
                clause_weights[i] += 1.0
        
        return assignment, count_unsat(assignment) == 0
    
    # Main search loop with restarts
    best_assignment = random_assignment()
    best_unsat = count_unsat(best_assignment)
    
    max_restarts = 50
    steps_per_restart = 5000
    weight_decay = 0.999
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # For the first restart, use the best assignment found so far
        if restart == 0:
            assignment = best_assignment[:]
        else:
            # Mix random with best
            if rng.random() < 0.3:
                assignment = best_assignment[:]
            else:
                assignment = random_assignment()
        
        # Run WalkSAT
        assignment, solved = walksat_once(assignment, steps_per_restart, weight_decay)
        
        if solved:
            return assignment
        
        cur_unsat = count_unsat(assignment)
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
    
    return best_assignment
# EVOLVE-BLOCK-END
