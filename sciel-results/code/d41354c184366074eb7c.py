"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 280.0  # safety margin before 300s kill

    # Convert clauses to 0-based, signed literals
    # We'll use a hybrid approach: WalkSAT with smarter heuristics + clause learning lite
    # For planted 3-SAT with 150 vars, a well-tuned WalkSAT or a DPLL with unit propagation
    # should work well. Let's implement a DPLL solver with unit propagation and pure literal elimination.
    
    # Actually, for planted satisfiable instances, a local search (WalkSAT) with good heuristics
    # often works very well. Let's implement an enhanced WalkSAT with:
    # 1. Multiple restarts with different seeds
    # 2. Better variable selection (weight-based)
    # 3. Clause weight tracking to focus on hard clauses
    
    best_assignment = None
    best_score = -1
    
    for seed in range(50):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Track clause weights (frequency of being unsatisfied)
        clause_weights = [1.0] * len(clauses)
        
        def is_satisfied(clause, assign):
            return any(assign[abs(l) - 1] == (l > 0) for l in clause)
        
        def count_unsat(clause, assign):
            return sum(1 for l in clause if not (assign[abs(l) - 1] == (l > 0)))
        
        # WalkSAT loop
        max_steps = 30000
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Find unsatisfied clauses
            unsat = []
            for i, c in enumerate(clauses):
                if not is_satisfied(c, assignment):
                    unsat.append(i)
            
            if not unsat:
                # Found a solution!
                return assignment
            
            # Select a clause to focus on, weighted by clause weight
            # Use weighted random selection
            total_weight = sum(clause_weights[i] for i in unsat)
            r = rng.random() * total_weight
            cumulative = 0
            selected_idx = unsat[0]
            for i in unsat:
                cumulative += clause_weights[i]
                if r <= cumulative:
                    selected_idx = i
                    break
            
            clause = clauses[selected_idx]
            
            # Update clause weights
            # Increase weight of unsatisfied clauses, decrease satisfied ones
            for i, c in enumerate(clauses):
                if is_satisfied(c, assignment):
                    clause_weights[i] = max(0.1, clause_weights[i] * 0.999)
                else:
                    clause_weights[i] = min(100.0, clause_weights[i] * 1.001)
            
            # Select a literal to flip from the chosen clause
            # Use a heuristic: flip the literal that causes the least damage
            # (i.e., the variable whose flip satisfies the most other unsatisfied clauses)
            
            best_literal = None
            best_score_for_literal = -1
            
            for lit in clause:
                var = abs(lit) - 1
                # Simulate flip
                old_val = assignment[var]
                assignment[var] = not old_val
                
                # Score: how many clauses are now satisfied that were unsatisfied before
                # and how many satisfied clauses become unsatisfied
                score = 0
                for c in clauses:
                    was_sat = is_satisfied(c, assignment)
                    # We need to know if it was sat before the flip
                    # This is expensive, so use a simpler heuristic
                assignment[var] = old_val  # revert
                
                # Simpler heuristic: pick random literal from clause
                # But let's use a 1-step lookahead for the chosen clause's literals
                pass
            
            # Use a simpler but effective heuristic: 
            # With probability 0.5, pick the literal that appears in the most unsatisfied clauses
            # Otherwise pick random
            if rng.random() < 0.5:
                # Count how many unsatisfied clauses each literal in this clause appears in
                lit_scores = {}
                for lit in clause:
                    lit_scores[lit] = 0
                for i in unsat:
                    for lit in clauses[i]:
                        if lit in lit_scores:
                            lit_scores[lit] += 1
                # Pick the literal with highest score (appears in most unsat clauses)
                best_lit = max(lit_scores, key=lit_scores.get)
                assignment[abs(best_lit) - 1] = not assignment[abs(best_lit) - 1]
            else:
                lit = rng.choice(clause)
                assignment[abs(lit) - 1] = not assignment[abs(lit) - 1]
            
            # Track best assignment seen
            num_unsat = len(unsat)
            if num_unsat < best_score:
                best_score = num_unsat
                best_assignment = assignment[:]
        
        # If we found a solution in any restart, return it
        # Check if current assignment satisfies all
        all_sat = all(is_satisfied(c, assignment) for c in clauses)
        if all_sat:
            return assignment
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    return assignment
# EVOLVE-BLOCK-END
