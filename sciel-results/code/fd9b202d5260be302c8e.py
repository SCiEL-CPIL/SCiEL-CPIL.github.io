import sys
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    rng = random.Random(42)
    
    # WalkSAT parameters
    max_restarts = 5000
    initial_steps = 500
    step_multiplier = 1.2
    
    for restart in range(max_restarts):
        # Random initialization
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        def is_satisfied(clause, assignment):
            for lit in clause:
                var = abs(lit) - 1
                val = assignment[var]
                is_true = (lit > 0)
                if val == is_true:
                    return True
            return False

        steps_limit = initial_steps
        for step in range(steps_limit):
            # Find unsatisfied clauses
            unsat_clauses = []
            for c in clauses:
                if not is_satisfied(c, assignment):
                    unsat_clauses.append(c)
            
            if not unsat_clauses:
                return assignment
            
            # Count frequency of variables in unsat clauses
            var_freq = [0] * n
            for c in unsat_clauses:
                for lit in c:
                    var = abs(lit) - 1
                    var_freq[var] += 1
            
            # Pick a random unsat clause
            clause = rng.choice(unsat_clauses)
            
            # Select variable to flip from this clause
            # Weighted by frequency in unsat clauses
            weights = [var_freq[abs(lit)-1] for lit in clause]
            total_weight = sum(weights)
            
            if total_weight == 0:
                # Fallback to random if weights are 0 (should not happen)
                var = abs(rng.choice(clause)) - 1
            else:
                # Weighted choice
                r = rng.random() * total_weight
                cumulative = 0
                var = abs(clause[0]) - 1
                for lit, w in zip(clause, weights):
                    cumulative += w
                    if r <= cumulative:
                        var = abs(lit) - 1
                        break
                
            # Flip
            assignment[var] = not assignment[var]
            
        # Increase steps limit for next restart
        steps_limit = int(steps_limit * step_multiplier)
        
    # If all restarts fail, return last assignment
    return assignment
