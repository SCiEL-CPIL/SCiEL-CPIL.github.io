"""Planted 3-SAT solver with optimized WalkSAT using NumPy."""
import numpy as np
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    time_limit = 290  # Leave margin

    # Preprocess clauses into NumPy arrays
    # clauses: list of lists of signed ints (1-based)
    # Convert to 0-based indices and polarities
    num_clauses = len(clauses)
    if num_clauses == 0:
        return [False] * n

    literals = np.zeros((num_clauses, 3), dtype=np.int32)
    polarities = np.zeros((num_clauses, 3), dtype=np.int32)
    
    for i, c in enumerate(clauses):
        for j, l in enumerate(c):
            literals[i, j] = abs(l) - 1
            polarities[i, j] = 1 if l > 0 else 0

    def get_unsat_clauses(assignment):
        """
        Returns indices of unsatisfied clauses.
        assignment: np.array of bools, length n
        """
        # assignment[literals] gives the current truth values of the literals
        # We compare with polarities
        # If they match, the literal is satisfied.
        # If any literal in the clause is satisfied, the clause is satisfied.
        lit_vals = assignment[literals]
        sat_literals = (lit_vals == polarities.astype(bool))
        clause_satisfied = sat_literals.any(axis=1)
        return np.where(~clause_satisfied)[0]

    def walksat(seed, max_steps, p):
        """
        WalkSAT implementation.
        p: probability of flipping a variable to satisfy the selected clause.
        """
        rng = np.random.default_rng(seed)
        # Random initial assignment
        assignment = rng.random(n) < 0.5
        
        # Check if initially satisfied
        unsat = get_unsat_clauses(assignment)
        if len(unsat) == 0:
            return assignment

        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                return None

            # Select a random unsatisfied clause
            if len(unsat) == 0:
                return assignment
            
            clause_idx = unsat[rng.integers(len(unsat))]
            
            # Select a variable from this clause
            lit_pos = rng.integers(3)
            var = literals[clause_idx, lit_pos]
            
            # Decide flip
            # If we flip to satisfy the clause:
            target_val = bool(polarities[clause_idx, lit_pos])
            
            if rng.random() < p:
                assignment[var] = target_val
            else:
                assignment[var] = not assignment[var]
                
            # Update unsat clauses
            unsat = get_unsat_clauses(assignment)
            
            if len(unsat) == 0:
                return assignment
                
        return None

    # Strategy: Restarting WalkSAT
    # Start with small step limit, increase exponentially.
    # Also vary p slightly.
    
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]
    p_values = [0.5, 0.6, 0.4, 0.7, 0.3]
    
    max_steps = 100
    max_steps_cap = 50000
    
    for seed in seeds:
        if time.time() - start_time > time_limit:
            break
            
        for p in p_values:
            if time.time() - start_time > time_limit:
                break
                
            # Restart strategy
            steps = max_steps
            while steps < max_steps_cap:
                if time.time() - start_time > time_limit:
                    break
                    
                result = walksat(seed, steps, p)
                if result is not None:
                    return result.tolist()
                    
                steps = min(steps * 2, max_steps_cap)
                
    # Fallback: longer runs with fixed p=0.5
    for i in range(10):
        if time.time() - start_time > time_limit:
            break
        seed = 10000 + i * 100
        result = walksat(seed, 100000, 0.5)
        if result is not None:
            return result.tolist()
            
    # Final fallback: random assignment
    rng = np.random.default_rng(0)
    return (rng.random(n) < 0.5).tolist()
