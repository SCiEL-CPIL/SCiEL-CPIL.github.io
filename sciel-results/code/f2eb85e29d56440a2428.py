# EVOLVE-BLOCK-START
import random
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    
    # Pre-process clauses: convert to 0-based indices and store sign
    processed_clauses = []
    for c in clauses:
        pc = []
        for lit in c:
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            pc.append((var_idx, sign))
        processed_clauses.append(pc)
    
    num_clauses = len(processed_clauses)
    
    # Pre-compute variable to clause mapping
    var_to_clauses = [[] for _ in range(n)]
    for i, pc in enumerate(processed_clauses):
        for var_idx, sign in pc:
            var_to_clauses[var_idx].append((i, sign))
    
    # Initialize random assignment
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    # Compute initial sat_count and unsat_count
    sat_count = [0] * num_clauses
    for i, pc in enumerate(processed_clauses):
        for var_idx, sign in pc:
            if assignment[var_idx] == (sign == 1):
                sat_count[i] += 1
    
    unsat_count = sum(1 for i in range(num_clauses) if sat_count[i] == 0)
    
    if unsat_count == 0:
        return assignment
    
    # Main search loop
    max_flips = 200000
    
    for step in range(max_flips):
        # Find unsatisfied clauses
        unsat_indices = [i for i in range(num_clauses) if sat_count[i] == 0]
        
        if not unsat_indices:
            break
        
        # Compute variable frequencies in unsat clauses
        var_freq = [0] * n
        for i in unsat_indices:
            for var_idx, sign in processed_clauses[i]:
                var_freq[var_idx] += 1
        
        # Choose an unsatisfied clause
        clause_idx = rng.choice(unsat_indices)
        clause_vars = processed_clauses[clause_idx]
        
        # Weight variables inversely by frequency
        weights = []
        for var_idx, sign in clause_vars:
            freq = var_freq[var_idx]
            w = 1.0 / (1.0 + freq)
            weights.append(w)
        
        total_w = sum(weights)
        if total_w == 0:
            weights = [1.0] * len(clause_vars)
            total_w = len(clause_vars)
        
        r = rng.random() * total_w
        cumulative = 0.0
        chosen_var_idx = clause_vars[0][0]
        for k, (var_idx, sign) in enumerate(clause_vars):
            cumulative += weights[k]
            if r <= cumulative:
                chosen_var_idx = var_idx
                break
        else:
            chosen_var_idx = clause_vars[-1][0]
        
        # Flip the variable
        old_val = assignment[chosen_var_idx]
        new_val = not old_val
        assignment[chosen_var_idx] = new_val
        
        # Update sat_count for clauses containing this variable
        for clause_i, sign in var_to_clauses[chosen_var_idx]:
            old_lit_sat = old_val == (sign == 1)
            new_lit_sat = new_val == (sign == 1)
            if old_lit_sat and not new_lit_sat:
                sat_count[clause_i] -= 1
                if sat_count[clause_i] == 0:
                    unsat_count += 1
            elif not old_lit_sat and new_lit_sat:
                sat_count[clause_i] += 1
                if sat_count[clause_i] == 1:
                    unsat_count -= 1
        
        if unsat_count == 0:
            break
    
    return assignment
# EVOLVE-BLOCK-END
