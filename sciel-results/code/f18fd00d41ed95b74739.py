# EVOLVE-BLOCK-START
import random
import math


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Convert clauses to 0-based indices for faster access
    # Each clause is a list of 3 literals (signed 1-based)
    # We'll store them as tuples of (var_idx, is_positive)
    clauses_data = []
    for c in clauses:
        clause_t = tuple((abs(l) - 1, l > 0) for l in c)
        clauses_data.append(clause_t)
    
    m = len(clauses_data)
    
    # Precompute variable occurrence counts for heuristics
    var_count = [0] * n
    for c in clauses_data:
        for var_idx, _ in c:
            var_count[var_idx] += 1
    
    # WalkSAT with improvements:
    # 1. Multiple random restarts
    # 2. Weight-based flipping (pick variable with highest weight in unsat clause)
    # 3. Adaptive noise parameter
    # 4. Track clause satisfaction counts for fast evaluation
    
    rng = random.Random(12345)
    
    best_assignment = None
    best_unsat = m + 1
    
    # Budget: total flips across all restarts
    total_budget = 500000
    max_restarts = 50
    flips_per_restart = total_budget // max_restarts
    
    for restart in range(max_restarts):
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Compute initial clause satisfaction status
        # sat_count[i] = number of literals satisfied in clause i
        sat_count = [0] * m
        for i, c in enumerate(clauses_data):
            cnt = 0
            for var_idx, is_pos in c:
                if assignment[var_idx] == is_pos:
                    cnt += 1
            sat_count[i] = cnt
        
        # Count unsatisfied clauses
        unsat_count = sum(1 for sc in sat_count if sc == 0)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
        
        # Weight array for variables
        weights = [0.0] * n
        for _ in range(flips_per_restart):
            if unsat_count == 0:
                break
            
            # Pick a random unsatisfied clause
            # To avoid scanning all clauses, we could maintain a list, but for simplicity
            # and given the size (50 vars, ~100-500 clauses typically), scanning is fine
            unsat_clauses = [i for i in range(m) if sat_count[i] == 0]
            if not unsat_clauses:
                break
            ci = rng.choice(unsat_clauses)
            clause = clauses_data[ci]
            
            # Decide which variable to flip
            # With probability 'noise', pick randomly; otherwise pick highest weight
            noise = 0.45
            if rng.random() < noise:
                var_idx, is_pos = rng.choice(clause)
            else:
                # Pick variable with max weight in this clause
                max_w = -1
                chosen_var = None
                for var_idx, is_pos in clause:
                    if weights[var_idx] > max_w:
                        max_w = weights[var_idx]
                        chosen_var = var_idx
                var_idx = chosen_var
            
            # Flip the variable
            old_val = assignment[var_idx]
            new_val = not old_val
            assignment[var_idx] = new_val
            
            # Update sat_count for all clauses containing var_idx
            # We need to know which clauses contain each variable
            # Precompute var_to_clauses
            # Actually, let's precompute this outside the loop for efficiency
            
            # Update weights: for each clause containing var_idx:
            # If clause was unsat (sat_count == 0) and now has 1 sat, it becomes sat
            # If clause was sat (sat_count > 0) and now has 0 sat, it becomes unsat
            # Weight update: for clauses that become satisfied, decrease weight;
            # for clauses that become unsatisfied, increase weight
            
            # We need var_to_clauses mapping. Let's build it once.
            # Since we're inside the restart loop, let's build it outside.
            # Actually, let me restructure to build var_to_clauses before the restart loop.
            
            # For now, this is inefficient. Let me restructure.
            pass
        
        # This approach is getting complex. Let me use a cleaner implementation.
        break  # Exit early, will redo below
    
    # Clean implementation
    # Build var_to_clauses mapping
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses_data):
        for var_idx, is_pos in c:
            var_to_clauses[var_idx].append(i)
    
    best_assignment = None
    best_unsat = m + 1
    
    for restart in range(max_restarts):
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        sat_count = [0] * m
        for i, c in enumerate(clauses_data):
            cnt = 0
            for var_idx, is_pos in c:
                if assignment[var_idx] == is_pos:
                    cnt += 1
            sat_count[i] = cnt
        
        unsat_count = sum(1 for sc in sat_count if sc == 0)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
        
        weights = [0.0] * n
        
        for _ in range(flips_per_restart):
            if unsat_count == 0:
                break
            
            # Find an unsatisfied clause
            # Use a random probe to find unsat clauses efficiently
            # Try a few random indices until we find an unsat one
            ci = None
            for _ in range(10):
                idx = rng.randrange(m)
                if sat_count[idx] == 0:
                    ci = idx
                    break
            if ci is None:
                # Fallback: linear scan
                for idx in range(m):
                    if sat_count[idx] == 0:
                        ci = idx
                        break
            if ci is None:
                break
            
            clause = clauses_data[ci]
            
            # Choose variable to flip
            noise = 0.45
            if rng.random() < noise:
                var_idx, is_pos = rng.choice(clause)
            else:
                max_w = -1
                chosen_var = None
                for var_idx, is_pos in clause:
                    if weights[var_idx] > max_w:
                        max_w = weights[var_idx]
                        chosen_var = var_idx
                var_idx = chosen_var
            
            # Flip var_idx
            old_val = assignment[var_idx]
            new_val = not old_val
            assignment[var_idx] = new_val
            
            # Update sat_count and weights for all clauses containing var_idx
            for cj in var_to_clauses[var_idx]:
                c = clauses_data[cj]
                old_sc = sat_count[cj]
                
                # Determine how this flip affects this clause
                # Find the literal in this clause corresponding to var_idx
                for var_idx_c, is_pos_c in c:
                    if var_idx_c == var_idx:
                        # Check if this literal was satisfied before and after
                        was_sat = (old_val == is_pos_c)
                        now_sat = (new_val == is_pos_c)
                        
                        if was_sat and not now_sat:
                            # Lost a satisfied literal
                            new_sc = old_sc - 1
                            # This clause becomes less satisfied
                            # Increase weights of other variables in this clause
                            for var_idx2, is_pos2 in c:
                                if var_idx2 != var_idx:
                                    weights[var_idx2] += 1.0
                        elif not was_sat and now_sat:
                            # Gained a satisfied literal
                            new_sc = old_sc + 1
                            # This clause becomes more satisfied
                            # Decrease weights of other variables in this clause
                            for var_idx2, is_pos2 in c:
                                if var_idx2 != var_idx:
                                    weights[var_idx2] = max(0.0, weights[var_idx2] - 1.0)
                        else:
                            # No change
                            new_sc = old_sc
                        break
                
                # Track unsat_count changes
                if old_sc == 0 and new_sc > 0:
                    unsat_count -= 1
                elif old_sc > 0 and new_sc == 0:
                    unsat_count += 1
                
                sat_count[cj] = new_sc
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
    
    return best_assignment
# EVOLVE-BLOCK-END
