import random
import sys
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Seed based on instance for reproducibility
    seed_val = sum(abs(l) for c in clauses for l in c) % (2**32)
    rng = random.Random(seed_val)
    
    num_clauses = len(clauses)
    
    # Pre-compute clause structure for faster access
    # clause_vars[i] = list of variable indices (0-based)
    # clause_signs[i] = list of signs (+1 for positive, -1 for negative)
    clause_vars = []
    clause_signs = []
    for c in clauses:
        vars_list = []
        signs_list = []
        for l in c:
            vars_list.append(abs(l) - 1)
            signs_list.append(1 if l > 0 else -1)
        clause_vars.append(vars_list)
        clause_signs.append(signs_list)
    
    # Variable to clauses index
    var_clauses = [[] for _ in range(n)]
    for i in range(num_clauses):
        for v in clause_vars[i]:
            var_clauses[v].append(i)
    
    def count_unsat(assignment):
        count = 0
        for i in range(num_clauses):
            sat = False
            for j, v in enumerate(clause_vars[i]):
                if assignment[v] == (clause_signs[i][j] > 0):
                    sat = True
                    break
            if not sat:
                count += 1
        return count
    
    def get_unsat_clauses(assignment):
        unsat = []
        for i in range(num_clauses):
            sat = False
            for j, v in enumerate(clause_vars[i]):
                if assignment[v] == (clause_signs[i][j] > 0):
                    sat = True
                    break
            if not sat:
                unsat.append(i)
        return unsat
    
    # Weighted WalkSAT with clause weights
    def walksat(max_restarts, max_flips):
        clause_weights = [1.0] * num_clauses
        best_assignment = None
        best_unsat_count = float('inf')
        
        for restart in range(max_restarts):
            assignment = [rng.random() < 0.5 for _ in range(n)]
            
            # Initial unsat count
            curr_unsat = count_unsat(assignment)
            if curr_unsat == 0:
                return assignment
            if curr_unsat < best_unsat_count:
                best_unsat_count = curr_unsat
                best_assignment = assignment[:]
            
            for flip_count in range(max_flips):
                unsat_indices = get_unsat_clauses(assignment)
                if not unsat_indices:
                    return assignment
                
                curr_unsat = len(unsat_indices)
                if curr_unsat < best_unsat_count:
                    best_unsat_count = curr_unsat
                    best_assignment = assignment[:]
                
                # Pick an unsatisfied clause with probability proportional to weight
                total_weight = sum(clause_weights[i] for i in unsat_indices)
                r = rng.random() * total_weight
                current = 0.0
                chosen_idx = unsat_indices[0]
                for idx in unsat_indices:
                    current += clause_weights[idx]
                    if r <= current:
                        chosen_idx = idx
                        break
                
                clause_weights[chosen_idx] *= 1.1  # Increase weight of chosen clause
                clause = chosen_idx
                clause_vars_local = clause_vars[clause]
                clause_signs_local = clause_signs[clause]
                
                # Pick a variable to flip from the clause
                # Use a heuristic: prefer variables that appear in more unsat clauses
                # with some randomness to escape local minima
                
                # Count how many unsat clauses each variable in this clause appears in
                var_scores = {}
                for v in clause_vars_local:
                    score = 0
                    for ci in unsat_indices:
                        if v in clause_vars[ci]:
                            score += 1
                    var_scores[v] = score
                
                # Add randomness
                best_v = None
                best_score = -1
                for v in clause_vars_local:
                    score = var_scores[v] + rng.random() * 2.0
                    if score > best_score:
                        best_score = score
                        best_v = v
                
                # Flip the variable
                assignment[best_v] = not assignment[best_v]
            
        return best_assignment
    
    # Time budget: 290 seconds (leave 10s buffer)
    start_time = time.time()
    time_budget = 290
    
    # Phase 1: Try standard WalkSAT with moderate settings
    # Use a larger number of flips per restart and more restarts
    max_restarts = 100
    max_flips = 5000
    
    result = walksat(max_restarts, max_flips)
    
    if result is not None and count_unsat(result) == 0:
        return result
    
    # Phase 2: If Phase 1 failed, try with different parameters
    # More aggressive clause weighting
    clause_weights = [1.0] * num_clauses
    best_assignment = None
    best_unsat_count = float('inf')
    
    remaining_time = time_budget - (time.time() - start_time)
    
    if remaining_time > 30:
        # Try with higher clause weight growth
        for restart in range(50):
            if time.time() - start_time > time_budget - 10:
                break
            
            assignment = [rng.random() < 0.5 for _ in range(n)]
            curr_unsat = count_unsat(assignment)
            if curr_unsat == 0:
                return assignment
            if curr_unsat < best_unsat_count:
                best_unsat_count = curr_unsat
                best_assignment = assignment[:]
            
            for flip_count in range(3000):
                if time.time() - start_time > time_budget - 10:
                    break
                
                unsat_indices = get_unsat_clauses(assignment)
                if not unsat_indices:
                    return assignment
                
                curr_unsat = len(unsat_indices)
                if curr_unsat < best_unsat_count:
                    best_unsat_count = curr_unsat
                    best_assignment = assignment[:]
                
                # Pick clause
                total_weight = sum(clause_weights[i] for i in unsat_indices)
                r = rng.random() * total_weight
                current = 0.0
                chosen_idx = unsat_indices[0]
                for idx in unsat_indices:
                    current += clause_weights[idx]
                    if r <= current:
                        chosen_idx = idx
                        break
                
                clause_weights[chosen_idx] *= 1.2
                clause = chosen_idx
                
                # Pick variable with higher weight on unsat clauses
                var_scores = {}
                for v in clause_vars[clause]:
                    score = 0
                    for ci in unsat_indices:
                        if v in clause_vars[ci]:
                            score += 1
                    var_scores[v] = score
                
                best_v = None
                best_score = -1
                for v in clause_vars[clause]:
                    score = var_scores[v] + rng.random() * 3.0
                    if score > best_score:
                        best_score = score
                        best_v = v
                
                assignment[best_v] = not assignment[best_v]
    
    return best_assignment
