import time
import random
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 40s margin
    
    # Preprocessing
    # Convert clauses to 0-based variables and signs (+1/-1)
    # clause_data[c] = list of (var_idx, sign)
    clause_data = []
    var_to_clauses = [[] for _ in range(n)]
    
    for c_idx, clause in enumerate(clauses):
        c_data = []
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            c_data.append((v, s))
            var_to_clauses[v].append(c_idx)
        clause_data.append(c_data)
        
    # Precompute variable-clause signs for fast updates
    # var_clause_signs[v] = list of (clause_idx, sign)
    var_clause_signs = [[] for _ in range(n)]
    for c_idx, c_data in enumerate(clause_data):
        for v, s in c_data:
            var_clause_signs[v].append((c_idx, s))

    # Function to compute initial clause_sat counts
    def compute_sat(x):
        sat_counts = np.zeros(m, dtype=int)
        for c_idx, c_data in enumerate(clause_data):
            count = 0
            for v, s in c_data:
                if (x[v] and s == 1) or (not x[v] and s == -1):
                    count += 1
            sat_counts[c_idx] = count
        return sat_counts

    # Function to flip variable v and update sat_counts
    def flip_var_optimized(v, x, sat_counts):
        old_val = x[v]
        new_val = not old_val
        x[v] = new_val
        
        for c_idx, s in var_clause_signs[v]:
            # Was it satisfied before?
            was_sat = (old_val and s == 1) or (not old_val and s == -1)
            # Is it satisfied now?
            is_sat = (new_val and s == 1) or (not new_val and s == -1)
            
            if was_sat and not is_sat:
                sat_counts[c_idx] -= 1
            elif not was_sat and is_sat:
                sat_counts[c_idx] += 1
                
    # Main Search Loop
    rng = random.Random(42)
    
    # Try multiple restarts
    max_restarts = 200
    steps_per_restart = 10000
    
    # DLS Heuristic:
    # For an unsatisfied clause C, pick the variable v in C that appears in the most other unsatisfied clauses.
    # We need to maintain a count of how many unsatisfied clauses each variable appears in.
    # Let unsat_var_count[v] be the number of unsatisfied clauses containing v.
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random assignment
        x = [rng.random() < 0.5 for _ in range(n)]
        sat_counts = compute_sat(x)
        
        # Check if already solved
        if np.all(sat_counts > 0):
            return x
            
        # Initialize unsat_var_count
        unsat_clauses_mask = (sat_counts == 0)
        unsat_var_count = np.zeros(n, dtype=int)
        for v in range(n):
            count = 0
            for c_idx, s in var_clause_signs[v]:
                if unsat_clauses_mask[c_idx]:
                    count += 1
            unsat_var_count[v] = count
            
        # Local Search
        for step in range(steps_per_restart):
            if time.time() > deadline:
                return x # Return best effort (last assignment)
                
            # Find unsatisfied clauses
            unsat_indices = np.where(sat_counts == 0)[0]
            if len(unsat_indices) == 0:
                return x
                
            # Pick a random unsatisfied clause
            c_idx = rng.choice(unsat_indices)
            c_data = clause_data[c_idx]
            
            # DLS: Select variable in c_data with max unsat_var_count
            # Tie-break randomly
            max_count = -1
            best_candidates = []
            for v, s in c_data:
                cnt = unsat_var_count[v]
                if cnt > max_count:
                    max_count = cnt
                    best_candidates = [v]
                elif cnt == max_count:
                    best_candidates.append(v)
            
            v = rng.choice(best_candidates)
            
            # Flip variable v
            old_val = x[v]
            new_val = not old_val
            x[v] = new_val
            
            # Update sat_counts and unsat_var_count
            for c_idx2, s in var_clause_signs[v]:
                was_sat = (old_val and s == 1) or (not old_val and s == -1)
                is_sat = (new_val and s == 1) or (not new_val and s == -1)
                
                if was_sat and not is_sat:
                    # Clause becomes unsatisfied
                    sat_counts[c_idx2] -= 1
                    if sat_counts[c_idx2] == 0:
                        # It just became unsatisfied, increment counts for all vars in this clause
                        for v2, s2 in clause_data[c_idx2]:
                            unsat_var_count[v2] += 1
                elif not was_sat and is_sat:
                    # Clause becomes satisfied
                    sat_counts[c_idx2] += 1
                    if sat_counts[c_idx2] == 1:
                        # It just became satisfied, decrement counts for all vars in this clause
                        for v2, s2 in clause_data[c_idx2]:
                            unsat_var_count[v2] -= 1

    # If we exhaust restarts, return the last assignment
    return x
