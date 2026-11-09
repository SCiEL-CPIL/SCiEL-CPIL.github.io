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
    def flip_var(v, x, sat_counts):
        old_val = x[v]
        new_val = not old_val
        x[v] = new_val
        
        for c_idx in var_to_clauses[v]:
            # Find the sign of v in clause c_idx
            # We need to know the sign. We can store it in a parallel structure or look it up.
            # For efficiency, let's store clause_data as two arrays or a dict?
            # Since n is small, linear search in c_data is fine, but let's optimize.
            # Actually, we can just check the condition directly.
            # But we need the sign to know if the flip satisfies or unsatisfies.
            # Let's precompute a matrix or just use the clause_data list.
            pass # Placeholder, will implement in main loop for clarity or precompute signs
            
    # Better precomputation:
    # clause_lit_var[c][k] = var index
    # clause_lit_sign[c][k] = sign
    # But variable indices in a clause are not contiguous.
    
    # Let's stick to the list of tuples in clause_data.
    # To speed up flip, we can precompute for each variable v, a list of (clause_idx, sign_in_clause).
    var_clause_signs = [[] for _ in range(n)]
    for c_idx, c_data in enumerate(clause_data):
        for v, s in c_data:
            var_clause_signs[v].append((c_idx, s))

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
    max_restarts = 100
    steps_per_restart = 5000
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random assignment
        x = [rng.random() < 0.5 for _ in range(n)]
        sat_counts = compute_sat(x)
        
        # Check if already solved
        if np.all(sat_counts > 0):
            return x
            
        # Local Search
        for step in range(steps_per_restart):
            if time.time() > deadline:
                return x # Return best effort (last assignment)
                
            # Find unsatisfied clauses
            unsat_clauses = np.where(sat_counts == 0)[0]
            if len(unsat_clauses) == 0:
                return x
                
            # Pick a random unsatisfied clause
            c_idx = rng.choice(unsat_clauses)
            c_data = clause_data[c_idx]
            
            # Heuristic: WalkSAT
            # With prob 0.5, random lit. Else, best gain lit.
            if rng.random() < 0.5:
                v, s = rng.choice(c_data)
            else:
                best_v = -1
                best_gain = -100
                for v, s in c_data:
                    # Calculate gain
                    gain = 0
                    for oc_idx, os in var_clause_signs[v]:
                        if oc_idx == c_idx:
                            continue
                        # Current state of this literal in other clause
                        old_val = x[v]
                        was_sat = (old_val and os == 1) or (not old_val and os == -1)
                        # New state
                        new_val = not old_val
                        is_sat = (new_val and os == 1) or (not new_val and os == -1)
                        
                        if was_sat and not is_sat:
                            gain -= 1
                        elif not was_sat and is_sat:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                v = best_v
                
            flip_var_optimized(v, x, sat_counts)
            
    # If we exhaust restarts, return the last assignment
    return x
