import time
import random
import sys
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 325.0  # Safety margin

    if m == 0:
        return [True] * n

    # Preprocessing
    var_occ = [[] for _ in range(n)]
    clause_vars = [0] * m
    clause_lits = [0] * m
    
    for i, clause in enumerate(clauses):
        clause_lits[i] = clause
        vars_in_clause = []
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_occ[v].append((i, sign))
            vars_in_clause.append(v)
        clause_vars[i] = vars_in_clause

    def is_lit_true(lit, assignment):
        v = abs(lit) - 1
        if lit > 0:
            return assignment[v]
        else:
            return not assignment[v]

    rng = random.Random(42)
    
    best_assignment = None
    best_unsat = m + 1
    
    max_restarts = 1000  # Increase restarts
    max_steps_per_restart = 1000  # Decrease steps, total budget similar
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random Initialization
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Calculate initial sat_counts using NumPy for speed
        # sat_counts[i] = number of true literals in clause i
        sat_counts = np.zeros(m, dtype=np.int8)
        for i in range(m):
            for lit in clause_lits[i]:
                if is_lit_true(lit, assignment):
                    sat_counts[i] += 1
        
        unsat_indices = np.nonzero(sat_counts == 0)[0]
        unsat_count = len(unsat_indices)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
        
        # WalkSAT Loop
        p_greedy = 0.5
        
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break
            
            if len(unsat_indices) == 0:
                return assignment
            
            if len(unsat_indices) < best_unsat:
                best_unsat = len(unsat_indices)
                best_assignment = assignment[:]
                
            # Pick a random unsatisfied clause
            c_idx = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]
            clause_lits_c = clause_lits[c_idx]
            
            # Determine which variable to flip
            best_gain = -100
            best_var = None
            gains = []
            
            for v in clause_vars[c_idx]:
                gain = 0
                val = assignment[v]
                for (c_prime, sign) in var_occ[v]:
                    # Determine if literal was true before flip
                    if sign == 1:
                        was_true = val
                    else:
                        was_true = not val
                    
                    if was_true:
                        # Flipping makes it False. sat_count decreases.
                        if sat_counts[c_prime] == 1:
                            gain -= 1
                    else:
                        # Flipping makes it True. sat_count increases.
                        if sat_counts[c_prime] == 0:
                            gain += 1
                gains.append(gain)
                if gain > best_gain:
                    best_gain = gain
                    best_var = v
            
            # Choose variable
            if rng.random() < p_greedy:
                max_g = best_gain
                candidates = [v for v, g in zip(clause_vars[c_idx], gains) if g == max_g]
                v_to_flip = candidates[rng.randint(0, len(candidates) - 1)]
            else:
                v_to_flip = clause_vars[c_idx][rng.randint(0, len(clause_vars[c_idx]) - 1)]
            
            # Flip v_to_flip
            assignment[v_to_flip] = not assignment[v_to_flip]
            
            # Update sat_counts
            for (c_prime, sign) in var_occ[v_to_flip]:
                # Determine new state
                if sign == 1:
                    is_true_now = assignment[v_to_flip]
                else:
                    is_true_now = not assignment[v_to_flip]
                
                if is_true_now:
                    sat_counts[c_prime] += 1
                else:
                    sat_counts[c_prime] -= 1
            
            # Update unsat_indices efficiently?
            # Instead of full re-scan, we can update the list.
            # But np.nonzero is fast. Let's just re-scan if the list is small?
            # Or just re-scan always.
            # m=750. np.nonzero is very fast.
            unsat_indices = np.nonzero(sat_counts == 0)[0]
    
    # Return best found
    if best_assignment is not None:
        return best_assignment
    
    # Fallback
    return [True] * n
