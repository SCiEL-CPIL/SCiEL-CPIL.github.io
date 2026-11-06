import time
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    start_time = time.time()
    TIME_LIMIT = 320.0
    
    # Precompute variable to clauses mapping
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            var_to_clauses[abs(l)-1].append(i)
    
    rng = random.Random(0)
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    # Initialize unsat counts
    unsat_count = [0] * len(clauses)
    for i, c in enumerate(clauses):
        for l in c:
            if assignment[abs(l)-1] != (l > 0):
                unsat_count[i] += 1
    
    def is_unsat(i):
        return unsat_count[i] == len(clauses[i])
    
    def flip(var_idx):
        nonlocal assignment
        old_val = assignment[var_idx]
        assignment[var_idx] = not old_val
        
        for c_idx in var_to_clauses[var_idx]:
            # Find literal in clause c_idx for var_idx
            lit = 0
            for l in clauses[c_idx]:
                if abs(l) - 1 == var_idx:
                    lit = l
                    break
            
            if lit > 0:
                if old_val:
                    unsat_count[c_idx] += 1
                else:
                    unsat_count[c_idx] -= 1
            else:
                if not old_val:
                    unsat_count[c_idx] += 1
                else:
                    unsat_count[c_idx] -= 1
    
    # Luby sequence
    def get_luby(i):
        # Simple Luby implementation
        # L(0)=1, L(1)=1, L(2)=2, L(3)=1, L(4)=1, L(5)=2, L(6)=4...
        # I'll use a precomputed list for the first 1000 values
        # Or a simple formula.
        # Let's use: 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8...
        # This is 2^k where k is the exponent of the highest power of 2 dividing (i+1)?
        # No.
        # I'll just use a doubling strategy.
        return 1 << (i.bit_length() - 1)
    
    restart_idx = 0
    base_budget = 500
    
    while True:
        if time.time() - start_time > TIME_LIMIT:
            break
        
        # Find unsat clauses
        unsat_clauses = [i for i in range(len(clauses)) if is_unsat(i)]
        
        if not unsat_clauses:
            return assignment
        
        budget = int(base_budget * (1 + restart_idx * 0.5))
        
        for _ in range(budget):
            if time.time() - start_time > TIME_LIMIT:
                break
            
            unsat_clauses = [i for i in range(len(clauses)) if is_unsat(i)]
            if not unsat_clauses:
                return assignment
            
            c_idx = rng.choice(unsat_clauses)
            clause = clauses[c_idx]
            
            # Pick variable with min damage
            best_var = -1
            best_damage = float('inf')
            
            for l in clause:
                var_idx = abs(l) - 1
                damage = 0
                old_val = assignment[var_idx]
                
                for c_idx2 in var_to_clauses[var_idx]:
                    was_unsat = is_unsat(c_idx2)
                    
                    # Simulate flip effect
                    # Find literal
                    lit2 = 0
                    for l2 in clauses[c_idx2]:
                        if abs(l2) - 1 == var_idx:
                            lit2 = l2
                            break
                    
                    if lit2 > 0:
                        if old_val:
                            # True -> False
                            if not was_unsat and unsat_count[c_idx2] + 1 == len(clauses[c_idx2]):
                                damage += 1
                        else:
                            # False -> True
                            if was_unsat:
                                damage -= 1
                    else:
                        if not old_val:
                            # True -> False
                            if not was_unsat and unsat_count[c_idx2] + 1 == len(clauses[c_idx2]):
                                damage += 1
                        else:
                            # False -> True
                            if was_unsat:
                                damage -= 1
                                
                if damage < best_damage:
                    best_damage = damage
                    best_var = var_idx
            
            if best_var != -1:
                flip(best_var)
                
        # Restart
        assignment = [rng.random() < 0.5 for _ in range(n)]
        for i in range(len(clauses)):
            unsat_count[i] = 0
            for l in clauses[i]:
                if assignment[abs(l)-1] != (l > 0):
                    unsat_count[i] += 1
        restart_idx += 1
        
    return assignment
