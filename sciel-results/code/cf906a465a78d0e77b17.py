# EVOLVE-BLOCK-START
import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline: 290 seconds to be safe within 300s limit
    deadline = time.time() + 290.0
    
    # Precompute variable to clause indices mapping
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            v = abs(l) - 1
            var_to_clauses[v].append(i)
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    max_restarts = 2000
    restart_threshold = 2000
    max_steps_per_restart = 50000
    
    rng = random.Random(42)
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize satisfaction state
        sat_state = [False] * len(clauses)
        
        def is_satisfied_clause(i, assign):
            for l in clauses[i]:
                if assign[abs(l) - 1] == (l > 0):
                    return True
            return False
        
        for i in range(len(clauses)):
            sat_state[i] = is_satisfied_clause(i, assignment)
        
        current_unsat = sum(1 for s in sat_state if not s)
        
        if current_unsat == 0:
            return assignment
        
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        steps_without_improvement = 0
        local_best_unsat = current_unsat
        
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment[:]
                return best_assignment
            
            if current_unsat == 0:
                return assignment
            
            # Find an unsatisfied clause
            unsat_indices = [i for i, s in enumerate(sat_state) if not s]
            if not unsat_indices:
                return assignment
            
            clause_idx = rng.choice(unsat_indices)
            clause = clauses[clause_idx]
            
            # Heuristic: Min-Conflicts
            min_sat_count = float('inf')
            best_var = -1
            
            for l in clause:
                v = abs(l) - 1
                sat_count = 0
                for c_idx in var_to_clauses[v]:
                    if sat_state[c_idx]:
                        sat_count += 1
                
                if sat_count < min_sat_count:
                    min_sat_count = sat_count
                    best_var = v
                    if sat_count == 0:
                        break
            
            if best_var == -1:
                best_var = abs(rng.choice(clause)) - 1
            
            # Flip the variable
            assignment[best_var] = not assignment[best_var]
            
            # Update sat_state and current_unsat incrementally
            for c_idx in var_to_clauses[best_var]:
                old_sat = sat_state[c_idx]
                new_sat = is_satisfied_clause(c_idx, assignment)
                sat_state[c_idx] = new_sat
                
                if old_sat and not new_sat:
                    current_unsat += 1
                elif not old_sat and new_sat:
                    current_unsat -= 1
            
            if current_unsat < local_best_unsat:
                local_best_unsat = current_unsat
                steps_without_improvement = 0
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment[:]
            else:
                steps_without_improvement += 1
            
            if steps_without_improvement > restart_threshold:
                break
                
    return best_assignment
# EVOLVE-BLOCK-END
