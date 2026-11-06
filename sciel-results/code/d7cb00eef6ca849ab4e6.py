"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n

    # Deadline guard: 320s safety margin out of 330s limit
    start_time = time.time()
    deadline = start_time + 320.0
    
    # Preprocess clauses into numpy arrays for vectorization
    # Convert 1-based signed literals to 0-based indices and signs
    indices = np.empty((m, 3), dtype=np.int32)
    signs = np.empty((m, 3), dtype=np.int8)
    
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            idx = abs(lit) - 1
            indices[i, j] = idx
            signs[i, j] = 1 if lit > 0 else -1
            
    # Variable occurrence lists for fast gain calculation
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = indices[i, j]
            var_to_clauses[v].append(i)
    
    # Precompute for each variable, the list of (clause_idx, position) pairs
    var_occurrences = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = indices[i, j]
            var_occurrences[v].append((i, j))
    
    # Helper to check if assignment satisfies clause
    def is_satisfied(ci, assignment):
        for j in range(3):
            if (assignment[indices[ci, j]] and signs[ci, j] > 0) or \
               (not assignment[indices[ci, j]] and signs[ci, j] < 0):
                return True
        return False
    
    def count_satisfied(ci, assignment):
        cnt = 0
        for j in range(3):
            if (assignment[indices[ci, j]] and signs[ci, j] > 0) or \
               (not assignment[indices[ci, j]] and signs[ci, j] < 0):
                cnt += 1
        return cnt
    
    def initialize_assignment():
        return [random.random() < 0.5 for _ in range(n)]
    
    def calculate_unsat(assignment):
        unsat = []
        for i in range(m):
            if not is_satisfied(i, assignment):
                unsat.append(i)
        return unsat
    
    def flip_variable(var, assignment):
        assignment[var] = not assignment[var]
    
    # Main Loop
    best_assignment = None
    best_unsat_count = m + 1
    
    # Precompute clause sizes (all 3) and variable frequencies
    var_freq = [len(var_to_clauses[v]) for v in range(n)]
    
    # Luby sequence for restarts with larger base
    def luby_sequence(base=100):
        k = 1
        while True:
            n = base
            while n < k * base:
                yield n
                n *= 2
            k += 1
    
    luby_gen = luby_sequence(base=100)
    
    while time.time() < deadline:
        # Get next restart length from Luby sequence
        try:
            flip_budget = next(luby_gen)
        except StopIteration:
            flip_budget = 1000000
        
        # Cap budget to avoid single run taking too long
        max_budget = 50000
        flip_budget = min(flip_budget, max_budget)
        
        # Restart logic
        assignment = initialize_assignment()
        unsat = calculate_unsat(assignment)
        current_unsat = len(unsat)
        
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
            
        if current_unsat == 0:
            return assignment
        
        steps = 0
        no_improve_steps = 0
        local_best_unsat = current_unsat
        
        # Adaptive Noise
        noise = 0.3
        
        while steps < flip_budget and time.time() < deadline:
            if len(unsat) == 0:
                return assignment
            
            # Pick a random unsatisfied clause
            clause_idx = unsat[random.randint(0, len(unsat) - 1)]
            
            # Pick a literal to flip
            if random.random() < noise:
                # Random flip from clause
                pos = random.randint(0, 2)
                var = indices[clause_idx, pos]
            else:
                # Find var in clause that maximizes gain
                best_gain = -100
                best_var = -1
                
                for pos in range(3):
                    var = indices[clause_idx, pos]
                    
                    # Calculate gain for flipping var
                    gain = 0
                    for (ci, pos_in_ci) in var_occurrences[var]:
                        desired = signs[ci, pos_in_ci] > 0
                        old_sat = assignment[var] == desired
                        new_val = not assignment[var]
                        new_sat = new_val == desired
                        
                        if old_sat and not new_sat:
                            if count_satisfied(ci, assignment) == 1:
                                gain -= 1
                        elif not old_sat and new_sat:
                            if not is_satisfied(ci, assignment):
                                gain += 1
                                
                    if gain > best_gain:
                        best_gain = gain
                        best_var = var
                            
                var = best_var
                if var == -1:
                    var = indices[clause_idx, 0]
                    
            # Perform flip
            flip_variable(var, assignment)
            
            # Update unsat list incrementally
            affected = var_to_clauses[var]
            affected_set = set(affected)
            
            new_unsat = []
            for ci in unsat:
                if ci in affected_set:
                    if not is_satisfied(ci, assignment):
                        new_unsat.append(ci)
                else:
                    new_unsat.append(ci)
            
            for ci in affected:
                if ci not in affected_set or ci in affected_set:
                    if not is_satisfied(ci, assignment):
                        if ci not in new_unsat:
                            new_unsat.append(ci)
            
            unsat = new_unsat
            current_unsat = len(unsat)
            
            if current_unsat < local_best_unsat:
                local_best_unsat = current_unsat
                no_improve_steps = 0
            else:
                no_improve_steps += 1
            
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
                
            steps += 1
            
            # Adapt noise
            if steps % 50 == 0:
                if no_improve_steps > 50:
                    noise = min(0.7, noise + 0.05)
                else:
                    noise = max(0.1, noise - 0.02)
            
            # Restart if stuck
            if no_improve_steps > 1000:
                break
    
    if best_assignment is not None:
        return best_assignment
    return assignment

# EVOLVE-BLOCK-END
