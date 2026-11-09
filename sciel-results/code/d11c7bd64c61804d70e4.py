import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Precompute variable to clauses mapping
    var_to_clauses = [[] for _ in range(n)]
    for c_idx, clause in enumerate(clauses):
        for lit in clause:
            v_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_to_clauses[v_idx].append((c_idx, sign))
            
    phase = [-1] * n
    assignment = [False] * n
    sat_count = [0] * len(clauses)
    unsat_list = []
    pos_in_unsat = [-1] * len(clauses) # -1 if not in list
    
    def get_val(lit):
        idx = abs(lit) - 1
        return assignment[idx] == (lit > 0)
        
    def add_unsat(c_i):
        if pos_in_unsat[c_i] == -1:
            pos_in_unsat[c_i] = len(unsat_list)
            unsat_list.append(c_i)
            
    def remove_unsat(c_i):
        pos = pos_in_unsat[c_i]
        if pos != -1:
            last_idx = len(unsat_list) - 1
            last_c = unsat_list[last_idx]
            unsat_list[pos] = last_c
            pos_in_unsat[last_c] = pos
            unsat_list.pop()
            pos_in_unsat[c_i] = -1

    def init_assignment():
        nonlocal assignment, sat_count, unsat_list, pos_in_unsat
        assignment = [False] * n
        for i in range(n):
            if phase[i] == 1:
                assignment[i] = True
            elif phase[i] == 0:
                assignment[i] = False
            else:
                assignment[i] = random.random() < 0.5
                
        sat_count = [0] * len(clauses)
        unsat_list = []
        pos_in_unsat = [-1] * len(clauses)
        for i, clause in enumerate(clauses):
            cnt = 0
            for lit in clause:
                if get_val(lit):
                    cnt += 1
            sat_count[i] = cnt
            if cnt == 0:
                add_unsat(i)

    start_time = time.time()
    time_limit = 290.0
    
    max_restarts = 10000
    step_budget = 50000
    p_flip_random = 0.3
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        init_assignment()
        
        if not unsat_list:
            return assignment
            
        for step in range(step_budget):
            if time.time() - start_time > time_limit:
                break
                
            if not unsat_list:
                return assignment
                
            # Pick a random unsatisfied clause
            # Random selection from unsat_list
            c_idx = random.choice(unsat_list)
            clause = clauses[c_idx]
            
            # Choose variable to flip
            if random.random() < p_flip_random:
                # Random flip
                lit = random.choice(clause)
                v_idx = abs(lit) - 1
            else:
                # Min-conflicts / G-SAT
                best_gain = -100
                best_lit = -1
                for lit in clause:
                    v_idx = abs(lit) - 1
                    # Calculate gain
                    gain = 1 # This clause becomes satisfied
                    
                    for (other_c_idx, other_sign) in var_to_clauses[v_idx]:
                        if other_c_idx == c_idx:
                            continue
                        
                        current_lit_val = assignment[v_idx] == (other_sign > 0)
                        if current_lit_val:
                            # Literal is currently True
                            # Flipping makes it False
                            # If this was the only True literal in the clause, it becomes unsat
                            if sat_count[other_c_idx] == 1:
                                gain -= 1
                        else:
                            # Literal is currently False
                            # Flipping makes it True
                            # If this clause was unsat, it becomes sat
                            if sat_count[other_c_idx] == 0:
                                gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_lit = lit
                        
                v_idx = abs(best_lit) - 1
                
            # Perform flip
            old_val = assignment[v_idx]
            new_val = not old_val
            assignment[v_idx] = new_val
            
            # Update sat_count and unsat_list
            for (c_i, sign) in var_to_clauses[v_idx]:
                old_lit_val = old_val == (sign > 0)
                if old_lit_val:
                    # Was True, now False
                    old_sc = sat_count[c_i]
                    sat_count[c_i] = old_sc - 1
                    if old_sc == 1:
                        # Became unsat
                        add_unsat(c_i)
                else:
                    # Was False, now True
                    old_sc = sat_count[c_i]
                    sat_count[c_i] = old_sc + 1
                    if old_sc == 0:
                        # Became sat, remove from unsat_list
                        remove_unsat(c_i)

            # Update phase
            # If the targeted clause is now satisfied, save the phase
            if sat_count[c_idx] > 0:
                phase[v_idx] = 1 if new_val else 0

    return assignment
