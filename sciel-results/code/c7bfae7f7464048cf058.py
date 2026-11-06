import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Preprocessing
    clause_vars = []
    clause_sigs = []
    
    for c in clauses:
        vars_in_c = []
        sigs_in_c = []
        for lit in c:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            vars_in_c.append(v)
            sigs_in_c.append(s)
        clause_vars.append(vars_in_c)
        clause_sigs.append(sigs_in_c)
        
    m = len(clauses)
    
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for v in clause_vars[i]:
            var_to_clauses[v].append(i)
            
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 325.0
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # State
    assignment = [False] * n
    sat_counts = [0] * m
    
    # Dynamic list of unsatisfied clauses
    unsat_list = []
    clause_pos_in_unsat = {} # clause_idx -> index in unsat_list
    
    def add_unsat(c_idx):
        if c_idx in clause_pos_in_unsat:
            return
        clause_pos_in_unsat[c_idx] = len(unsat_list)
        unsat_list.append(c_idx)
        
    def remove_unsat(c_idx):
        if c_idx not in clause_pos_in_unsat:
            return
        pos = clause_pos_in_unsat[c_idx]
        last_idx = len(unsat_list) - 1
        if pos != last_idx:
            last_clause = unsat_list[last_idx]
            unsat_list[pos] = last_clause
            clause_pos_in_unsat[last_clause] = pos
        unsat_list.pop()
        del clause_pos_in_unsat[c_idx]
        
    def update_state(var_idx, new_val):
        old_val = assignment[var_idx]
        if old_val == new_val:
            return
        assignment[var_idx] = new_val
        
        for c_idx in var_to_clauses[var_idx]:
            sign = 0
            for j, v in enumerate(clause_vars[c_idx]):
                if v == var_idx:
                    sign = clause_sigs[c_idx][j]
                    break
            
            old_sat = (old_val == (sign == 1))
            new_sat = (new_val == (sign == 1))
            
            delta = (1 if new_sat else 0) - (1 if old_sat else 0)
            sat_counts[c_idx] += delta
            
            if delta > 0:
                remove_unsat(c_idx)
            elif delta < 0:
                add_unsat(c_idx)
                
    def init_state(assign):
        nonlocal unsat_list, clause_pos_in_unsat, sat_counts
        unsat_list = []
        clause_pos_in_unsat = {}
        sat_counts = [0] * m
        for i in range(n):
            assignment[i] = assign[i]
            
        for c_idx in range(m):
            count = 0
            for j, v in enumerate(clause_vars[c_idx]):
                sign = clause_sigs[c_idx][j]
                if assignment[v] == (sign == 1):
                    count += 1
            sat_counts[c_idx] = count
            if count == 0:
                add_unsat(c_idx)
                
    def get_best_flip(clause_idx):
        best_var = -1
        best_score = -1
        candidates = []
        
        for v in clause_vars[clause_idx]:
            score = 0
            for c_prime in var_to_clauses[v]:
                if c_prime in clause_pos_in_unsat:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_var = v
                candidates = [v]
            elif score == best_score:
                candidates.append(v)
                
        if not candidates:
            return rng.choice(clause_vars[clause_idx])
            
        return rng.choice(candidates)
        
    def walksat_run(max_steps, initial_assign=None):
        if initial_assign is None:
            assign = [rng.random() < 0.5 for _ in range(n)]
        else:
            assign = list(initial_assign)
            
        init_state(assign)
        
        if not unsat_list:
            return list(assignment), 0
            
        for step in range(max_steps):
            if time.time() > deadline:
                return list(assignment), len(unsat_list)
                
            if not unsat_list:
                return list(assignment), 0
                
            # Pick random unsat clause in O(1)
            idx = rng.randint(0, len(unsat_list) - 1)
            c_idx = unsat_list[idx]
            
            v = get_best_flip(c_idx)
            new_val = not assignment[v]
            update_state(v, new_val)
            
        return list(assignment), len(unsat_list)
        
    # Phase 1: Random
    for _ in range(10):
        if time.time() > deadline:
            break
        assign, count = walksat_run(500)
        if count < best_unsat_count:
            best_unsat_count = count
            best_assignment = assign
        if best_unsat_count == 0:
            return best_assignment
            
    # Phase 2: Guided
    while time.time() < deadline:
        if best_assignment and rng.random() < 0.5:
            init = list(best_assignment)
            for _ in range(rng.randint(1, 5)):
                v = rng.randint(0, n-1)
                init[v] = not init[v]
        else:
            init = None
            
        assign, count = walksat_run(1000, init)
        if count < best_unsat_count:
            best_unsat_count = count
            best_assignment = assign
        if best_unsat_count == 0:
            return best_assignment
            
    return best_assignment if best_assignment else [False]*n
