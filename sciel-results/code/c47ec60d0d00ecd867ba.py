import numpy as np
import time
import random

def solve(instance):
    n = instance["n"]
    clauses_list = instance["clauses"]
    m = len(clauses_list)
    
    if m == 0:
        return [False] * n

    clause_lits = np.zeros((m, 3), dtype=np.int32)
    for i, c in enumerate(clauses_list):
        for j, l in enumerate(c):
            clause_lits[i, j] = l
            
    var_indices = np.abs(clause_lits) - 1
    signs = np.where(clause_lits > 0, 1, -1)

    var_occurrences = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = var_indices[i, j]
            var_occurrences[v].append((i, j))

    def compute_initial_state(assignment):
        lit_sat = np.zeros((m, 3), dtype=bool)
        sat_lit_count = np.zeros(m, dtype=np.int32)
        for i in range(m):
            for j in range(3):
                v = var_indices[i, j]
                s = signs[i, j]
                val = assignment[v]
                if s == 1:
                    ls = val
                else:
                    ls = not val
                lit_sat[i, j] = ls
                if ls:
                    sat_lit_count[i] += 1
        clause_sat = sat_lit_count > 0
        unsat_count = np.sum(~clause_sat)
        return lit_sat, clause_sat, unsat_count, sat_lit_count

    def calculate_gain(var_idx, lit_sat, sat_lit_count):
        gain = 0
        v = var_idx
        for ci, lp in var_occurrences[v]:
            lit_is_sat_now = lit_sat[ci, lp]
            count = sat_lit_count[ci]
            
            if lit_is_sat_now:
                # Flipping makes it unsat
                # Clause becomes unsat if count was 1
                if count == 1:
                    gain += 1
            else:
                # Flipping makes it sat
                # Clause becomes sat if count was 0
                if count == 0:
                    gain -= 1
        return gain

    rng = np.random.default_rng(42)
    start_time = time.time()
    deadline = start_time + 290.0

    best_assign = None
    best_unsat = float('inf')
    
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        assignment = rng.random(n) < 0.5
        lit_sat, clause_sat, unsat_count, sat_lit_count = compute_initial_state(assignment)
        
        if unsat_count == 0:
            return assignment.tolist()
            
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assign = assignment.copy()
            
        walk_prob = 0.2
        max_steps = 5000 + restart * 1000
        no_improve_steps = 0
        last_unsat = unsat_count
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            if unsat_count == 0:
                return assignment.tolist()
                
            unsat_indices = np.where(~clause_sat)[0]
            if len(unsat_indices) == 0:
                return assignment.tolist()
                
            c_idx = rng.choice(unsat_indices)
            
            if rng.random() < walk_prob:
                lit_pos = rng.integers(0, 3)
                var_idx = var_indices[c_idx, lit_pos]
            else:
                best_gain = float('inf')
                best_var = None
                for k in range(3):
                    v = var_indices[c_idx, k]
                    g = calculate_gain(v, lit_sat, sat_lit_count)
                    if g < best_gain:
                        best_gain = g
                        best_var = v
                var_idx = best_var
                
            assignment[var_idx] = not assignment[var_idx]
            
            for ci, lp in var_occurrences[var_idx]:
                old_lit_sat = lit_sat[ci, lp]
                new_lit_sat = not old_lit_sat
                lit_sat[ci, lp] = new_lit_sat
                
                old_count = sat_lit_count[ci]
                new_count = old_count
                if old_lit_sat:
                    new_count -= 1
                else:
                    new_count += 1
                sat_lit_count[ci] = new_count
                
                old_sat = old_count > 0
                new_sat = new_count > 0
                
                if old_sat and not new_sat:
                    unsat_count += 1
                elif not old_sat and new_sat:
                    unsat_count -= 1
                    
                clause_sat[ci] = new_sat
                
            if unsat_count < last_unsat:
                last_unsat = unsat_count
                no_improve_steps = 0
            else:
                no_improve_steps += 1
                if no_improve_steps > 500:
                    walk_prob = min(0.8, walk_prob * 1.1)
                    
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assignment.copy()
                
            if no_improve_steps > 2000:
                break

    return best_assign.tolist() if best_assign is not None else [False] * n
