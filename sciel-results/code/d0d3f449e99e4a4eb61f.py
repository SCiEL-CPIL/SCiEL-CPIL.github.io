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

    # Convert to 0-based signed literals
    # lit > 0 means variable (lit-1) is True
    # lit < 0 means variable (lit-1) is False
    clause_data = []
    for clause in clauses:
        cd = []
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            cd.append((v, sign))
        clause_data.append(cd)
    
    # Variable occurrence: var_occ[v] = list of (clause_idx, sign)
    var_occ = [[] for _ in range(n)]
    for i, cd in enumerate(clause_data):
        for (v, sign) in cd:
            var_occ[v].append((i, sign))
    
    # Unit propagation and pure literal elimination
    assignment = [None] * n
    clause_active = [True] * m
    sat_counts = [0] * m
    
    def propagate():
        changed = True
        while changed:
            changed = False
            
            # Pure literal elimination
            for v in range(n):
                if assignment[v] is not None:
                    continue
                has_pos = False
                has_neg = False
                for (ci, sign) in var_occ[v]:
                    if not clause_active[ci]:
                        continue
                    if sign == 1:
                        has_pos = True
                    else:
                        has_neg = True
                if has_pos and not has_neg:
                    assignment[v] = True
                    changed = True
                elif has_neg and not has_pos:
                    assignment[v] = False
                    changed = True
            
            # Unit propagation
            for ci in range(m):
                if not clause_active[ci]:
                    continue
                unassigned = []
                true_count = 0
                for (v, sign) in clause_data[ci]:
                    if assignment[v] is None:
                        unassigned.append((v, sign))
                    else:
                        if sign == 1:
                            if assignment[v]:
                                true_count += 1
                        else:
                            if not assignment[v]:
                                true_count += 1
                if true_count > 0:
                    clause_active[ci] = False
                    sat_counts[ci] = true_count
                    changed = True
                elif len(unassigned) == 1:
                    v, sign = unassigned[0]
                    assignment[v] = (sign == 1)
                    changed = True
        
        # Update sat_counts for all active clauses
        for ci in range(m):
            if clause_active[ci]:
                sc = 0
                for (v, sign) in clause_data[ci]:
                    if assignment[v] is None:
                        continue
                    if sign == 1:
                        if assignment[v]:
                            sc += 1
                    else:
                        if not assignment[v]:
                            sc += 1
                sat_counts[ci] = sc
    
    propagate()
    
    # Check if already solved
    if all(not clause_active[ci] or sat_counts[ci] > 0 for ci in range(m)):
        return [a if a is not None else True for a in assignment]
    
    # Get unassigned variables
    unassigned_vars = [v for v in range(n) if assignment[v] is None]
    if not unassigned_vars:
        if all(not clause_active[ci] or sat_counts[ci] > 0 for ci in range(m)):
            return assignment[:]
    
    # Prepare for WalkSAT with improved heuristics
    rng = random.Random(42)
    
    # Initialize: use propagated values where available, randomize the rest
    for v in unassigned_vars:
        assignment[v] = rng.random() < 0.5
    
    # Recompute sat_counts
    for ci in range(m):
        if not clause_active[ci]:
            continue
        sc = 0
        for (v, sign) in clause_data[ci]:
            if sign == 1:
                if assignment[v]:
                    sc += 1
            else:
                if not assignment[v]:
                    sc += 1
        sat_counts[ci] = sc
        if sc > 0:
            clause_active[ci] = False
    
    unsat_clauses = [ci for ci in range(m) if clause_active[ci] and sat_counts[ci] == 0]
    
    if not unsat_clauses:
        return assignment[:]
    
    best_unsat = len(unsat_clauses)
    best_assignment = assignment[:]
    
    # Adaptive WalkSAT with multiple strategies
    max_restarts = 200
    max_steps_per_restart = 3000
    
    # Track history for adaptive parameter tuning
    recent_unsat_counts = []
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Adaptive restart strategy
        if restart > 0 and restart % 5 == 0:
            # Perturb best assignment with random flips
            assignment = best_assignment[:]
            num_flips = max(1, min(n // 10, 15))
            for _ in range(num_flips):
                v = rng.randint(0, n - 1)
                assignment[v] = not assignment[v]
        else:
            # Random initialization
            assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Recompute sat_counts and unsat clauses
        for ci in range(m):
            sc = 0
            for (v, sign) in clause_data[ci]:
                if sign == 1:
                    if assignment[v]:
                        sc += 1
                else:
                    if not assignment[v]:
                        sc += 1
            sat_counts[ci] = sc
        clause_active = [sat_counts[ci] == 0 for ci in range(m)]
        unsat_clauses = [ci for ci in range(m) if clause_active[ci]]
        
        if not unsat_clauses:
            return assignment
        
        if len(unsat_clauses) < best_unsat:
            best_unsat = len(unsat_clauses)
            best_assignment = assignment[:]
        
        # Adaptive p_greedy based on progress
        p_greedy = 0.6 + 0.2 * (1 - len(unsat_clauses) / max(m, 1))
        
        stuck_count = 0
        last_unsat_count = len(unsat_clauses)
        no_improve_steps = 0
        min_unsat_in_restart = len(unsat_clauses)
        
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break
            
            if not unsat_clauses:
                return assignment
            
            if len(unsat_clauses) < best_unsat:
                best_unsat = len(unsat_clauses)
                best_assignment = assignment[:]
            
            if len(unsat_clauses) < min_unsat_in_restart:
                min_unsat_in_restart = len(unsat_clauses)
                no_improve_steps = 0
            else:
                no_improve_steps += 1
            
            # Detect stuck
            if len(unsat_clauses) >= last_unsat_count:
                stuck_count += 1
            else:
                stuck_count = 0
                last_unsat_count = len(unsat_clauses)
            
            # If stuck too long, break early to restart
            if stuck_count > 200:
                break
            
            # Pick random unsat clause
            c_idx = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
            cd_c = clause_data[c_idx]
            
            # Calculate gains for each variable in the clause
            gains = []
            for (v, sign) in cd_c:
                gain = 0
                for (ci2, sign2) in var_occ[v]:
                    if sign2 == 1:
                        is_true = assignment[v]
                    else:
                        is_true = not assignment[v]
                    
                    if is_true:
                        # Flipping makes it false
                        if sat_counts[ci2] == 1:
                            gain -= 1
                    else:
                        # Flipping makes it true
                        if sat_counts[ci2] == 0:
                            gain += 1
                gains.append(gain)
            
            best_gain = max(gains)
            
            # Choose variable with weighted probability
            if rng.random() < p_greedy:
                # Greedy: pick from variables with best gain
                best_vars = [v for (v, g) in zip([v for (v, s) in cd_c], gains) if g == best_gain]
                v_to_flip = best_vars[rng.randint(0, len(best_vars) - 1)]
            else:
                # Random: pick any variable in the clause
                v_to_flip = cd_c[rng.randint(0, len(cd_c) - 1)][0]
            
            # Flip
            old_val = assignment[v_to_flip]
            assignment[v_to_flip] = not old_val
            
            # Update sat_counts
            for (ci2, sign2) in var_occ[v_to_flip]:
                if sign2 == 1:
                    is_true_now = assignment[v_to_flip]
                else:
                    is_true_now = not assignment[v_to_flip]
                
                if is_true_now:
                    # Was false, now true
                    sat_counts[ci2] += 1
                    if sat_counts[ci2] == 1 and clause_active[ci2]:
                        clause_active[ci2] = False
                        try:
                            idx = unsat_clauses.index(ci2)
                            unsat_clauses.pop(idx)
                        except ValueError:
                            pass
                else:
                    # Was true, now false
                    sat_counts[ci2] -= 1
                    if sat_counts[ci2] == 0:
                        clause_active[ci2] = True
                        unsat_clauses.append(ci2)
        
        # End of restart
    
    if best_assignment is not None:
        return best_assignment
    
    return [True] * n
