import time
import random
import sys
import math
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 325.0  # Safety margin

    if m == 0:
        return [True] * n

    # Preprocessing: Convert to 0-based, build structures
    # clause_data: list of lists of (var, sign)
    # var_occ: list of lists of (clause_idx, sign)
    clause_data = []
    var_occ = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses):
        cd = []
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            cd.append((v, sign))
            var_occ[v].append((i, sign))
        clause_data.append(cd)

    # Unit Propagation and Pure Literal Elimination
    assignment = [None] * n
    clause_active = [True] * m
    sat_counts = [0] * m
    
    def propagate():
        changed = True
        while changed:
            changed = False
            # Pure Literal Elimination
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
            
            # Unit Propagation
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
        
        # Update sat_counts for active clauses
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
    
    # Check if solved
    if all(not clause_active[ci] or sat_counts[ci] > 0 for ci in range(m)):
        return [a if a is not None else True for a in assignment]

    # Initialize unassigned variables
    rng = random.Random(42)
    unassigned_vars = [v for v in range(n) if assignment[v] is None]
    for v in unassigned_vars:
        assignment[v] = rng.random() < 0.5

    # Recompute state
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
        clause_active[ci] = (sc == 0)

    unsat_clauses = [ci for ci in range(m) if clause_active[ci]]
    
    if not unsat_clauses:
        return assignment[:]

    best_unsat = len(unsat_clauses)
    best_assignment = assignment[:]

    # Simulated Annealing Parameters
    # We use a cooling schedule and restarts
    max_restarts = 100
    max_steps_per_restart = 10000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize or Perturb
        if restart == 0:
            current_assignment = best_assignment[:]
        else:
            # Perturb the best assignment found so far
            current_assignment = best_assignment[:]
            # Flip a random subset of variables
            num_flips = max(1, n // 10)
            for _ in range(num_flips):
                v = rng.randint(0, n - 1)
                current_assignment[v] = not current_assignment[v]
        
        # Recompute state for current_assignment
        for ci in range(m):
            sc = 0
            for (v, sign) in clause_data[ci]:
                if sign == 1:
                    if current_assignment[v]:
                        sc += 1
                else:
                    if not current_assignment[v]:
                        sc += 1
            sat_counts[ci] = sc
            clause_active[ci] = (sc == 0)
        
        current_unsat = [ci for ci in range(m) if clause_active[ci]]
        current_cost = len(current_unsat)
        
        if current_cost == 0:
            return current_assignment
        
        if current_cost < best_unsat:
            best_unsat = current_cost
            best_assignment = current_assignment[:]

        # SA Parameters
        T_start = 1.0
        T_end = 0.01
        cooling_rate = (T_end / T_start) ** (1.0 / max_steps_per_restart)
        T = T_start
        
        # Precompute variable occurrence in unsat clauses for faster heuristic
        # var_unsat_count[v] = number of unsat clauses containing v
        
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break
            
            if not current_unsat:
                return current_assignment
            
            # Update best
            if current_cost < best_unsat:
                best_unsat = current_cost
                best_assignment = current_assignment[:]
            
            # Select a variable to flip
            # Strategy: With 50% probability, pick a variable from a random unsat clause
            # that appears in the most unsat clauses (Greedy).
            # Otherwise, pick a random variable from a random unsat clause.
            
            if rng.random() < 0.5 and len(current_unsat) > 0:
                # Pick a random unsat clause
                c_idx = current_unsat[rng.randint(0, len(current_unsat) - 1)]
                cd_c = clause_data[c_idx]
                
                # Find variable in this clause with highest "unsat pressure"
                best_v = cd_c[0][0]
                best_pressure = -1
                for (v, sign) in cd_c:
                    # Pressure is the number of unsat clauses this variable is in
                    # We can approximate this by counting how many times v appears in current_unsat
                    # To avoid O(m) scan per variable, we can maintain a counter, 
                    # but for 150 vars and 3-SAT, a simple count is fast enough.
                    pressure = 0
                    for ci in current_unsat:
                        # Check if v is in clause ci
                        # This is O(m) per variable, which is O(m^2) total. 
                        # For m~1000, this is 1e6 ops, which is fine in Python if optimized.
                        # Let's use a faster check: pre-compute var_in_unsat?
                        # Actually, we can just iterate through the clause's variables.
                        pass 
                    
                    # Faster approach: 
                    # Count occurrences of v in the list of unsat clauses' variables.
                    # We can maintain a dictionary or array `var_unsat_count`.
                    # Let's maintain `var_unsat_count` dynamically.
                    # For now, let's just pick randomly from the clause to keep it simple 
                    # but use a "weighted" random pick based on global frequency.
                    
                    # Let's stick to a simpler heuristic: 
                    # Pick the variable that appears in the most unsat clauses.
                    # We will compute this on the fly for the 3 variables in the clause.
                    count_v = 0
                    for ci in current_unsat:
                        if v in [vv for (vv, ss) in clause_data[ci]]:
                            count_v += 1
                    if count_v > best_pressure:
                        best_pressure = count_v
                        best_v = v
                
                v_to_flip = best_v
            else:
                # Random variable from random unsat clause
                c_idx = current_unsat[rng.randint(0, len(current_unsat) - 1)]
                cd_c = clause_data[c_idx]
                v_to_flip = cd_c[rng.randint(0, len(cd_c) - 1)][0]
            
            # Calculate delta cost
            # Flip v_to_flip
            old_val = current_assignment[v_to_flip]
            new_val = not old_val
            
            delta = 0
            for (ci, sign) in var_occ[v_to_flip]:
                if sign == 1:
                    # Literal is v
                    was_true = old_val
                    is_true = new_val
                else:
                    # Literal is -v
                    was_true = not old_val
                    is_true = not new_val
                
                if was_true and not is_true:
                    # Clause becomes unsatisfied
                    if sat_counts[ci] == 1:
                        delta += 1
                elif not was_true and is_true:
                    # Clause becomes satisfied
                    if sat_counts[ci] == 0:
                        delta -= 1
            
            # Acceptance Probability
            if delta <= 0:
                accept = True
            else:
                # Metropolis Criterion
                if T > 0:
                    p = math.exp(-delta / T)
                    accept = (rng.random() < p)
                else:
                    accept = False
            
            if accept:
                # Apply flip
                current_assignment[v_to_flip] = new_val
                current_cost += delta
                
                # Update sat_counts and clause_active
                for (ci, sign) in var_occ[v_to_flip]:
                    if sign == 1:
                        is_true = current_assignment[v_to_flip]
                    else:
                        is_true = not current_assignment[v_to_flip]
                    
                    if is_true:
                        sat_counts[ci] += 1
                        if sat_counts[ci] == 1:
                            # Clause became satisfied
                            if clause_active[ci]:
                                clause_active[ci] = False
                                try:
                                    idx = current_unsat.index(ci)
                                    current_unsat.pop(idx)
                                except ValueError:
                                    pass
                    else:
                        sat_counts[ci] -= 1
                        if sat_counts[ci] == 0:
                            # Clause became unsatisfied
                            if not clause_active[ci]:
                                clause_active[ci] = True
                                current_unsat.append(ci)
            else:
                # Revert flip (no change to assignment, cost, or counts)
                pass
            
            # Cool down
            T *= cooling_rate

    if best_assignment is not None:
        return best_assignment
    
    return [True] * n
