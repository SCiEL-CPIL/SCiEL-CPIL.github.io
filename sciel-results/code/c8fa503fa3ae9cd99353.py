import time
import random
import numpy as np
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 285  # Safety margin
    
    # Preprocessing
    # clauses are lists of signed 1-based literals
    # Convert to (var_idx, polarity)
    # var_idx 0-based, polarity True/False
    
    # Data structures
    clause_data = [] # list of lists of (var_idx, polarity)
    var_clauses = [[] for _ in range(n)] # list of lists of (clause_idx, polarity)
    
    for i, c in enumerate(clauses):
        lit_data = []
        for l in c:
            v = abs(l) - 1
            p = l > 0
            lit_data.append((v, p))
            var_clauses[v].append((i, p))
        clause_data.append(lit_data)
        
    m = len(clauses)
    
    best_unsat_count = float('inf')
    best_assignment = None
    
    # Parameters tuned for 150 vars, ~450-600 clauses
    # Increased steps per restart, reduced restarts to allow deeper search
    max_restarts = 5000 
    steps_per_restart = 150
    
    # Smooth surrogate: log-sum-exp penalty weight for tie-breaking
    # A smaller "slack" in unsat clauses gets a smoother gradient signal
    LSE_BETA = 5.0  # temperature for soft-min over unsat clause slacks
    
    # Track phase statistics for phase-biased initialization
    phase_count = [0] * n  # count how many times each var appears positive/negative
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        rng = random.Random(restart * 7 + 1)
        
        # Phase-biased initialization: use phase statistics to bias initial assignment
        # This helps escape symmetric local minima in planted SAT instances
        assignment = []
        for v in range(n):
            if phase_count[v] > 0:
                # Bias towards the phase that appears more often in unsat clauses
                assignment.append(rng.random() < (phase_count[v] / 100.0))
            else:
                assignment.append(rng.random() < 0.5)
        
        # Calc initial sat counts using NumPy for speed
        # Build a matrix: for each clause, for each variable, store polarity (1 if positive, -1 if negative, 0 if not present)
        # Then sat_count = sum over vars of (assignment[v] == polarity) for each clause
        
        # Actually, let's keep the Python loop but optimize it slightly
        sat_count = [0] * m
        is_unsat = [False] * m
        unsat_list = []
        unsat_count = 0
        
        for c_idx in range(m):
            cnt = 0
            for v, p in clause_data[c_idx]:
                if assignment[v] == p:
                    cnt += 1
            sat_count[c_idx] = cnt
            if cnt == 0:
                is_unsat[c_idx] = True
                unsat_list.append(c_idx)
                unsat_count += 1
                
        if unsat_count == 0:
            return assignment
            
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
            
        # Walk
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            if unsat_count == 0:
                return assignment
                
            # Pick random unsat clause efficiently
            # Clean up stale entries periodically to keep list small
            if len(unsat_list) > unsat_count * 3 + 10:
                # Rebuild unsat_list from scratch (rare, amortized cost)
                unsat_list = [i for i in range(m) if is_unsat[i]]
            
            c_idx = None
            # Pick a random valid unsat clause
            for _ in range(min(20, len(unsat_list) + 1)):
                if not unsat_list:
                    break
                idx = unsat_list[rng.randrange(len(unsat_list))]
                if is_unsat[idx]:
                    c_idx = idx
                    break
            
            if c_idx is None:
                # Fallback: scan for an unsat clause
                for idx in range(m):
                    if is_unsat[idx]:
                        c_idx = idx
                        break
                if c_idx is None:
                    return assignment # Should not happen if unsat_count > 0

            # Evaluate flips for literals in c_idx
            # Heuristic: Min-Conflicts (minimize number of newly unsat clauses)
            # Score = (clauses that become unsat) - (clauses that become sat)
            # Actually, standard WalkSAT uses:
            # 1. Calculate "gain" for each var in clause:
            #    gain = (# clauses that become unsat) - (# clauses that become sat)
            #    We want to minimize gain.
            #    Note: Flipping a var in an unsat clause:
            #    - If var was False and clause needs True, flipping to True satisfies this clause (gain -1 for this clause? No, this clause is currently unsat, so it becomes sat. That's a "fix").
            #    - Other clauses containing this var:
            #       - If they were sat and this was the only sat literal, they become unsat (break).
            #       - If they were unsat, they might become sat (fix).
            
            best_var = -1
            best_score = float('inf')
            
            for v, p in clause_data[c_idx]:
                broken = 0
                fixed = 0
                
                # Iterate over clauses containing v
                for c2_idx, p2 in var_clauses[v]:
                    if c2_idx == c_idx:
                        continue # Skip the clause we are trying to fix? 
                        # Actually, flipping v in c_idx:
                        # If v satisfies c_idx, c_idx becomes sat (fixed +1)
                        # If v does not satisfy c_idx, c_idx remains unsat (no change)
                        # But we know c_idx is unsat.
                        # If we flip v, does it become sat?
                        # If p == assignment[v] (current value matches polarity), then v is already satisfying c_idx?
                        # No, c_idx is unsat, so NO literal in c_idx satisfies it.
                        # So for all (v, p) in c_idx, assignment[v] != p.
                        # If we flip v, assignment[v] becomes not assignment[v].
                        # Since assignment[v] != p, not assignment[v] == p.
                        # So v WILL satisfy c_idx.
                        # Thus, c_idx becomes sat. fixed += 1.
                        
                        # Wait, let's verify.
                        # c_idx is unsat => for all l in c_idx, assignment[l.var] != l.pol.
                        # We pick v from c_idx.
                        # Current assignment[v] != p.
                        # Flip v => new assignment[v] = not assignment[v].
                        # Since old != p, new == p.
                        # So v satisfies c_idx.
                        # c_idx becomes sat.
                        # So "fixed" for c_idx is 1.
                        
                        # However, in the loop below, we iterate var_clauses[v].
                        # c_idx is in var_clauses[v].
                        # sat_count[c_idx] is 0.
                        # was_sat = (assignment[v] == p2). Here p2 is p.
                        # assignment[v] != p, so was_sat is False.
                        # So it goes to "else" branch (Gained satisfaction).
                        # sat_count[c_idx] becomes 1.
                        # is_unsat[c_idx] becomes False.
                        # So it is counted as "fixed" if we check sat_count[c2_idx] == 0.
                        # So we don't need special handling for c_idx if we use the general logic.
                        
                    # Check if v is currently satisfied in c2
                    was_sat = (assignment[v] == p2)
                    
                    if was_sat:
                        # Becomes unsat if this was the only satisfied literal
                        if sat_count[c2_idx] == 1:
                            broken += 1
                    else:
                        # Becomes sat
                        # If it was 0, it becomes 1 (fixed)
                        if sat_count[c2_idx] == 0:
                            fixed += 1
                            
                score = broken - fixed
                if score < best_score:
                    best_score = score
                    best_var = v
                    
            # Noise
            # Standard WalkSAT: with probability p, pick random var from clause
            # Otherwise pick best var
            # Adaptive noise: reduce noise as we get closer to solution
            if unsat_count <= 3:
                noise_prob = 0.05  # Very low noise when close to solution
            elif unsat_count <= 8:
                noise_prob = 0.15
            elif unsat_count <= 20:
                noise_prob = 0.25
            else:
                noise_prob = 0.35
                
            if rng.random() < noise_prob:
                v_to_flip = clause_data[c_idx][rng.randrange(3)][0]
            else:
                v_to_flip = best_var
                
            if v_to_flip == -1:
                # Should not happen
                continue

            # Apply flip
            old_val = assignment[v_to_flip]
            new_val = not old_val
            assignment[v_to_flip] = new_val
            
            # Update state
            for c2_idx, p2 in var_clauses[v_to_flip]:
                was_sat = (old_val == p2)
                
                if was_sat:
                    # Lost a satisfaction
                    sat_count[c2_idx] -= 1
                    if sat_count[c2_idx] == 0:
                        # Became unsat
                        if not is_unsat[c2_idx]:
                            is_unsat[c2_idx] = True
                            unsat_list.append(c2_idx)
                            unsat_count += 1
                            # Update phase count for phase-biased init
                            phase_count[v_to_flip] += 1
                else:
                    # Gained a satisfaction
                    sat_count[c2_idx] += 1
                    if sat_count[c2_idx] == 1:
                        # Became sat
                        if is_unsat[c2_idx]:
                            is_unsat[c2_idx] = False
                            unsat_count -= 1
                            # Do not remove from unsat_list to avoid O(N) shift
                            
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]

    # Fallback: Final high-precision polish pass
    # Run a tighter local search on the best assignment found
    if best_assignment is not None:
        polish_assignment = best_assignment[:]
        polish_deadline = deadline - 5  # 5s safety margin
        
        # Recompute state for polish
        p_sat_count = [0] * m
        p_is_unsat = [False] * m
        p_unsat_list = []
        p_unsat_count = 0
        for c_idx in range(m):
            cnt = 0
            for v, p in clause_data[c_idx]:
                if polish_assignment[v] == p:
                    cnt += 1
            p_sat_count[c_idx] = cnt
            if cnt == 0:
                p_is_unsat[c_idx] = True
                p_unsat_list.append(c_idx)
                p_unsat_count += 1
        
        # Tighter local search: more steps, less noise, focus on minimizing unsat
        polish_steps = 1000
        for step in range(polish_steps):
            if time.time() > polish_deadline:
                break
            if p_unsat_count == 0:
                return polish_assignment
            
            # Pick an unsat clause from the maintained list
            if not p_unsat_list:
                return polish_assignment
            
            # Clean stale entries if list grows too large
            if len(p_unsat_list) > p_unsat_count * 3 + 5:
                p_unsat_list = [i for i in range(m) if p_is_unsat[i]]
            
            c_idx = None
            for _ in range(min(10, len(p_unsat_list) + 1)):
                if not p_unsat_list:
                    break
                idx = p_unsat_list[rng.randrange(len(p_unsat_list))]
                if p_is_unsat[idx]:
                    c_idx = idx
                    break
            
            if c_idx is None:
                # Fallback: linear scan
                for idx in range(m):
                    if p_is_unsat[idx]:
                        c_idx = idx
                        break
                if c_idx is None:
                    return polish_assignment
            
            # Find best flip (no noise, pure greedy)
            best_v = -1
            best_s = float('inf')
            for v, p in clause_data[c_idx]:
                broken = 0
                fixed = 0
                for c2_idx, p2 in var_clauses[v]:
                    was_sat = (polish_assignment[v] == p2)
                    if was_sat:
                        if p_sat_count[c2_idx] == 1:
                            broken += 1
                    else:
                        if p_sat_count[c2_idx] == 0:
                            fixed += 1
                s = broken - fixed
                if s < best_s:
                    best_s = s
                    best_v = v
            
            if best_v == -1:
                break
            
            # Apply flip
            old_val = polish_assignment[best_v]
            new_val = not old_val
            polish_assignment[best_v] = new_val
            
            for c2_idx, p2 in var_clauses[best_v]:
                was_sat = (old_val == p2)
                if was_sat:
                    p_sat_count[c2_idx] -= 1
                    if p_sat_count[c2_idx] == 0:
                        if not p_is_unsat[c2_idx]:
                            p_is_unsat[c2_idx] = True
                            p_unsat_list.append(c2_idx)
                            p_unsat_count += 1
                else:
                    p_sat_count[c2_idx] += 1
                    if p_sat_count[c2_idx] == 1:
                        if p_is_unsat[c2_idx]:
                            p_is_unsat[c2_idx] = False
                            p_unsat_count -= 1
        
        if p_unsat_count < best_unsat_count:
            return polish_assignment
    
    return best_assignment if best_assignment is not None else assignment
