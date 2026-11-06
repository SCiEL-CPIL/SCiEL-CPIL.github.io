"""Improved 3-SAT solver using WalkSAT with incremental clause tracking and adaptive noise."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    start_time = time.time()
    time_limit = 290.0
    
    # Precompute: for each variable, list of (clause_index, sign) where sign=1 means positive literal
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else 0
            var_to_clauses[v].append((ci, sign))
    
    best_assignment = None
    best_unsat_count = num_clauses
    
    max_restarts = 2000
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit - 3.0:
            break
        
        rng = random.Random(restart * 12345 + 7)
        
        # Initial random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize clause satisfaction counts
        clause_sat = [0] * num_clauses
        for ci, clause in enumerate(clauses):
            count = 0
            for lit in clause:
                v = abs(lit) - 1
                if assignment[v] == (lit > 0):
                    count += 1
            clause_sat[ci] = count
        
        curr_unsat = sum(1 for c in clause_sat if c == 0)
        
        if curr_unsat == 0:
            return assignment
        
        if curr_unsat < best_unsat_count:
            best_unsat_count = curr_unsat
            best_assignment = assignment[:]
        
        # WalkSAT with adaptive noise
        base_noise = 0.35
        max_flips = 50000 + 100 * n
        
        recent_unsat_changes = []
        
        total_flips = 0
        while total_flips < max_flips:
            if time.time() - start_time > time_limit - 1.0:
                break
            
            # Find an unsatisfied clause (random offset for diversity)
            start_idx = rng.randint(0, num_clauses - 1)
            unsat_clause_idx = -1
            for i in range(num_clauses):
                ci = (start_idx + i) % num_clauses
                if clause_sat[ci] == 0:
                    unsat_clause_idx = ci
                    break
            
            if unsat_clause_idx == -1:
                return assignment
            
            clause = clauses[unsat_clause_idx]
            
            # Compute gain for each literal in the unsat clause
            # Gain = (# clauses that become satisfied) - (# clauses that become unsatisfied)
            gains = []
            for lit in clause:
                v = abs(lit) - 1
                sign = 1 if lit > 0 else 0
                
                satisfied = 0
                unsatisfied = 0
                for (ci2, sign2) in var_to_clauses[v]:
                    current_sat = (assignment[v] == (sign2 == 1))
                    new_sat = ((not assignment[v]) == (sign2 == 1))
                    
                    if current_sat and not new_sat:
                        # This clause was satisfied, will become unsatisfied
                        # Only counts if it was exactly satisfied (count == 1)
                        if clause_sat[ci2] == 1:
                            unsatisfied += 1
                    elif not current_sat and new_sat:
                        # This clause was unsatisfied, will become satisfied
                        # Only counts if it was exactly unsatisfied (count == 0)
                        if clause_sat[ci2] == 0:
                            satisfied += 1
                
                gain = satisfied - unsatisfied
                gains.append((gain, v))
            
            # Select flip: with probability p, random; otherwise best gain
            if rng.random() < base_noise:
                lit_choice = rng.choice(clause)
                v = abs(lit_choice) - 1
            else:
                max_gain = max(g for g, _ in gains)
                candidates = [v for g, v in gains if g == max_gain]
                v = rng.choice(candidates)
            
            # Perform the flip
            old_val = assignment[v]
            assignment[v] = not old_val
            
            # Update clause_sat incrementally
            delta_unsat = 0
            for (ci, sign) in var_to_clauses[v]:
                old_sat = (old_val == (sign == 1))
                new_sat = (assignment[v] == (sign == 1))
                
                if old_sat and not new_sat:
                    clause_sat[ci] -= 1
                    if clause_sat[ci] == 0:
                        delta_unsat += 1
                elif not old_sat and new_sat:
                    clause_sat[ci] += 1
                    if clause_sat[ci] == 1:
                        delta_unsat -= 1
            
            curr_unsat += delta_unsat
            total_flips += 1
            
            if curr_unsat == 0:
                return assignment
            
            if curr_unsat < best_unsat_count:
                best_unsat_count = curr_unsat
                best_assignment = assignment[:]
            
            # Adapt noise parameter every 50 flips
            if total_flips % 50 == 0:
                recent_unsat_changes.append(curr_unsat)
                if len(recent_unsat_changes) > 20:
                    recent_unsat_changes.pop(0)
                
                if len(recent_unsat_changes) >= 10:
                    recent_avg = sum(recent_unsat_changes[-5:]) / 5.0
                    prev_avg = sum(recent_unsat_changes[:5]) / 5.0
                    if recent_avg < prev_avg - 0.5:
                        base_noise = max(0.1, base_noise - 0.02)
                    elif recent_avg > prev_avg + 0.5:
                        base_noise = min(0.6, base_noise + 0.03)
        
        if time.time() - start_time > time_limit - 5.0:
            break
    
    if best_assignment is not None:
        return best_assignment
    return assignment
# EVOLVE-BLOCK-END
