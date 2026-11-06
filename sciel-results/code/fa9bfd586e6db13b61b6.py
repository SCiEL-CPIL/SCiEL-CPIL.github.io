import time
import random
import math
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n
    
    start_time = time.time()
    deadline = start_time + 325.0  # Safety margin
    
    # Pre-compute clause structure
    # clause_vars: list of lists of (var_idx, polarity)
    # var_clauses: list of lists of (clause_idx, polarity)
    clause_vars = []
    var_clauses = [[] for _ in range(n)]
    
    for ci, c in enumerate(clauses):
        cv = []
        for l in c:
            vi = abs(l) - 1
            pol = 1 if l > 0 else -1
            cv.append((vi, pol))
            var_clauses[vi].append((ci, pol))
        clause_vars.append(cv)
    
    # Initialize assignment and clause unsat counts
    def init_state(rng):
        # assignment as list of 0/1
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Calculate initial unsat counts
        clause_unsat_count = [0] * m
        for ci in range(m):
            count = 0
            for vi, pol in clause_vars[ci]:
                # satisfied if (assignment[vi] == 1 and pol == 1) or (assignment[vi] == 0 and pol == -1)
                if (assignment[vi] == 1 and pol == 1) or (assignment[vi] == 0 and pol == -1):
                    pass
                else:
                    count += 1
            clause_unsat_count[ci] = count
        
        total_unsat = sum(1 for c in clause_unsat_count if c == 3)
        return assignment, clause_unsat_count, total_unsat

    def flip_var(vi, assignment, clause_unsat_count, total_unsat):
        """
        Flip variable vi.
        Returns new total_unsat.
        """
        old_val = assignment[vi]
        new_val = not old_val
        assignment[vi] = new_val
        
        # Update clause_unsat_count for all clauses containing vi
        delta = 0
        for ci, pol in var_clauses[vi]:
            old_count = clause_unsat_count[ci]
            
            # Determine if the literal was satisfied before
            # satisfied if (old_val == 1 and pol == 1) or (old_val == 0 and pol == -1)
            was_sat = (old_val == 1 and pol == 1) or (old_val == 0 and pol == -1)
            
            if was_sat:
                # Now unsatisfied
                clause_unsat_count[ci] = old_count + 1
                if old_count == 2:
                    # Clause was 2 unsat (1 sat), now 3 unsat (0 sat) -> becomes unsatisfied
                    delta += 1
            else:
                # Now satisfied
                clause_unsat_count[ci] = old_count - 1
                if old_count == 3:
                    # Clause was 3 unsat (0 sat), now 2 unsat (1 sat) -> becomes satisfied
                    delta -= 1
        
        total_unsat += delta
        return total_unsat

    def get_gain(vi, assignment, clause_unsat_count):
        """
        Calculate the change in total_unsat if we flip vi.
        """
        old_val = assignment[vi]
        new_val = not old_val
        gain = 0
        for ci, pol in var_clauses[vi]:
            old_count = clause_unsat_count[ci]
            was_sat = (old_val == 1 and pol == 1) or (old_val == 0 and pol == -1)
            
            if was_sat:
                # Flipping makes it unsat
                if old_count == 2:
                    gain += 1
            else:
                # Flipping makes it sat
                if old_count == 3:
                    gain -= 1
        return gain

    def solve_walksat():
        rng = random.Random(42)
        
        best_assignment = None
        best_unsat = m + 1
        
        max_restarts = 1000
        base_steps = 10000
        max_steps = 100000
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            assignment, clause_unsat_count, current_unsat = init_state(rng)
            
            if current_unsat == 0:
                return [bool(x) for x in assignment]
            
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment[:]
            
            steps_in_restart = base_steps + restart * 1000
            if steps_in_restart > max_steps:
                steps_in_restart = max_steps
            
            local_best = current_unsat
            steps_since_improve = 0
            walk_prob = 0.3
            
            for step in range(steps_in_restart):
                if time.time() > deadline:
                    break
                
                # Find unsatisfied clauses
                unsat_clauses = [ci for ci in range(m) if clause_unsat_count[ci] == 3]
                
                if not unsat_clauses:
                    return [bool(x) for x in assignment]
                
                # Pick a random unsatisfied clause
                ci = rng.choice(unsat_clauses)
                
                # Get variables in this clause
                cv = clause_vars[ci]
                
                if rng.random() < walk_prob:
                    # Random flip from the clause
                    vi, _ = rng.choice(cv)
                else:
                    # Greedy flip: pick variable with best gain
                    best_gain = 100
                    best_vi = -1
                    for vi, pol in cv:
                        g = get_gain(vi, assignment, clause_unsat_count)
                        # Tie-breaking: prefer variables with higher frequency in unsat set?
                        # For now, just min gain
                        if g < best_gain:
                            best_gain = g
                            best_vi = vi
                    vi = best_vi
                
                current_unsat = flip_var(vi, assignment, clause_unsat_count, current_unsat)
                
                if current_unsat == 0:
                    return [bool(x) for x in assignment]
                
                if current_unsat < local_best:
                    local_best = current_unsat
                    steps_since_improve = 0
                    walk_prob = max(0.1, walk_prob * 0.95)
                else:
                    steps_since_improve += 1
                    walk_prob = min(0.8, walk_prob * 1.05)
                
                if current_unsat < best_unsat:
                    best_unsat = current_unsat
                    best_assignment = assignment[:]
                
                # If stuck for a long time, reset walk prob
                if steps_since_improve > 5000:
                    walk_prob = 0.3
        
        if best_assignment is not None:
            return [bool(x) for x in best_assignment]
        return [False] * n

    return solve_walksat()
