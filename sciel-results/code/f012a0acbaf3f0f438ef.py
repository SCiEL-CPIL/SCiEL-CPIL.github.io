"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
    
    # Deadline guard: 325s safety margin out of 330s limit
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Pre-compute clause structure
    clause_vars = []
    for c in clauses:
        cv = [(abs(l) - 1, l > 0) for l in c]
        clause_vars.append(cv)
    
    # For each variable, store list of (clause_idx, is_positive)
    var_clauses = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for vi, pos in cv:
            var_clauses[vi].append((ci, pos))
    
    # Build adjacency: for each variable, list of clause indices
    var_clause_list = []
    for vi in range(n):
        var_clause_list.append([ci for ci, pos in var_clauses[vi]])
    
    # Build clause-to-variable mapping for fast lookup
    # clause_var_idx[ci] = list of variable indices in clause ci
    clause_var_idx = []
    for ci, cv in enumerate(clause_vars):
        clause_var_idx.append([vi for vi, pos in cv])
    
    def init_state(rng):
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_unsat_count = [0] * m
        clause_weight = [1.0] * m
        for ci, cv in enumerate(clause_vars):
            count = 0
            for vi, pos in cv:
                if assignment[vi] != pos:
                    count += 1
            clause_unsat_count[ci] = count
        return assignment, clause_unsat_count, clause_weight
    
    def get_weighted_unsat(clause_unsat_count, clause_weight):
        total = 0.0
        for i in range(m):
            if clause_unsat_count[i] >= 3:
                total += clause_weight[i]
        return total
    
    def find_unsat_clause(clause_unsat_count, clause_weight, rng):
        max_w = -1
        candidates = []
        for ci in range(m):
            if clause_unsat_count[ci] >= 3:
                if clause_weight[ci] > max_w:
                    max_w = clause_weight[ci]
                    candidates = [ci]
                elif clause_weight[ci] == max_w:
                    candidates.append(ci)
        if not candidates:
            return -1
        return rng.choice(candidates)
    
    def flip_var(vi, assignment, clause_unsat_count):
        old_val = assignment[vi]
        assignment[vi] = not old_val
        for ci, pos in var_clauses[vi]:
            if old_val == pos:
                clause_unsat_count[ci] += 1
            else:
                clause_unsat_count[ci] -= 1
    
    def polish_pass(assignment, clause_unsat_count, max_iter=100000):
        """
        Final high-precision polish pass using exact local search.
        Uses a more sophisticated approach with backtracking on small subproblems.
        """
        assignment = list(assignment)
        clause_unsat_count = list(clause_unsat_count)
        
        # First pass: greedy flips
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Find the variable that, when flipped, reduces the total unsat count the most
            best_var = -1
            best_improvement = 0
            
            # Only consider variables that appear in unsatisfied clauses
            candidates = set()
            for ci in range(m):
                if clause_unsat_count[ci] >= 3:
                    for vi in clause_var_idx[ci]:
                        candidates.add(vi)
            
            for vi in candidates:
                improvement = 0
                for cj, cpos in var_clauses[vi]:
                    if assignment[vi] == cpos:
                        if clause_unsat_count[cj] == 2:
                            improvement -= 1
                    else:
                        if clause_unsat_count[cj] == 3:
                            improvement += 1
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_var = vi
            
            if best_var == -1 or best_improvement <= 0:
                break
            
            flip_var(best_var, assignment, clause_unsat_count)
            
            all_sat = all(clause_unsat_count[ci] < 3 for ci in range(m))
            if all_sat:
                return assignment
        
        # Second pass: try 2-flip moves to escape local minima
        for _ in range(min(max_iter // 2, 50000)):
            if time.time() > deadline:
                break
            
            best_pair = None
            best_pair_improvement = 0
            
            # Get unsatisfied clauses
            unsat_clauses = [ci for ci in range(m) if clause_unsat_count[ci] >= 3]
            
            if not unsat_clauses:
                break
            
            # Try pairs of variables from the first few unsatisfied clauses
            for ci in unsat_clauses[:5]:
                for vi in clause_var_idx[ci]:
                    for vj in clause_var_idx[ci]:
                        if vi >= vj:
                            continue
                        
                        # Calculate improvement for flipping both vi and vj
                        improvement = 0
                        for cj, cpos in var_clauses[vi]:
                            if assignment[vi] == cpos:
                                if clause_unsat_count[cj] == 2:
                                    improvement -= 1
                            else:
                                if clause_unsat_count[cj] == 3:
                                    improvement += 1
                        
                        for cj, cpos in var_clauses[vj]:
                            if assignment[vj] == cpos:
                                if clause_unsat_count[cj] == 2:
                                    improvement -= 1
                            else:
                                if clause_unsat_count[cj] == 3:
                                    improvement += 1
                        
                        if improvement > best_pair_improvement:
                            best_pair_improvement = improvement
                            best_pair = (vi, vj)
            
            if best_pair is None or best_pair_improvement <= 0:
                break
            
            vi, vj = best_pair
            flip_var(vi, assignment, clause_unsat_count)
            flip_var(vj, assignment, clause_unsat_count)
            
            all_sat = all(clause_unsat_count[ci] < 3 for ci in range(m))
            if all_sat:
                return assignment
        
        return assignment
    
    best_assignment = None
    best_weighted_unsat = float('inf')
    
    rng = random.Random(0)
    
    # Parameters for Weighted WalkSAT
    max_restarts = 150
    base_steps = 15000
    max_steps = 200000
    stagnation_limit = 1500
    weight_boost = 2.0
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        assignment, clause_unsat_count, clause_weight = init_state(rng)
        current_weighted_unsat = get_weighted_unsat(clause_unsat_count, clause_weight)
        
        if current_weighted_unsat == 0:
            return assignment
        
        if current_weighted_unsat < best_weighted_unsat:
            best_weighted_unsat = current_weighted_unsat
            best_assignment = list(assignment)
        
        steps_in_restart = base_steps + restart * 3000
        steps_in_restart = min(steps_in_restart, max_steps)
        
        steps_without_improvement = 0
        local_best_unsat = current_weighted_unsat
        
        for step in range(steps_in_restart):
            if time.time() > deadline:
                break
            
            unsat_idx = find_unsat_clause(clause_unsat_count, clause_weight, rng)
            if unsat_idx == -1:
                return assignment
            
            cv = clause_vars[unsat_idx]
            
            best_var = -1
            best_delta = float('inf')
            
            for vi, pos in cv:
                delta = 0.0
                for cj, cpos in var_clauses[vi]:
                    if cj == unsat_idx:
                        if clause_unsat_count[cj] == 3:
                            delta -= clause_weight[cj]
                    else:
                        if assignment[vi] == cpos:
                            if clause_unsat_count[cj] == 2:
                                delta += clause_weight[cj]
                        else:
                            if clause_unsat_count[cj] == 3:
                                delta -= clause_weight[cj]
                
                score = delta + rng.random() * 0.01
                if score < best_delta:
                    best_delta = score
                    best_var = vi
            
            if best_var == -1:
                best_var = rng.choice(cv)[0]
            
            flip_var(best_var, assignment, clause_unsat_count)
            
            current_weighted_unsat = get_weighted_unsat(clause_unsat_count, clause_weight)
            
            if current_weighted_unsat < local_best_unsat:
                local_best_unsat = current_weighted_unsat
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
            
            if current_weighted_unsat == 0:
                return assignment
            
            if current_weighted_unsat < best_weighted_unsat:
                best_weighted_unsat = current_weighted_unsat
                best_assignment = list(assignment)
            
            if steps_without_improvement > stagnation_limit:
                for ci in range(m):
                    if clause_unsat_count[ci] >= 3:
                        clause_weight[ci] *= weight_boost
                steps_without_improvement = 0
        
        if time.time() > deadline:
            break
    
    if best_assignment is not None:
        clause_unsat_count = [0] * m
        for ci, cv in enumerate(clause_vars):
            count = 0
            for vi, pos in cv:
                if best_assignment[vi] != pos:
                    count += 1
            clause_unsat_count[ci] = count
        
        polished = polish_pass(best_assignment, clause_unsat_count, max_iter=50000)
        return polished
    
    rng2 = random.Random(42)
    return [rng2.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
