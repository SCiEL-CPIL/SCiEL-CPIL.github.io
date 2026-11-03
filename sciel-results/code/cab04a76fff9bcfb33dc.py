import time
import sys
import random
import numpy as np
import subprocess
import os
import shutil
import glob
import json
import re
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # 1. Precompute neighbors for all words
    total_space = q ** n
    
    def get_digits(num):
        digits = [0] * n
        for i in range(n):
            digits[i] = num % q
            num //= q
        return digits

    q_powers = [1] * n
    for i in range(1, n):
        q_powers[i] = q_powers[i-1] * q

    neighbors = [[] for _ in range(total_space)]
    for i in range(total_space):
        digits = get_digits(i)
        neighbors[i].append(i) # self
        for pos in range(n):
            orig_val = digits[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                delta = (val - orig_val) * q_powers[pos]
                new_index = i + delta
                neighbors[i].append(new_index)
    
    # 2. Greedy Phase: Build a feasible cover
    # We use a fast greedy set cover initialization.
    uncovered_mask = np.ones(total_space, dtype=bool)
    code_set = set()
    code_list = []
    
    # Precompute neighbor arrays for numpy indexing
    neigh_arrays = [np.array(neighbors[i], dtype=np.int64) for i in range(total_space)]
    
    while uncovered_mask.any():
        best_word = -1
        max_cover = 0
        
        # Optimization: Only consider words that are still uncovered?
        # Actually, a word that is already covered might still cover OTHER uncovered words.
        # But if a word is covered, it doesn't mean it's in the code.
        # We want to pick a word that maximizes the number of *newly* covered words.
        
        # Vectorized approach:
        # For each candidate w, count = sum(uncovered_mask[neigh_arrays[w]])
        # We can do this for all w at once if we had a matrix, but memory is tight.
        # Loop is 6561 * 17 ops. ~100k ops per iteration.
        # ~500 iterations. ~50M ops.
        
        # To speed up, we can use a numpy matrix if it fits in memory.
        # 6561 * 6561 * 1 byte = 43 MB. This is fine.
        
        if not hasattr(construct, '_neigh_matrix'):
             # Build the adjacency matrix once
             # M[w, v] = 1 if v is in neighbors[w]
             M = np.zeros((total_space, total_space), dtype=np.uint8)
             for w in range(total_space):
                 M[w, neigh_arrays[w]] = 1
             construct._neigh_matrix = M
             
        M = construct._neigh_matrix
        # counts[w] = sum(uncovered_mask * M[w, :])
        # This is a matrix-vector product.
        counts = M @ uncovered_mask.astype(np.float32)
        
        # We want to pick w with max counts[w]
        # But we should prefer words not already in code?
        # If w is in code, its neighbors are already covered, so counts[w] should be 0?
        # No, uncovered_mask is updated. If w is in code, all its neighbors were set to False.
        # So counts[w] will be 0 for w in code.
        
        best_word = np.argmax(counts)
        
        if counts[best_word] == 0:
             # Should not happen if uncovered_mask.any() is True
             break
             
        code_list.append(best_word)
        code_set.add(best_word)
        uncovered_mask[neigh_arrays[best_word]] = False
        
        if time.time() > deadline:
            break
    
    if not code_list:
        return []
        
    # 3. Local Search / Annealing Phase (Fallback and Pre-processor for CP)
    # We run a fast simulated annealing to get a good initial solution for CP,
    # and as a fallback if CP fails or times out.
    
    def build_coverage(code_set):
        cov = np.zeros(total_space, dtype=np.int32)
        for c in code_set:
            cov[neigh_arrays[c]] += 1
        return cov

    coverage = build_coverage(code_set)
    
    def can_remove(c):
        return np.min(coverage[neigh_arrays[c]]) > 1

    # Simple SA loop
    T = 1.0
    cooling = 0.9995
    min_T = 0.001
    best_code_size = len(code_list)
    best_code_set = set(code_set)
    
    # Time budget for SA: 30 seconds
    sa_deadline = start_time + 30.0
    
    while T > min_T and time.time() < sa_deadline:
        c = random.choice(code_list)
        
        if can_remove(c):
            code_list.remove(c)
            code_set.remove(c)
            coverage[neigh_arrays[c]] -= 1
            if len(code_list) < best_code_size:
                best_code_size = len(code_list)
                best_code_set = set(code_set)
            continue
            
        # Try swap
        c_prime = random.randint(0, total_space - 1)
        if c_prime in code_set:
            T *= cooling
            continue
            
        set_c = set(neigh_arrays[c])
        set_cp = set(neigh_arrays[c_prime])
        critical = set_c - set_cp
        
        valid_swap = True
        for w in critical:
            if coverage[w] == 1:
                valid_swap = False
                break
                
        if valid_swap:
            code_list.remove(c)
            code_set.remove(c)
            code_list.append(c_prime)
            code_set.add(c_prime)
            
            coverage[neigh_arrays[c]] -= 1
            coverage[neigh_arrays[c_prime]] += 1
            
            # Try immediate removal of c_prime or another
            if can_remove(c_prime):
                code_list.remove(c_prime)
                code_set.remove(c_prime)
                coverage[neigh_arrays[c_prime]] -= 1
                if len(code_list) < best_code_size:
                    best_code_size = len(code_list)
                    best_code_set = set(code_set)
            else:
                # Try removing a random other word
                if len(code_list) > 1:
                    c_other = random.choice(code_list)
                    if can_remove(c_other):
                        code_list.remove(c_other)
                        code_set.remove(c_other)
                        coverage[neigh_arrays[c_other]] -= 1
                        if len(code_list) < best_code_size:
                            best_code_size = len(code_list)
                            best_code_set = set(code_set)
        else:
            pass
            
        T *= cooling

    # Restore best found in SA
    if len(best_code_set) < len(code_set):
        code_set = best_code_set
        code_list = list(code_set)
        coverage = build_coverage(code_set)
        
    # 4. CP-SAT Phase
    # Try to improve using OR-Tools CP-SAT
    # We use the SA result as an initial hint and upper bound.
    
    # Check if ortools is available
    try:
        from ortools.sat.python import cp_model
        ortools_available = True
    except ImportError:
        ortools_available = False
        
    if ortools_available and time.time() < deadline - 10:
        model = cp_model.CpModel()
        
        # Variables: x[i] = 1 if word i is in the code
        x = [model.NewBoolVar(f"x_{i}") for i in range(total_space)]
        
        # Constraints: For every word v, at least one neighbor u must be in the code
        # sum(x[u] for u in neighbors[v]) >= 1
        # Note: neighbors[v] includes v itself.
        
        # To speed up, we can use the symmetry breaking:
        # Fix x[0] = 1. The automorphism group of the Hamming graph is transitive,
        # so there exists an optimal solution with 0 in the code.
        model.Add(x[0] == 1)
        
        for v in range(total_space):
            model.Add(model.Sum(x[u] for u in neighbors[v]) >= 1)
            
        # Objective: Minimize sum(x)
        model.Minimize(model.Sum(x))
        
        # Hint: Use the current best code as a hint
        for i in code_set:
            model.AddHint(x[i], 1)
        for i in range(total_space):
            if i not in code_set:
                model.AddHint(x[i], 0)
                
        solver = cp_model.CpSolver()
        
        # Time limit for CP: remaining time - 5s buffer
        remaining_time = deadline - time.time() - 5.0
        if remaining_time > 0:
            solver.parameters.max_time_in_seconds = remaining_time
            solver.parameters.num_workers = 4
            solver.parameters.random_seed = 42
            
            status = solver.Solve(model)
            
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                new_code_set = set()
                for i in range(total_space):
                    if solver.Value(x[i]) == 1:
                        new_code_set.add(i)
                
                # Verify validity
                new_coverage = build_coverage(new_code_set)
                if np.min(new_coverage) >= 1:
                    if len(new_code_set) < len(code_set):
                        code_set = new_code_set
                        code_list = list(code_set)
                    # Even if not smaller, if it's optimal, we are done.
                    # If feasible but larger, keep the smaller one.
                    # Actually, if status is OPTIMAL, this is the best possible.
                    # If FEASIBLE, it might be better than our SA result if SA was bad?
                    # We initialized with SA, so CP starts from there.
                    # If CP finds a better solution, it updates the objective.
                    # We should check if the solution found is actually better than best_code_size
                    # Wait, we updated code_set to new_code_set.
                    # Let's compare with best_code_size from SA.
                    if len(new_code_set) < best_code_size:
                         best_code_size = len(new_code_set)
                         best_code_set = new_code_set
                         code_set = new_code_set
                         code_list = list(code_set)
                
                # If we found an optimal solution, we can return immediately
                if status == cp_model.OPTIMAL:
                    # Convert and return
                    result = []
                    for idx in sorted(code_set):
                        digits = get_digits(idx)
                        result.append(digits)
                    return result

    # 5. Final conversion
    # Use the best code found
    if len(best_code_set) < len(code_set):
        code_set = best_code_set
        
    result = []
    for idx in sorted(code_set):
        digits = get_digits(idx)
        result.append(digits)
        
    return result
