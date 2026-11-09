"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline guard: 300s limit, use 280s for safety
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Convert to 0-based
    clause_vars = []
    for c in clauses:
        clause_vars.append([(abs(l) - 1, l > 0) for l in c])
    
    # Build variable-clause incidence for component detection
    var_to_clauses = defaultdict(list)
    for ci, cv in enumerate(clause_vars):
        for var, sign in cv:
            var_to_clauses[var].append(ci)
    
    # Find connected components via BFS on variable-clause bipartite graph
    visited_vars = [False] * n
    visited_clauses = [False] * m
    components = []
    
    for v in range(n):
        if visited_vars[v]:
            continue
        # BFS from variable v
        comp_vars = set()
        comp_clauses = set()
        queue = [(v, 'var')]
        visited_vars[v] = True
        
        while queue:
            node, node_type = queue.pop(0)
            if node_type == 'var':
                comp_vars.add(node)
                for ci in var_to_clauses[node]:
                    if not visited_clauses[ci]:
                        visited_clauses[ci] = True
                        queue.append((ci, 'clause'))
            else:
                comp_clauses.add(node)
                for var, sign in clause_vars[node]:
                    if not visited_vars[var]:
                        visited_vars[var] = True
                        queue.append((var, 'var'))
        
        components.append((list(comp_vars), list(comp_clauses)))
    
    # Sort components by size (largest first) to prioritize hard ones
    components.sort(key=lambda x: -len(x[0]))
    
    # Solve each component independently
    assignment = [False] * n
    rng = random.Random(42)
    
    for comp_idx, (comp_vars, comp_clauses) in enumerate(components):
        if time.time() > deadline - 5.0:
            break
        
        # Time budget per component: proportional to size, min 0.5s, max 60s
        num_vars = len(comp_vars)
        num_clauses = len(comp_clauses)
        time_budget = min(60.0, max(0.5, 120.0 * num_vars / n))
        comp_start = time.time()
        
        # Map component variables to local indices
        var_map = {v: i for i, v in enumerate(comp_vars)}
        local_n = len(comp_vars)
        
        # Local clauses
        local_clauses = []
        for ci in comp_clauses:
            local_c = []
            for var, sign in clause_vars[ci]:
                local_c.append((var_map[var], sign))
            local_clauses.append(local_c)
        
        # Local variable-to-clause mapping
        local_var_to_clauses = [[] for _ in range(local_n)]
        for ci, lc in enumerate(local_clauses):
            for var, sign in lc:
                local_var_to_clauses[var].append(ci)
        
        # Solve with WalkSAT on the component
        comp_assignment = [rng.random() < 0.5 for _ in range(local_n)]
        
        def local_satisfied(clause):
            return any(comp_assignment[var] == sign for var, sign in clause)
        
        def count_unsat():
            return sum(1 for c in local_clauses if not local_satisfied(c))
        
        # WalkSAT with gain heuristic
        max_iters = min(50000, max(1000, 50 * num_clauses))
        found = False
        
        for it in range(max_iters):
            if time.time() > comp_start + time_budget:
                break
            
            unsat_count = count_unsat()
            if unsat_count == 0:
                found = True
                break
            
            # Find unsatisfied clauses
            unsat_clauses = [c for c in local_clauses if not local_satisfied(c)]
            if not unsat_clauses:
                found = True
                break
            
            # Pick a random unsatisfied clause
            clause = rng.choice(unsat_clauses)
            
            # Compute gain for each literal in the clause
            best_var = None
            best_gain = -1
            for var, sign in clause:
                # Count how many unsat clauses would be satisfied if we set var=sign
                # vs how many sat clauses would become unsat
                gain = 0
                for ci in local_var_to_clauses[var]:
                    c = local_clauses[ci]
                    is_currently_sat = local_satisfied(c)
                    # If we flip var, check if this clause's satisfaction changes
                    # Current value of var: comp_assignment[var]
                    # New value: not comp_assignment[var]
                    new_val = not comp_assignment[var]
                    
                    # Check if clause is satisfied with new assignment
                    clause_sat_with_flip = any(
                        (comp_assignment[v] if v != var else new_val) == s 
                        for v, s in c
                    )
                    
                    if is_currently_sat and not clause_sat_with_flip:
                        gain -= 1
                    elif not is_currently_sat and clause_sat_with_flip:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_var = var
            
            # If no positive gain, pick randomly (WalkSAT noise)
            if best_gain <= 0 and rng.random() < 0.6:
                best_var = rng.choice(clause)[0]
            
            # Safety check: if best_var is still None, pick randomly from clause
            if best_var is None:
                best_var = rng.choice(clause)[0]
            
            comp_assignment[best_var] = not comp_assignment[best_var]
        
        # If not found via WalkSAT, try a few more random restarts
        if not found:
            for restart in range(3):
                if time.time() > comp_start + time_budget:
                    break
                comp_assignment = [rng.random() < 0.5 for _ in range(local_n)]
                for it in range(max_iters // 2):
                    if time.time() > comp_start + time_budget:
                        break
                    unsat_count = count_unsat()
                    if unsat_count == 0:
                        found = True
                        break
                    unsat_clauses = [c for c in local_clauses if not local_satisfied(c)]
                    if not unsat_clauses:
                        found = True
                        break
                    clause = rng.choice(unsat_clauses)
                    var, sign = rng.choice(clause)
                    comp_assignment[var] = not comp_assignment[var]
                if found:
                    break
        
        # Copy local assignment back to global
        for i, v in enumerate(comp_vars):
            assignment[v] = comp_assignment[i]
    
    return assignment
# EVOLVE-BLOCK-END
