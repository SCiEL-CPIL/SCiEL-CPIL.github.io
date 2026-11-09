"""SAT solver for planted 3-SAT instances."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.linalg import eigh


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 295.0  # 5s margin before 300s limit
    
    # Convert to 0-based internal representation
    clauses_internal = []
    for c in clauses:
        clause_int = []
        for lit in c:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            clause_int.append((var, sign))
        clauses_internal.append(clause_int)
    
    # Build variable-to-clause index
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses_internal):
        for var, sign in clause:
            var_to_clauses[var].append((ci, sign))
    
    # === VARIABLE ELIMINATION HEURISTIC ===
    # Greedily fix variables with highest consistency (same polarity in most clauses)
    # This exploits planted structure where the planted assignment makes most clauses consistent
    
    assignment = [None] * n  # None = unassigned, True/False = assigned
    remaining_clauses = set(range(m))  # indices of not-yet-satisfied clauses
    
    # For each variable, track: total clauses, pos clauses, neg clauses
    var_pos = [0] * n  # count of clauses where var appears positive
    var_neg = [0] * n  # count of clauses where var appears negative
    
    for ci, clause in enumerate(clauses_internal):
        for var, sign in clause:
            if sign == 1:
                var_pos[var] += 1
            else:
                var_neg[var] += 1
    
    # Iterative variable elimination
    # At each step, pick the unassigned variable with highest "consistency score"
    # consistency = max(pos, neg) / (pos + neg) if pos+neg > 0
    # Fix it to the majority polarity
    
    # Use a priority queue approach: recompute scores periodically
    # For efficiency, we'll do batches of eliminations
    
    max_elim_steps = 100000  # safety cap
    elim_steps = 0
    batch_size = 50  # eliminate in batches to amortize cost
    
    while elim_steps < max_elim_steps:
        if time.time() > deadline - 10:
            break
        
        # Compute consistency scores for unassigned variables
        best_var = -1
        best_score = -1.0
        best_val = False
        
        for v in range(n):
            if assignment[v] is not None:
                continue
            total = var_pos[v] + var_neg[v]
            if total == 0:
                continue
            # Consistency: how skewed is the polarity distribution
            pos_frac = var_pos[v] / total
            neg_frac = var_neg[v] / total
            consistency = max(pos_frac, neg_frac)
            # Weight by total clause count to prefer variables in more clauses
            score = consistency * np.log1p(total)
            
            if score > best_score:
                best_score = score
                best_var = v
                best_val = (var_pos[v] >= var_neg[v])
        
        if best_var == -1:
            break
        
        # Assign the variable
        assignment[best_var] = best_val
        
        # Update remaining clauses: remove satisfied clauses, update counts
        new_remaining = set()
        for ci in remaining_clauses:
            clause = clauses_internal[ci]
            satisfied = False
            for var, sign in clause:
                if assignment[var] is not None:
                    if assignment[var] == (sign == 1):
                        satisfied = True
                        break
                    # Variable assigned but doesn't satisfy this clause
                    # Remove from clause consideration
                # else: unassigned, keep
            if not satisfied:
                new_remaining.add(ci)
        
        remaining_clauses = new_remaining
        
        if not remaining_clauses:
            break
        
        elim_steps += 1
        
        # Periodically recompute clause counts for remaining clauses
        if elim_steps % batch_size == 0:
            var_pos = [0] * n
            var_neg = [0] * n
            for ci in remaining_clauses:
                clause = clauses_internal[ci]
                for var, sign in clause:
                    if assignment[var] is None:
                        if sign == 1:
                            var_pos[var] += 1
                        else:
                            var_neg[var] += 1
    
    # Check if all clauses are satisfied
    all_satisfied = True
    for ci in remaining_clauses:
        clause = clauses_internal[ci]
        sat = False
        for var, sign in clause:
            if assignment[var] is not None and assignment[var] == (sign == 1):
                sat = True
                break
        if not sat:
            all_satisfied = False
            break
    
    if all_satisfied:
        # Fill unassigned variables arbitrarily
        for v in range(n):
            if assignment[v] is None:
                assignment[v] = False
        return [bool(x) for x in assignment]
    
    # Variable elimination didn't fully solve it.
    # Now use WalkSAT on the remaining unsatisfied clauses with the partial assignment.
    
    # Build the residual problem
    unsat_clause_indices = []
    for ci in remaining_clauses:
        clause = clauses_internal[ci]
        sat = False
        for var, sign in clause:
            if assignment[var] is not None and assignment[var] == (sign == 1):
                sat = True
                break
        if not sat:
            unsat_clause_indices.append(ci)
    
    if not unsat_clause_indices:
        for v in range(n):
            if assignment[v] is None:
                assignment[v] = False
        return [bool(x) for x in assignment]
    
    # Collect unassigned variables
    unassigned_vars = [v for v in range(n) if assignment[v] is None]
    if not unassigned_vars:
        # All assigned but not all satisfied - shouldn't happen, return as-is
        return [bool(x) if x is not None else False for x in assignment]
    
    # Build local index
    var_map = {v: i for i, v in enumerate(unassigned_vars)}
    local_n = len(unassigned_vars)
    
    # Build local clauses (only unassigned variables, since assigned ones that don't satisfy are effectively false)
    local_clauses = []
    for ci in unsat_clause_indices:
        clause = clauses_internal[ci]
        local_clause = []
        for var, sign in clause:
            if assignment[var] is None:
                local_clause.append((var_map[var], sign))
        # If clause has no unassigned vars, it's unsatisfiable with current assignment
        # But we can't change assigned vars, so skip (or handle specially)
        if local_clause:
            local_clauses.append(local_clause)
    
    if not local_clauses:
        # All remaining clauses have no free variables - unsatisfiable with current assignment
        # Fall through to full WalkSAT restart
        pass
    else:
        # WalkSAT on the residual problem
        rng = random.Random(42)
        
        # Initial assignment for unassigned vars: use the elimination's hint if available
        # Otherwise random
        current = [False] * local_n
        for v in unassigned_vars:
            # Use the polarity bias from elimination
            total = var_pos[v] + var_neg[v]
            if total > 0:
                current[var_map[v]] = (var_pos[v] >= var_neg[v])
            else:
                current[var_map[v]] = rng.random() < 0.5
        
        def count_unsat(assign):
            count = 0
            for lc in local_clauses:
                sat = False
                for li, sign in lc:
                    if assign[li] == (sign == 1):
                        sat = True
                        break
                if not sat:
                    count += 1
            return count
        
        def get_unsat_clauses(assign):
            unsat = []
            for i, lc in enumerate(local_clauses):
                sat = False
                for li, sign in lc:
                    if assign[li] == (sign == 1):
                        sat = True
                        break
                if not sat:
                    unsat.append(i)
            return unsat
        
        current_unsat = count_unsat(current)
        
        # WalkSAT with restarts
        ws_deadline = deadline - 5
        best_assign = list(current)
        best_unsat = current_unsat
        noise = 0.5
        patience = 2000
        max_steps_per_run = 50000
        
        while time.time() < ws_deadline:
            no_improve = 0
            run_best = current_unsat
            run_best_assign = list(current)
            
            for step in range(max_steps_per_run):
                if time.time() > ws_deadline:
                    break
                
                if current_unsat == 0:
                    # Success!
                    for i, v in enumerate(unassigned_vars):
                        assignment[v] = current[i]
                    for v in range(n):
                        if assignment[v] is None:
                            assignment[v] = False
                    return [bool(x) for x in assignment]
                
                if current_unsat < run_best:
                    run_best = current_unsat
                    run_best_assign = list(current)
                    no_improve = 0
                    if current_unsat < best_unsat:
                        best_unsat = current_unsat
                        best_assign = list(current)
                else:
                    no_improve += 1
                
                if no_improve > patience:
                    break
                
                unsat_clauses = get_unsat_clauses(current)
                if not unsat_clauses:
                    for i, v in enumerate(unassigned_vars):
                        assignment[v] = current[i]
                    for v in range(n):
                        if assignment[v] is None:
                            assignment[v] = False
                    return [bool(x) for x in assignment]
                
                chosen = rng.choice(unsat_clauses)
                lc = local_clauses[chosen]
                
                # Pick variable to flip
                if rng.random() < noise:
                    candidates = [li for li, sign in lc if current[li] != (sign == 1)]
                    if not candidates:
                        candidates = [li for li, sign in lc]
                    best_var = rng.choice(candidates)
                else:
                    # Min-delta heuristic
                    best_var = None
                    best_delta = float('inf')
                    for li, sign in lc:
                        if current[li] == (sign == 1):
                            continue
                        delta = 0
                        for ci2, lc2 in enumerate(local_clauses):
                            was_sat = any(current[lli] == (s == 1) for lli, s in lc2)
                            would_be_sat = False
                            for lli, s in lc2:
                                if lli == li:
                                    if (not current[li]) == (s == 1):
                                        would_be_sat = True
                                        break
                                else:
                                    if current[lli] == (s == 1):
                                        would_be_sat = True
                                        break
                            if was_sat and not would_be_sat:
                                delta += 1
                            elif not was_sat and would_be_sat:
                                delta -= 1
                        if delta < best_delta:
                            best_delta = delta
                            best_var = li
                    if best_var is None:
                        candidates = [li for li, sign in lc if current[li] != (sign == 1)]
                        if not candidates:
                            candidates = [li for li, sign in lc]
                        best_var = rng.choice(candidates)
                
                current[best_var] = not current[best_var]
                current_unsat = count_unsat(current)
            
            # Restart with best found so far
            current = list(run_best_assign)
            current_unsat = run_best
            noise = max(0.2, noise * 0.95)
        
        # Use best assignment found
        for i, v in enumerate(unassigned_vars):
            assignment[v] = best_assign[i]
    
    # Fill any remaining unassigned
    for v in range(n):
        if assignment[v] is None:
            assignment[v] = False
    
    return [bool(x) for x in assignment]


# EVOLVE-BLOCK-END
