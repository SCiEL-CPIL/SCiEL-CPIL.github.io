"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Precompute clause literals as (var_index, polarity) for speed
    # var_index is 0-based, polarity is +1 or -1
    cl = []
    for c in clauses:
        cl.append([(abs(l) - 1, 1 if l > 0 else -1) for l in c])
    
    # Precompute variable to clauses mapping
    # var_to_clauses[var] = list of (clause_idx, polarity_in_clause)
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(cl):
        for var, pol in clause:
            var_to_clauses[var].append((i, pol))
    
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s/330s limit
    
    # Best known assignment (by number of satisfied clauses)
    best_sat_count = -1
    best_assign = None
    
    rng = random.Random(0)
    
    # WalkSAT parameters
    max_restarts = 2000
    base_steps = 8000
    
    def init_state(assign):
        """Initialize clause satisfaction counts and unsat set."""
        clause_sat_count = [0] * m
        unsat_set = set()
        
        for i in range(m):
            sat_count = 0
            for var, pol in cl[i]:
                if assign[var] == (pol > 0):
                    sat_count += 1
            clause_sat_count[i] = sat_count
            if sat_count == 0:
                unsat_set.add(i)
                
        return clause_sat_count, unsat_set

    def flip_variable(assign, var, target_val, clause_sat_count, unsat_set):
        """Flip variable to target_val and update state with O(1) unsat tracking."""
        old_val = assign[var]
        if old_val == target_val:
            return
        
        assign[var] = target_val
        
        for c_idx, pol in var_to_clauses[var]:
            new_lit_sat = (pol > 0) == target_val
            old_lit_sat = (pol > 0) == old_val
            
            if new_lit_sat != old_lit_sat:
                old_count = clause_sat_count[c_idx]
                new_count = old_count + (1 if new_lit_sat else -1)
                clause_sat_count[c_idx] = new_count
                
                old_unsat = (old_count == 0)
                new_unsat = (new_count == 0)
                
                if old_unsat and not new_unsat:
                    unsat_set.discard(c_idx)
                elif not old_unsat and new_unsat:
                    unsat_set.add(c_idx)

    def dpll_unit_propagation(assign, clause_sat_count, unsat_set):
        """
        Perform unit propagation. Returns (success, assign, clause_sat_count, unsat_set).
        """
        unassigned = [True] * n
        
        # Build a list of clauses that are currently unit
        unit_clauses = []
        for i in range(m):
            if clause_sat_count[i] > 0:
                continue
            unassigned_count = 0
            for var, pol in cl[i]:
                if unassigned[var]:
                    unassigned_count += 1
            if unassigned_count == 1:
                unit_clauses.append(i)
        
        while unit_clauses:
            if time.time() > deadline:
                return False, assign, clause_sat_count, unsat_set
            
            ci = unit_clauses.pop()
            
            for var, pol in cl[ci]:
                if unassigned[var]:
                    target_val = (pol > 0)
                    unassigned[var] = False
                    flip_variable(assign, var, target_val, clause_sat_count, unsat_set)
                    
                    for c_idx, c_pol in var_to_clauses[var]:
                        if clause_sat_count[c_idx] > 0:
                            continue
                        unassigned_count = 0
                        for v2, p2 in cl[c_idx]:
                            if unassigned[v2]:
                                unassigned_count += 1
                        if unassigned_count == 1:
                            unit_clauses.append(c_idx)
                        elif unassigned_count == 0:
                            return False, assign, clause_sat_count, unsat_set
                    break
        
        if not unsat_set:
            return True, assign, clause_sat_count, unsat_set
        
        return False, assign, clause_sat_count, unsat_set

    def spectral_init():
        """Use SVD to get an initial assignment."""
        try:
            import numpy as np
            
            A = np.zeros((m, n))
            for i, clause in enumerate(cl):
                for var, pol in clause:
                    A[i, var] = pol
            
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            
            best_assign = None
            best_sat = -1
            
            for k in [1, 2, 3, 5, 10]:
                if k > Vt.shape[0]:
                    break
                scores = Vt[:k, :] @ (s[:k] ** 2)
                assign_k = [score > 0 for score in scores]
                
                sat_count = 0
                for i in range(m):
                    for var, pol in cl[i]:
                        if assign_k[var] == (pol > 0):
                            sat_count += 1
                            break
                
                if sat_count > best_sat:
                    best_sat = sat_count
                    best_assign = list(assign_k)
            
            if best_assign is not None and best_sat > m * 0.5:
                return best_assign
            
        except Exception:
            pass
        
        return None

    def penalty_walksat(assign, max_steps, rng):
        """
        Penalty-weighted WalkSAT: dynamic penalty increases for persistently unsat clauses.
        Models Max-SAT as optimization with smooth surrogate objective.
        Returns (satisfied, assign, sat_count).
        """
        clause_sat_count, unsat_set = init_state(assign)
        
        # Penalty weights for each clause - start at 1.0
        clause_weights = [1.0] * m
        # Track how long each clause has been unsatisfied
        unsat_duration = [0] * m
        
        penalty_growth_rate = 0.05  # How fast penalties grow
        max_penalty = 10.0
        
        for step in range(max_steps):
            if not unsat_set:
                return True, assign, m
            
            if step % 200 == 0 and time.time() > deadline:
                break
            
            # Update penalties: increase for unsat clauses, decay for sat clauses
            if step % 50 == 0:
                for ci in range(m):
                    if ci in unsat_set:
                        unsat_duration[ci] += 1
                        clause_weights[ci] = min(max_penalty, 1.0 + penalty_growth_rate * unsat_duration[ci])
                    else:
                        unsat_duration[ci] = 0
                        clause_weights[ci] = max(1.0, clause_weights[ci] * 0.9)
            
            # Pick a weighted-random unsatisfied clause
            unsat_list = list(unsat_set)
            if len(unsat_list) <= 1:
                ci = unsat_list[0] if unsat_list else 0
            else:
                # Weighted selection based on clause_weights
                weights = [clause_weights[ci] for ci in unsat_list]
                total_w = sum(weights)
                r = rng.random() * total_w
                cumulative = 0
                ci = unsat_list[-1]
                for idx, w in enumerate(weights):
                    cumulative += w
                    if r <= cumulative:
                        ci = unsat_list[idx]
                        break
            
            clause = cl[ci]
            
            # Adaptive noise: higher when stuck (many unsat), lower when close
            n_unsat = len(unsat_set)
            base_noise = 0.3 + 0.4 * (n_unsat / m)
            noise = base_noise + rng.gauss(0, 0.1)
            noise = max(0.1, min(0.9, noise))
            
            # Find best variable to flip using penalty-weighted conflicts
            best_var = None
            best_score = float('inf')
            best_pol = None
            
            for var, pol in clause:
                target_val = (pol > 0)
                
                # Calculate penalty-weighted conflicts
                score = 0.0
                
                for c_idx, c_pol in var_to_clauses[var]:
                    if c_idx == ci:
                        continue
                    
                    if clause_sat_count[c_idx] > 0:
                        if (c_pol > 0) != target_val:
                            if clause_sat_count[c_idx] == 1:
                                score += clause_weights[c_idx]
                    
                # Small random tie-breaker
                score += rng.random() * 0.01
                
                if score < best_score:
                    best_score = score
                    best_var = var
                    best_pol = pol
            
            # With probability noise, flip a random variable in the clause
            if rng.random() < noise:
                var, pol = clause[rng.randrange(len(clause))]
            else:
                var = best_var
                pol = best_pol
            
            target_val = (pol > 0)
            flip_variable(assign, var, target_val, clause_sat_count, unsat_set)
        
        sat_count = m - len(unsat_set)
        return False, assign, sat_count

    # Try spectral initialization first
    spectral_assign = spectral_init()
    
    # Main loop with restarts
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize: use spectral init for first few restarts, then random
        if restart < 3 and spectral_assign is not None:
            assign = list(spectral_assign)
        else:
            assign = [rng.random() < 0.5 for _ in range(n)]
        
        # Try unit propagation first
        clause_sat_count, unsat_set = init_state(assign)
        solved, assign, clause_sat_count, unsat_set = dpll_unit_propagation(assign, clause_sat_count, unsat_set)
        
        if solved:
            return assign
        
        # Run penalty WalkSAT
        solved, assign, sat_count = penalty_walksat(assign, base_steps, rng)
        
        if sat_count > best_sat_count:
            best_sat_count = sat_count
            best_assign = list(assign)
        
        if solved:
            return assign
    
    # If we didn't solve, return the best assignment found
    if best_assign is not None:
        return best_assign
    
    # Fallback: random assignment
    return [random.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
