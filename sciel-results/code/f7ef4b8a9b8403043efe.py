"""Improved 3-SAT solver with portfolio search: WalkSAT with restarts, clause weighting,
min-conflicts heuristic, variable occurrence tracking, and time budget."""
import random
import time
import math


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Explicit deadline guard: 330s wall-clock limit, use 315s as safety margin
    start_time = time.time()
    TIME_LIMIT = 315.0
    
    def time_remaining():
        return TIME_LIMIT - (time.time() - start_time)
    
    if time_remaining() <= 0:
        return [True] * n
    
    # Precompute clause literals as (var_index, is_positive) for speed
    clause_data = []
    for c in clauses:
        cd = [(abs(l) - 1, l > 0) for l in c]
        clause_data.append(cd)
    
    m = len(clause_data)
    
    # Track variable occurrences for better heuristics
    var_count = [0] * n
    var_clauses = [[] for _ in range(n)]  # var -> list of clause indices containing var
    for i, clause in enumerate(clause_data):
        for var, pos in clause:
            var_count[var] += 1
            var_clauses[var].append(i)
    
    # Precompute for each clause, which variables it contains (for fast lookup)
    clause_vars = [set(v for v, _ in c) for c in clause_data]
    
    def is_satisfied(assignment, clause):
        for var, pos in clause:
            if assignment[var] == pos:
                return True
        return False
    
    def count_unsat(assignment):
        cnt = 0
        for clause in clause_data:
            sat = False
            for var, pos in clause:
                if assignment[var] == pos:
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt
    
    def full_satisfied(assignment):
        for clause in clause_data:
            sat = False
            for var, pos in clause:
                if assignment[var] == pos:
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    def find_unsat_clauses(assignment):
        """Return list of indices of unsatisfied clauses."""
        unsat = []
        for i, clause in enumerate(clause_data):
            sat = False
            for var, pos in clause:
                if assignment[var] == pos:
                    sat = True
                    break
            if not sat:
                unsat.append(i)
        return unsat
    
    def compute_flip_gain(assignment, var, clause_idx):
        """
        Compute the net gain in satisfied clauses if we flip variable 'var'.
        Gain = (# clauses that become satisfied) - (# clauses that become unsatisfied)
        Uses precomputed var_clauses for efficiency.
        """
        gain = 0
        current_val = assignment[var]
        
        for ci in var_clauses[var]:
            clause = clause_data[ci]
            
            # Check if this clause is currently satisfied
            currently_sat = False
            only_this_satisfies = True  # Is 'var' the only satisfying literal?
            
            for v, p in clause:
                if assignment[v] == p:
                    currently_sat = True
                    if v != var:
                        only_this_satisfies = False
                    break
            
            if currently_sat:
                # Clause is satisfied; if we flip var, it might become unsatisfied
                if only_this_satisfies:
                    # var is the only one satisfying this clause, so flipping breaks it
                    gain -= 1
            else:
                # Clause is unsatisfied; check if flipping var would satisfy it
                # Find the literal for var in this clause
                for v, p in clause:
                    if v == var:
                        # After flip, var's value is not current_val
                        if (not current_val) == p:
                            gain += 1
                        break
        
        return gain
    
    def walksat_restarts(seed, max_restarts, steps_per_restart, clause_weight_update=True, use_min_conflicts=True):
        """WalkSAT with restarts, optional clause weighting, min-conflicts, and best-tracked assignment."""
        rng = random.Random(seed)
        
        best_assignment = None
        best_unsat_count = float('inf')
        
        # Initialize with random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Clause weights for weighted random selection
        weights = [1.0] * m
        
        max_steps = steps_per_restart
        max_restarts_effective = max_restarts
        
        current_unsat = count_unsat(assignment)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        if current_unsat == 0:
            return assignment[:]
        
        for restart in range(max_restarts_effective):
            if time_remaining() < 1.0:
                break
            
            # Decay weights on restart
            if clause_weight_update and restart > 0:
                for i in range(m):
                    weights[i] = max(1.0, weights[i] * 0.85)
            
            # Run WalkSAT for max_steps
            for step in range(max_steps):
                if time_remaining() < 1.0:
                    break
                
                # Find unsatisfied clauses
                unsat_indices = find_unsat_clauses(assignment)
                
                if not unsat_indices:
                    return assignment[:]
                
                # Pick a random unsatisfied clause (weighted)
                if clause_weight_update and len(unsat_indices) > 0:
                    unsat_weights = [weights[i] for i in unsat_indices]
                    total_w = sum(unsat_weights)
                    if total_w > 0:
                        r = rng.random() * total_w
                        cum = 0
                        chosen_idx_in_unsat = 0
                        for j, w in enumerate(unsat_weights):
                            cum += w
                            if r <= cum:
                                chosen_idx_in_unsat = j
                                break
                        chosen_unsat_idx = unsat_indices[chosen_idx_in_unsat]
                    else:
                        chosen_unsat_idx = unsat_indices[rng.randrange(len(unsat_indices))]
                else:
                    chosen_unsat_idx = unsat_indices[rng.randrange(len(unsat_indices))]
                
                clause = clause_data[chosen_unsat_idx]
                
                # Update weight of this clause
                if clause_weight_update:
                    weights[chosen_unsat_idx] *= 1.15
                
                # Choose which variable to flip
                if use_min_conflicts:
                    # Min-conflicts: pick the variable in the clause that gives best gain
                    best_var = -1
                    best_gain = -float('inf')
                    noise_std = 0.3  # small noise to escape local optima
                    
                    for var, pos in clause:
                        gain = compute_flip_gain(assignment, var, chosen_unsat_idx)
                        # Add Gaussian noise
                        gain += rng.gauss(0, noise_std)
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_var = var
                    
                    if best_var >= 0:
                        assignment[best_var] = not assignment[best_var]
                    else:
                        var, pos = rng.choice(clause)
                        assignment[var] = not assignment[var]
                else:
                    # Original WalkSAT: pick random literal from clause
                    var, pos = rng.choice(clause)
                    assignment[var] = not assignment[var]
                
                # Check progress
                current_unsat = count_unsat(assignment)
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment[:]
                
                if current_unsat == 0:
                    return assignment[:]
            
            # End of this restart; reinitialize
            assignment = [rng.random() < 0.5 for _ in range(n)]
            current_unsat = count_unsat(assignment)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
        
        return best_assignment
    
    # Simulated annealing strategy for Max-SAT formulation
    def simulated_annealing(seed, max_iters, t0=10.0, t1=0.01):
        """
        Simulated annealing on Max-SAT objective (count satisfied clauses).
        Accepts worsening moves with probability exp(-delta/T) to escape local minima.
        """
        rng = random.Random(seed)
        
        # Initialize with random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        current_unsat = count_unsat(assignment)
        current_sat = m - current_unsat
        
        best_assignment = assignment[:]
        best_sat = current_sat
        
        # Precompute: for each variable, list of (clause_idx, position_in_clause)
        # This allows fast delta computation when flipping a variable
        var_clause_pos = [[] for _ in range(n)]
        for ci, clause in enumerate(clause_data):
            for vi, (var, pos) in enumerate(clause):
                var_clause_pos[var].append((ci, vi))
        
        # Temperature schedule: exponential cooling
        if max_iters <= 1:
            return best_assignment if best_sat == m else None
        
        cooling_rate = (t1 / t0) ** (1.0 / max_iters)
        
        for it in range(max_iters):
            if time_remaining() < 2.0:
                break
            
            # Temperature
            T = t0 * (cooling_rate ** it)
            
            # Pick a random variable to flip
            var = rng.randrange(n)
            
            # Compute delta in satisfied clauses
            # delta = new_sat - current_sat
            delta = 0
            for ci, vi in var_clause_pos[var]:
                clause = clause_data[ci]
                # Check if clause is currently satisfied
                currently_sat = False
                only_this = True
                for v, p in clause:
                    if assignment[v] == p:
                        currently_sat = True
                        if v != var:
                            only_this = False
                        break
                
                if currently_sat:
                    if only_this:
                        # Flipping var makes this clause unsatisfied
                        delta -= 1
                else:
                    # Clause is unsatisfied; check if flipping var satisfies it
                    # Find var's literal in this clause
                    for v, p in clause:
                        if v == var:
                            # After flip, assignment[var] = not assignment[var]
                            if (not assignment[var]) == p:
                                delta += 1
                            break
            
            # Accept or reject
            if delta >= 0:
                # Always accept improving or neutral moves
                assignment[var] = not assignment[var]
                current_sat += delta
            else:
                # Worsening move: accept with probability exp(delta / T)
                if T > 0:
                    prob = math.exp(delta / T)
                    if rng.random() < prob:
                        assignment[var] = not assignment[var]
                        current_sat += delta
                # else: T ~ 0, reject all worsening moves (greedy)
            
            # Track best
            if current_sat > best_sat:
                best_sat = current_sat
                best_assignment = assignment[:]
            
            if best_sat == m:
                return best_assignment
        
        return best_assignment if best_sat == m else best_assignment
    
    # Strategy 1: WalkSAT with min-conflicts and clause weighting
    best_result = None
    best_score = float('inf')
    
    seeds_to_try = [42, 123, 777, 2024, 555, 999, 31337, 12345, 67890, 24680]
    
    for seed in seeds_to_try:
        if time_remaining() < 5.0:
            break
        
        steps_per_restart = max(1500, n * 25)
        max_restarts = max(5, min(40, int(150000 / steps_per_restart)))
        
        result = walksat_restarts(seed, max_restarts, steps_per_restart, clause_weight_update=True, use_min_conflicts=True)
        
        if result is not None:
            if full_satisfied(result):
                return result
            else:
                unsat_cnt = count_unsat(result)
                if unsat_cnt < best_score:
                    best_score = unsat_cnt
                    best_result = result
    
    # Strategy 1.5: Simulated Annealing (Max-SAT formulation)
    if time_remaining() > 20.0:
        sa_seeds = [1001, 2002, 3003, 4004, 5005, 6006, 7007, 8008]
        for seed in sa_seeds:
            if time_remaining() < 5.0:
                break
            
            # Use a reasonable number of iterations based on time budget
            # Estimate: each iteration is O(degree) ~ O(m/n * 3) on average
            max_iters = max(5000, min(50000, int(1500000 / n)))
            
            result = simulated_annealing(seed, max_iters, t0=10.0, t1=0.01)
            
            if result is not None:
                if full_satisfied(result):
                    return result
                else:
                    unsat_cnt = count_unsat(result)
                    if unsat_cnt < best_score:
                        best_score = unsat_cnt
                        best_result = result
        
        # Try SA with different temperature schedules
        if time_remaining() > 10.0:
            for seed in [9009, 10010, 11011]:
                if time_remaining() < 5.0:
                    break
                
                max_iters = max(5000, min(50000, int(1500000 / n)))
                
                # Higher initial temperature for more exploration
                result = simulated_annealing(seed, max_iters, t0=20.0, t1=0.05)
                
                if result is not None:
                    if full_satisfied(result):
                        return result
                    else:
                        unsat_cnt = count_unsat(result)
                        if unsat_cnt < best_score:
                            best_score = unsat_cnt
                            best_result = result
    
    # Strategy 2: Pure WalkSAT (random flips) with more restarts
    if time_remaining() > 15.0:
        for seed in [111, 222, 333, 444, 555, 666, 777, 888, 999, 1010]:
            if time_remaining() < 5.0:
                break
            
            steps_per_restart = max(800, n * 12)
            max_restarts = max(10, min(80, int(250000 / steps_per_restart)))
            
            result = walksat_restarts(seed, max_restarts, steps_per_restart, clause_weight_update=False, use_min_conflicts=False)
            
            if result is not None:
                if full_satisfied(result):
                    return result
                else:
                    unsat_cnt = count_unsat(result)
                    if unsat_cnt < best_score:
                        best_score = unsat_cnt
                        best_result = result
    
    # Strategy 3: Hybrid - min-conflicts with no clause weighting, shorter steps
    if time_remaining() > 10.0:
        for seed in [77, 88, 99, 100, 111, 222, 333, 444]:
            if time_remaining() < 3.0:
                break
            
            steps_per_restart = max(400, n * 8)
            max_restarts = max(20, min(150, int(350000 / steps_per_restart)))
            
            result = walksat_restarts(seed, max_restarts, steps_per_restart, clause_weight_update=False, use_min_conflicts=True)
            
            if result is not None:
                if full_satisfied(result):
                    return result
                else:
                    unsat_cnt = count_unsat(result)
                    if unsat_cnt < best_score:
                        best_score = unsat_cnt
                        best_result = result
    
    # Final fallback: try one more set with very short steps
    if time_remaining() > 5.0:
        for seed in [5, 15, 25, 35, 45, 55, 65, 75]:
            if time_remaining() < 2.0:
                break
            
            steps_per_restart = max(200, n * 4)
            max_restarts = max(30, min(300, int(400000 / steps_per_restart)))
            
            result = walksat_restarts(seed, max_restarts, steps_per_restart, clause_weight_update=True, use_min_conflicts=True)
            
            if result is not None:
                if full_satisfied(result):
                    return result
                else:
                    unsat_cnt = count_unsat(result)
                    if unsat_cnt < best_score:
                        best_score = unsat_cnt
                        best_result = result
    
    # Return best found
    if best_result is not None:
        return best_result
    
    # Absolute fallback
    return [True] * n
