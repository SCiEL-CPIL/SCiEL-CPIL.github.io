"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Deadline: 330s wall-clock, use 325s for safety
    deadline = time.time() + 325.0
    
    # Preprocess: convert clauses to 0-based indices with sign
    clause_vars = []
    for c in clauses:
        cv = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            cv.append((var_idx, is_pos))
        clause_vars.append(cv)
    
    # Build variable-to-clause index
    var_clauses = [[] for _ in range(n)]
    for i, cv in enumerate(clause_vars):
        for var_idx, is_pos in cv:
            var_clauses[var_idx].append((i, is_pos))
    
    clause_sizes = [len(cv) for cv in clause_vars]
    
    def init_clause_state(assignment):
        sat_counts = [0] * m
        unsat_clauses = []
        for i, cv in enumerate(clause_vars):
            cnt = 0
            for var_idx, is_pos in cv:
                if assignment[var_idx] == is_pos:
                    cnt += 1
            sat_counts[i] = cnt
            if cnt == 0:
                unsat_clauses.append(i)
        return sat_counts, unsat_clauses
    
    def apply_flip(assignment, var_idx, sat_counts, unsat_clauses_list, unsat_clauses_set):
        """Apply a flip and update state incrementally. Returns old value."""
        old_val = assignment[var_idx]
        assignment[var_idx] = not old_val
        
        for ci2, is_pos2 in var_clauses[var_idx]:
            old_contrib = (old_val == is_pos2)
            new_contrib = (not old_val) == is_pos2
            old_sat = sat_counts[ci2] > 0
            
            if old_contrib and not new_contrib:
                sat_counts[ci2] -= 1
                if sat_counts[ci2] == 0:
                    if ci2 not in unsat_clauses_set:
                        unsat_clauses_set.add(ci2)
                        unsat_clauses_list.append(ci2)
            elif not old_contrib and new_contrib:
                sat_counts[ci2] += 1
                if sat_counts[ci2] == 1:
                    if ci2 in unsat_clauses_set:
                        unsat_clauses_set.discard(ci2)
                        try:
                            pos = unsat_clauses_list.index(ci2)
                            unsat_clauses_list[pos] = unsat_clauses_list[-1]
                            unsat_clauses_list.pop()
                        except ValueError:
                            pass
        
        return old_val
    
    def compute_delta(assignment, var_idx, sat_counts):
        """Compute the change in unsatisfied clause count if we flip var_idx."""
        delta = 0
        for ci2, is_pos2 in var_clauses[var_idx]:
            old_sat = sat_counts[ci2] > 0
            old_contrib = (assignment[var_idx] == is_pos2)
            new_contrib = (not assignment[var_idx]) == is_pos2
            
            if old_sat:
                if old_contrib and not new_contrib:
                    other_sat = False
                    for v2, p2 in clause_vars[ci2]:
                        if v2 != var_idx and assignment[v2] == p2:
                            other_sat = True
                            break
                    if not other_sat:
                        delta += 1
            else:
                if not old_contrib and new_contrib:
                    delta -= 1
        return delta
    
    def walk_sat(assignment, rng, max_flips, deadline, use_weighted_clause=False):
        sat_counts, unsat_clauses_list = init_clause_state(assignment)
        unsat_clauses_set = set(unsat_clauses_list)
        if len(unsat_clauses_list) == 0:
            return True
        
        var_weights = [0.0] * n
        for i in range(m):
            for var_idx, is_pos in clause_vars[i]:
                var_weights[var_idx] += 1.0 / max(1, len(var_clauses[var_idx]))
        
        flip_count = 0
        best_unsat_seen = len(unsat_clauses_list)
        stagnation_counter = 0
        last_improvement_flip = 0
        
        while flip_count < max_flips:
            if time.time() > deadline:
                break
            
            if not unsat_clauses_list:
                return True
            
            current_unsat = len(unsat_clauses_list)
            if current_unsat < best_unsat_seen:
                best_unsat_seen = current_unsat
                last_improvement_flip = flip_count
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            if stagnation_counter > 2000 and flip_count > 500:
                break
            
            if use_weighted_clause and len(unsat_clauses_list) > 1:
                weights = [1.0 / max(1, clause_sizes[ci]) for ci in unsat_clauses_list]
                total_w = sum(weights)
                r = rng.random() * total_w
                cumsum = 0
                ci = unsat_clauses_list[0]
                for idx, w in enumerate(weights):
                    cumsum += w
                    if r <= cumsum:
                        ci = unsat_clauses_list[idx]
                        break
            else:
                ci = rng.choice(unsat_clauses_list)
            
            cv = clause_vars[ci]
            noise = 0.25 * max(0, 1 - flip_count / max_flips)
            
            best_var = -1
            best_delta = float('inf')
            
            for var_idx, is_pos in cv:
                delta = compute_delta(assignment, var_idx, sat_counts)
                if noise > 0:
                    delta += rng.gauss(0, noise)
                if delta < best_delta:
                    best_delta = delta
                    best_var = var_idx
            
            if best_var < 0:
                best_var = cv[0][0]
            
            apply_flip(assignment, best_var, sat_counts, unsat_clauses_list, unsat_clauses_set)
            flip_count += 1
            
            if flip_count % 32 == 0 and not unsat_clauses_list:
                return True
        
        return len(unsat_clauses_list) == 0
    
    def count_unsat(assignment):
        count = 0
        for cv in clause_vars:
            sat = False
            for var_idx, is_pos in cv:
                if assignment[var_idx] == is_pos:
                    sat = True
                    break
            if not sat:
                count += 1
        return count
    
    def polish(assignment, deadline, max_polish_flips=5000):
        sat_counts, unsat_clauses_list = init_clause_state(assignment)
        unsat_clauses_set = set(unsat_clauses_list)
        
        if not unsat_clauses_list:
            return True
        
        flip_count = 0
        
        while flip_count < max_polish_flips and time.time() < deadline:
            if not unsat_clauses_list:
                return True
            
            ci = unsat_clauses_list[rng_polish.randint(0, len(unsat_clauses_list) - 1)]
            cv = clause_vars[ci]
            
            best_var = -1
            best_delta = float('inf')
            
            for var_idx, is_pos in cv:
                delta = compute_delta(assignment, var_idx, sat_counts)
                if delta < best_delta:
                    best_delta = delta
                    best_var = var_idx
            
            if best_var < 0:
                best_var = cv[0][0]
            
            if best_delta <= 0:
                apply_flip(assignment, best_var, sat_counts, unsat_clauses_list, unsat_clauses_set)
            else:
                var_idx, is_pos = cv[rng_polish.randint(0, len(cv) - 1)]
                apply_flip(assignment, var_idx, sat_counts, unsat_clauses_list, unsat_clauses_set)
            
            flip_count += 1
        
        return len(unsat_clauses_list) == 0
    
    def stochastic_local_consistency(assignment, rng, max_propagations=50, deadline=None):
        """Stochastic arc-consistency: randomly pick variables and try to fix them
        if doing so doesn't increase unsatisfied clause count."""
        if deadline is None:
            deadline = time.time() + 10.0
        
        sat_counts, unsat_clauses_list = init_clause_state(assignment)
        unsat_clauses_set = set(unsat_clauses_list)
        num_fixed = 0
        
        for _ in range(max_propagations):
            if time.time() > deadline:
                break
            if not unsat_clauses_list:
                return assignment, num_fixed, True
            
            var_idx = rng.randint(0, n - 1)
            delta = compute_delta(assignment, var_idx, sat_counts)
            
            if delta <= 0:
                apply_flip(assignment, var_idx, sat_counts, unsat_clauses_list, unsat_clauses_set)
                num_fixed += 1
        
        return assignment, num_fixed, len(unsat_clauses_list) == 0
    
    # Multiple restarts
    best_assignment = None
    best_unsat = float('inf')
    
    restart = 0
    while time.time() < deadline:
        rng = random.Random(42 + restart * 7919)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Stochastic local consistency preprocessing
        slc_rng = random.Random(10000 + restart * 31)
        assignment, num_fixed, slc_solved = stochastic_local_consistency(
            assignment, slc_rng, max_propagations=30, deadline=deadline
        )
        
        if slc_solved:
            return assignment
        
        max_flips = 15000 + restart * 3000
        if restart > 15:
            max_flips = 60000
        
        use_weighted = restart % 3 == 2
        
        if walk_sat(assignment, rng, max_flips, deadline, use_weighted):
            return assignment
        
        global rng_polish
        rng_polish = random.Random(1000 + restart * 13)
        if polish(assignment, deadline):
            return assignment
        
        unsat = count_unsat(assignment)
        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = assignment[:]
        
        restart += 1
    
    if best_assignment is not None:
        rng_polish = random.Random(9999)
        if polish(best_assignment, deadline):
            return best_assignment
        return best_assignment
    
    return [True] * n
# EVOLVE-BLOCK-END
