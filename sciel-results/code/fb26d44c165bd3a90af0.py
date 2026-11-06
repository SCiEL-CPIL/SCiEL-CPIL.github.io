"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Deadline: 330s wall-clock, use 325s for safety
    deadline = time.time() + 325.0
    
    # Preprocess: convert clauses to 0-based indices with sign
    # clause_vars[i] = list of (var_idx, is_positive)
    clause_vars = []
    for c in clauses:
        cv = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            cv.append((var_idx, is_pos))
        clause_vars.append(cv)
    
    # Build variable-to-clause index: for each var, list of (clause_idx, is_positive)
    var_clauses = [[] for _ in range(n)]
    for i, cv in enumerate(clause_vars):
        for var_idx, is_pos in cv:
            var_clauses[var_idx].append((i, is_pos))
    
    def init_clause_state(assignment):
        """Compute sat_count for each clause and unsat set."""
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
    
    def walk_sat(assignment, rng, max_flips, deadline):
        """Incremental WalkSAT with clause state tracking."""
        sat_counts, unsat_clauses = init_clause_state(assignment)
        unsat_count = len(unsat_clauses)
        if unsat_count == 0:
            return True
        
        flip_count = 0
        while flip_count < max_flips:
            if time.time() > deadline:
                break
            
            if not unsat_clauses:
                return True
            
            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            cv = clause_vars[ci]
            
            # Adaptive noise: decrease over time
            noise = 0.3 * max(0, 1 - flip_count / max_flips)
            
            # For each variable in the unsatisfied clause, compute delta
            best_var = -1
            best_delta = float('inf')
            
            for var_idx, is_pos in cv:
                delta = 0
                for ci2, is_pos2 in var_clauses[var_idx]:
                    old_sat = sat_counts[ci2] > 0
                    # After flip: does this clause remain satisfied?
                    # The contribution of var_idx changes
                    old_contrib = (assignment[var_idx] == is_pos2)
                    new_contrib = (not assignment[var_idx]) == is_pos2
                    
                    if old_sat:
                        # Was satisfied. If var was the only satisfying literal, becomes unsat
                        if old_contrib and not new_contrib:
                            # Check if any other literal satisfies
                            other_sat = False
                            for v2, p2 in clause_vars[ci2]:
                                if v2 != var_idx and assignment[v2] == p2:
                                    other_sat = True
                                    break
                            if not other_sat:
                                delta += 1
                    else:
                        # Was unsatisfied. If flip makes this literal satisfy, becomes sat
                        if not old_contrib and new_contrib:
                            delta -= 1
                
                # Add noise for exploration
                if noise > 0:
                    delta += rng.gauss(0, noise)
                
                if delta < best_delta:
                    best_delta = delta
                    best_var = var_idx
            
            if best_var < 0:
                best_var = cv[0][0]
            
            # Apply flip and update state incrementally
            old_val = assignment[best_var]
            assignment[best_var] = not old_val
            
            for ci2, is_pos2 in var_clauses[best_var]:
                old_contrib = (old_val == is_pos2)
                new_contrib = (not old_val) == is_pos2
                old_sat = sat_counts[ci2] > 0
                
                if old_contrib and not new_contrib:
                    # This literal was satisfying, now it's not
                    sat_counts[ci2] -= 1
                    if sat_counts[ci2] == 0:
                        # Became unsatisfied
                        if ci2 not in unsat_clauses:
                            unsat_clauses.append(ci2)
                elif not old_contrib and new_contrib:
                    # This literal was not satisfying, now it is
                    sat_counts[ci2] += 1
                    if sat_counts[ci2] == 1:
                        # Became satisfied (was unsat before since sat_count was 0)
                        if ci2 in unsat_clauses:
                            unsat_clauses.remove(ci2)
                
                # Note: if both old_contrib and new_contrib are False or both True,
                # sat_count doesn't change (can't happen for a single flip of one literal)
            
            flip_count += 1
            
            # Periodic check for solution
            if flip_count % 50 == 0 and not unsat_clauses:
                return True
        
        return len(unsat_clauses) == 0
    
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
    
    # Multiple restarts with different seeds
    best_assignment = None
    best_unsat = float('inf')
    
    restart = 0
    while time.time() < deadline:
        # Each restart: random initial assignment
        rng = random.Random(42 + restart * 7919)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Max flips per restart: start moderate, increase over time
        max_flips = 10000 + restart * 2000
        if restart > 20:
            max_flips = 50000
        
        if walk_sat(assignment, rng, max_flips, deadline):
            return assignment
        
        unsat = count_unsat(assignment)
        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = assignment[:]
        
        restart += 1
    
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: all true
    return [True] * n
# EVOLVE-BLOCK-END
