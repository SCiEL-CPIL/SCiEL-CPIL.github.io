"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Deadline: 330s wall-clock, use 325s for safety
    start_time = time.time()
    deadline = start_time + 325.0

    # Preprocess: convert clauses to 0-based indices
    # clause_vars[i] = list of (var_idx, is_positive)
    clause_vars = []
    for c in clauses:
        cv = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            cv.append((var_idx, is_pos))
        clause_vars.append(cv)
    
    # Build variable-to-clause index
    # var_clauses[var_idx] = list of (clause_idx, is_positive_in_clause)
    var_clauses = [[] for _ in range(n)]
    for i, cv in enumerate(clause_vars):
        for var_idx, is_pos in cv:
            var_clauses[var_idx].append((i, is_pos))

    def check_solution(assignment):
        """Verify if assignment satisfies all clauses."""
        for cv in clause_vars:
            sat = False
            for var_idx, is_pos in cv:
                if assignment[var_idx] == is_pos:
                    sat = True
                    break
            if not sat:
                return False
        return True

    def walk_sat(assignment, rng, max_flips, deadline):
        """
        Optimized WalkSAT with incremental clause tracking.
        """
        # Initialize clause satisfaction counts
        # clause_sat_count[i] = number of literals in clause i that are satisfied
        clause_sat_count = [0] * m
        unsat_clauses = []
        
        for i, cv in enumerate(clause_vars):
            count = 0
            for var_idx, is_pos in cv:
                if assignment[var_idx] == is_pos:
                    count += 1
            clause_sat_count[i] = count
            if count == 0:
                unsat_clauses.append(i)
        
        if not unsat_clauses:
            return True

        flip_count = 0
        # Adaptive noise parameters
        base_noise = 0.5
        min_noise = 0.1
        
        while flip_count < max_flips:
            if not unsat_clauses:
                return True
            
            # Check time periodically
            if time.time() > deadline:
                break

            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            cv = clause_vars[ci]
            
            # Calculate delta for each variable in the clause
            best_var = -1
            best_delta = float('inf')
            
            # Current noise level
            progress = flip_count / max_flips
            noise = min_noise + (base_noise - min_noise) * (1 - progress)
            
            for var_idx, is_pos in cv:
                # Calculate delta for flipping var_idx
                # Delta = (new_unsat_count - old_unsat_count)
                # We are currently in an unsatisfied clause ci.
                # Flipping var_idx will satisfy this literal.
                # Since clause ci is unsatisfied (count=0), flipping this literal makes count=1, so ci becomes satisfied.
                # This reduces unsat count by 1.
                delta = -1
                
                # Check other clauses containing var_idx
                for ci2, is_pos2 in var_clauses[var_idx]:
                    if ci2 == ci:
                        continue
                    
                    cur_count = clause_sat_count[ci2]
                    
                    # If this clause is currently satisfied (cur_count > 0)
                    if cur_count > 0:
                        # Check if flipping var_idx makes it unsatisfied.
                        # It becomes unsatisfied only if:
                        # 1. The literal in this clause is currently satisfied (contributes to cur_count)
                        # 2. AND it is the ONLY satisfied literal (cur_count == 1)
                        
                        # Is the literal currently satisfied?
                        if assignment[var_idx] == is_pos2:
                            if cur_count == 1:
                                # Flipping it will make count 0, so clause becomes unsatisfied
                                delta += 1
                    # If cur_count == 0, flipping a literal in it might satisfy it, reducing delta
                    # But we only care about the net change in unsat count.
                    # If cur_count == 0, and we flip a literal that satisfies it, it becomes sat (delta -1).
                    # However, we are evaluating the flip of var_idx.
                    # If assignment[var_idx] != is_pos2, flipping it makes it satisfied.
                    # So if cur_count == 0 and assignment[var_idx] != is_pos2, delta -= 1.
                    else:
                        if assignment[var_idx] != is_pos2:
                            delta -= 1

                # Apply noise
                delta += rng.gauss(0, noise)
                
                if delta < best_delta:
                    best_delta = delta
                    best_var = var_idx
            
            if best_var < 0:
                # Fallback: random flip from the clause
                best_var = rng.choice([v for v, _ in cv])[0]
            
            # Perform the flip
            old_val = assignment[best_var]
            assignment[best_var] = not old_val
            new_val = assignment[best_var]
            
            # Update clause_sat_count and unsat_clauses incrementally
            # Only clauses containing best_var are affected
            for ci2, is_pos2 in var_clauses[best_var]:
                old_count = clause_sat_count[ci2]
                
                # Determine if the literal was satisfied before
                was_sat = (old_val == is_pos2)
                # Determine if the literal is satisfied now
                is_sat = (new_val == is_pos2)
                
                new_count = old_count
                if was_sat and not is_sat:
                    new_count -= 1
                elif not was_sat and is_sat:
                    new_count += 1
                
                clause_sat_count[ci2] = new_count
                
                # Update unsat_clauses list
                # This is O(m) in worst case if we use a list, but we can optimize.
                # For now, we maintain the list but check membership carefully.
                # A better way: maintain a boolean array is_unsat and a list.
                # Let's use a set for O(1) lookup and removal, or just filter.
                # Given n=150, m is likely in the hundreds or low thousands.
                # Filtering unsat_clauses every flip is O(m).
                # Let's try to be smarter:
                
                if old_count == 0 and new_count > 0:
                    # Clause became satisfied, remove from unsat_clauses
                    try:
                        unsat_clauses.remove(ci2)
                    except ValueError:
                        pass
                elif old_count > 0 and new_count == 0:
                    # Clause became unsatisfied, add to unsat_clauses
                    unsat_clauses.append(ci2)
            
            flip_count += 1

        return len(unsat_clauses) == 0

    # Multiple restarts with different seeds
    best_assignment = None
    best_unsat = float('inf')
    
    restart = 0
    # Estimate max flips per restart to allow ~50 restarts in 325s
    # If one restart takes ~5s, we can do ~60 restarts.
    # Let's start with a conservative max_flips and increase it if time permits.
    base_max_flips = 2000
    
    while time.time() < deadline:
        # Each restart: random initial assignment
        rng = random.Random(42 + restart)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Increase max_flips slightly with restart to allow deeper search later
        max_flips = base_max_flips + restart * 200
        
        # Run WalkSAT
        if walk_sat(assignment, rng, max_flips, deadline):
            # Verify solution before returning
            if check_solution(assignment):
                return assignment
        
        # Update best solution
        # Count unsat clauses for current assignment
        unsat_count = 0
        for i, cv in enumerate(clause_vars):
            sat = False
            for var_idx, is_pos in cv:
                if assignment[var_idx] == is_pos:
                    sat = True
                    break
            if not sat:
                unsat_count += 1
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
        
        restart += 1
        
        # If we are running out of time, do one final long run with the best assignment found?
        # Or just stop. The current logic returns best_assignment at the end.
        
        # Safety break if we've done too many restarts
        if restart > 1000:
            break

    if best_assignment is not None:
        return best_assignment
    
    # Fallback: all true
    return [True] * n
# EVOLVE-BLOCK-END
