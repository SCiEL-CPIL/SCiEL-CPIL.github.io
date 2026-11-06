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
    
    # Build variable-to-clause index
    # var_clauses[v] = list of (clause_idx, is_pos_in_clause)
    var_clauses = [[] for _ in range(n)]
    for i, cv in enumerate(clause_vars):
        for var_idx, is_pos in cv:
            var_clauses[var_idx].append((i, is_pos))
    
    def calculate_sat_counts(assignment):
        """Calculate number of satisfied literals for each clause."""
        sat_counts = [0] * m
        for i, cv in enumerate(clause_vars):
            count = 0
            for var_idx, is_pos in cv:
                if assignment[var_idx] == is_pos:
                    count += 1
            sat_counts[i] = count
        return sat_counts
    
    def get_unsat_clauses(sat_counts):
        """Get list of indices of unsatisfied clauses."""
        return [i for i, c in enumerate(sat_counts) if c == 0]

    def walk_sat(assignment, rng, max_flips, sat_counts, unsat_clauses):
        """
        WalkSAT with incremental sat_counts update.
        Returns True if solution found, False otherwise.
        """
        flip_count = 0
        
        while flip_count < max_flips:
            if not unsat_clauses:
                return True
            
            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            cv = clause_vars[ci]
            
            # Adaptive noise: decrease over time
            # Start with high noise to explore, end with low noise to exploit
            noise = 0.5 * max(0, 1 - flip_count / max_flips)
            
            best_var = -1
            best_delta = float('inf')
            
            # Evaluate each variable in the chosen clause
            for var_idx, is_pos in cv:
                delta = 0
                old_val = assignment[var_idx]
                new_val = not old_val
                
                # Iterate through all clauses containing this variable
                for ci2, is_pos2 in var_clauses[var_idx]:
                    # Determine if the literal (var_idx, is_pos2) is currently satisfied
                    # It is satisfied if assignment[var_idx] == is_pos2
                    currently_sat = (old_val == is_pos2)
                    
                    # Determine if it will be satisfied after flip
                    # It will be satisfied if new_val == is_pos2
                    post_sat = (new_val == is_pos2)
                    
                    if currently_sat and not post_sat:
                        # Literal goes from SAT to UNSAT
                        # This reduces sat_counts[ci2] by 1
                        # If sat_counts[ci2] was 1, it becomes 0 (clause becomes UNSAT) -> Delta +1
                        # If sat_counts[ci2] was > 1, it stays SAT -> Delta 0
                        if sat_counts[ci2] == 1:
                            delta += 1
                    elif not currently_sat and post_sat:
                        # Literal goes from UNSAT to SAT
                        # This increases sat_counts[ci2] by 1
                        # If sat_counts[ci2] was 0, it becomes 1 (clause becomes SAT) -> Delta -1
                        # If sat_counts[ci2] was > 0, it stays SAT -> Delta 0
                        if sat_counts[ci2] == 0:
                            delta -= 1
                    # else: no change in satisfaction status of this literal, delta 0
                
                # Apply noise
                delta += rng.gauss(0, noise)
                
                if delta < best_delta:
                    best_delta = delta
                    best_var = var_idx
            
            if best_var < 0:
                # Fallback: random flip from the clause
                best_var = rng.choice(cv)[0]
            
            # Perform the flip
            assignment[best_var] = not assignment[best_var]
            
            # Update sat_counts and unsat_clauses incrementally
            new_val = assignment[best_var]
            for ci2, is_pos2 in var_clauses[best_var]:
                # Check if the literal status changed
                # Old status: (not new_val) == is_pos2  (since we just flipped)
                # New status: new_val == is_pos2
                
                # If new_val == is_pos2, the literal is now SAT.
                # Before flip, it was (not new_val) == is_pos2.
                # If new_val == is_pos2, then not new_val != is_pos2.
                # So it went from UNSAT to SAT.
                if new_val == is_pos2:
                    # Literal became SAT
                    sat_counts[ci2] += 1
                    # If it was 0 before, it's now 1. Clause became SAT.
                    # Remove from unsat_clauses
                    if sat_counts[ci2] == 1:
                        # It was unsat (0), now sat (1)
                        # Check if it's in unsat_clauses
                        # Since we are updating incrementally, it should be if it was 0
                        # But we need to be careful: it was 0, so it WAS in unsat_clauses
                        try:
                            unsat_clauses.remove(ci2)
                        except ValueError:
                            pass # Should not happen if logic is correct
                else:
                    # new_val != is_pos2, so literal is UNSAT.
                    # Before flip, it was (not new_val) == is_pos2.
                    # Since new_val != is_pos2, not new_val == is_pos2.
                    # So it was SAT, now UNSAT.
                    sat_counts[ci2] -= 1
                    # If it was 1 before, it's now 0. Clause became UNSAT.
                    # Add to unsat_clauses
                    if sat_counts[ci2] == 0:
                        # It was sat (1), now unsat (0)
                        unsat_clauses.append(ci2)
            
            flip_count += 1
            
            # Periodic check for solution (optional, as unsat_clauses is maintained)
            if not unsat_clauses:
                return True
        
        return len(unsat_clauses) == 0
    
    # Multiple restarts
    best_assignment = None
    best_unsat = float('inf')
    
    restart = 0
    while time.time() < deadline:
        # Each restart: random initial assignment
        # Use a different seed for each restart to ensure diversity
        rng = random.Random(42 + restart)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize sat_counts and unsat_clauses
        sat_counts = calculate_sat_counts(assignment)
        unsat_clauses = get_unsat_clauses(sat_counts)
        
        # Check if already solved
        if not unsat_clauses:
            return assignment
        
        # Max flips per restart
        # Start with a reasonable number, increase over time
        max_flips = 10000 + restart * 1000
        
        if walk_sat(assignment, rng, max_flips, sat_counts, unsat_clauses):
            return assignment
        
        # Track best
        unsat = len(unsat_clauses)
        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = assignment[:]
        
        restart += 1
        
        # Safety check: if restarts take too long, break
        if restart > 10000:
            break
    
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: all true
    return [True] * n
# EVOLVE-BLOCK-END
