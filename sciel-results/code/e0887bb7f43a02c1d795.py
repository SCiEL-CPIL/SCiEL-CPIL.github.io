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
    
    # Incremental tracking: sat_count[i] = number of satisfied literals in clause i
    # A clause is satisfied iff sat_count[i] > 0
    def compute_sat_counts(assignment):
        """Compute satisfaction count for each clause."""
        sat_counts = [0] * m
        for i, cv in enumerate(clause_vars):
            for var_idx, is_pos in cv:
                if assignment[var_idx] == is_pos:
                    sat_counts[i] += 1
        return sat_counts
    
    def get_unsat_clauses(sat_counts):
        """Get list of unsatisfied clause indices."""
        return [i for i in range(m) if sat_counts[i] == 0]
    
    def flip_and_update(assignment, sat_counts, var_idx):
        """Flip variable var_idx and update sat_counts incrementally."""
        old_val = assignment[var_idx]
        new_val = not old_val
        assignment[var_idx] = new_val
        
        for ci, is_pos in var_clauses[var_idx]:
            if old_val == is_pos:
                # Was satisfying, now not
                sat_counts[ci] -= 1
            else:
                # Was not satisfying, now is
                sat_counts[ci] += 1
    
    def walk_sat(assignment, rng, max_flips):
        """WalkSAT with incremental clause tracking and adaptive noise."""
        sat_counts = compute_sat_counts(assignment)
        unsat_clauses = get_unsat_clauses(sat_counts)
        
        if not unsat_clauses:
            return True
        
        flip_count = 0
        # Adaptive noise schedule
        noise_start = 0.35
        noise_end = 0.15
        
        while flip_count < max_flips:
            if not unsat_clauses:
                return True
            
            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            cv = clause_vars[ci]
            
            # Adaptive noise: decrease over time
            progress = flip_count / max_flips
            noise = noise_start + (noise_end - noise_start) * progress
            
            # For each variable in the clause, compute delta in unsat count
            best_var = -1
            best_delta = float('inf')
            
            for var_idx, is_pos in cv:
                delta = 0
                for ci2, is_pos2 in var_clauses[var_idx]:
                    if sat_counts[ci2] == 0:
                        # Clause is currently unsatisfied
                        if not assignment[var_idx] == is_pos2:
                            # Flipping would make this literal satisfy
                            delta -= 1
                    else:
                        # Clause is currently satisfied
                        if assignment[var_idx] == is_pos2 and sat_counts[ci2] == 1:
                            # This variable is the only one satisfying; flipping breaks it
                            delta += 1
                
                # Apply noise
                delta += rng.gauss(0, noise)
                
                if delta < best_delta:
                    best_delta = delta
                    best_var = var_idx
            
            if best_var < 0:
                # Fallback: random flip from clause
                best_var = rng.choice([v for v, _ in cv])[0]
            
            # Flip and update
            flip_and_update(assignment, sat_counts, best_var)
            flip_count += 1
            
            # Update unsat_clauses list efficiently
            # Only need to check clauses affected by the flip
            if flip_count % 5 == 0:
                # Periodic full rebuild to keep list accurate
                unsat_clauses = get_unsat_clauses(sat_counts)
                if not unsat_clauses:
                    return True
            else:
                # Incremental update: check only affected clauses
                new_unsat = []
                old_unsat_set = set(unsat_clauses)
                # Check all clauses that might have changed
                for ci2, _ in var_clauses[best_var]:
                    if sat_counts[ci2] == 0:
                        new_unsat.append(ci2)
                    elif ci2 in old_unsat_set:
                        # Was unsat, now sat - remove from list
                        pass
                # Rebuild properly for correctness
                unsat_clauses = [i for i in unsat_clauses if sat_counts[i] == 0]
                for ci2, _ in var_clauses[best_var]:
                    if sat_counts[ci2] == 0 and ci2 not in set(unsat_clauses):
                        unsat_clauses.append(ci2)
                if not unsat_clauses:
                    return True
        
        return len(unsat_clauses) == 0
    
    # Multiple restarts with different seeds
    best_assignment = None
    best_unsat = float('inf')
    
    restart = 0
    while time.time() < deadline:
        # Each restart: random initial assignment
        rng = random.Random(42 + restart)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Max flips per restart
        max_flips = 10000 + restart * 1000
        
        if walk_sat(assignment, rng, max_flips):
            return assignment
        
        # Count unsat for tracking best
        sat_counts = compute_sat_counts(assignment)
        unsat = sum(1 for c in sat_counts if c == 0)
        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = assignment[:]
        
        restart += 1
    
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: all true
    return [True] * n
# EVOLVE-BLOCK-END
