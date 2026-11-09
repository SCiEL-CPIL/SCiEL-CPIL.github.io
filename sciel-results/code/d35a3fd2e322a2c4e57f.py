"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard: 280 seconds to be safe
    deadline = time.time() + 280
    
    # Precompute: for each variable, list of (clause_idx, sign)
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_to_clauses[var_idx].append((ci, sign))
    
    m = len(clauses)
    
    # Parameters
    base_noise = 0.4
    max_steps_per_restart = 50000
    restarts = 0
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    def is_lit_true(var_idx, sign, assignment):
        """Check if a literal is true given the assignment."""
        val = assignment[var_idx]
        return val if sign == 1 else (not val)
    
    def compute_clause_sat_counts(assignment):
        """Compute how many literals are true in each clause."""
        sat_counts = [0] * m
        for ci, clause in enumerate(clauses):
            count = 0
            for lit in clause:
                var_idx = abs(lit) - 1
                sign = 1 if lit > 0 else -1
                if is_lit_true(var_idx, sign, assignment):
                    count += 1
            sat_counts[ci] = count
        return sat_counts

    def get_unsat_clauses(sat_counts):
        return [i for i, sc in enumerate(sat_counts) if sc == 0]

    while time.time() < deadline:
        restarts += 1
        
        # Phase Saving: If we have a best assignment, bias the new random assignment towards it
        if best_assignment is not None and random.random() < 0.5:
            # Copy best assignment with some noise
            assignment = best_assignment[:]
            for i in range(n):
                if random.random() < 0.1: # 10% flip rate from best
                    assignment[i] = not assignment[i]
        else:
            # Pure random
            assignment = [random.random() < 0.5 for _ in range(n)]
        
        # Calculate initial sat counts and unsat count
        sat_counts = compute_clause_sat_counts(assignment)
        unsat_clauses = get_unsat_clauses(sat_counts)
        unsat_count = len(unsat_clauses)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        
        # Adaptive noise: Start higher, decrease if making progress
        # If unsat_count is low, we are close, so use less noise (exploitation)
        # If unsat_count is high, use more noise (exploration)
        noise = base_noise
        if unsat_count < 5:
            noise = 0.1
        elif unsat_count < 10:
            noise = 0.2
        else:
            noise = base_noise

        # WalkSAT loop
        steps = 0
        while steps < max_steps_per_restart and time.time() < deadline:
            if not unsat_clauses:
                return assignment
            
            # Pick a random unsatisfied clause
            # Weighted selection? For now, random.
            unsat_clause_idx = random.choice(unsat_clauses)
            unsat_clause = clauses[unsat_clause_idx]
            
            # Heuristic: Score each literal in the unsatisfied clause
            # Score = Gain - Damage
            best_score = float('-inf')
            best_lit = unsat_clause[0]
            
            for lit in unsat_clause:
                var_idx = abs(lit) - 1
                sign = 1 if lit > 0 else -1
                
                gain = 0
                damage = 0
                
                for (ci, s) in var_to_clauses[var_idx]:
                    if ci == unsat_clause_idx:
                        continue # Ignore the current clause
                    
                    # Check current status of clause ci
                    if sat_counts[ci] == 0:
                        # Clause is currently unsatisfied.
                        # If we flip var_idx, the literal (var_idx, s) changes value.
                        # Currently, since ci is unsatisfied, all its literals are False.
                        # So is_lit_true(var_idx, s, assignment) is False.
                        # After flip, it becomes True.
                        # So ci becomes satisfied. This is GAIN.
                        gain += 1
                    else:
                        # Clause is currently satisfied.
                        # Check if the literal (var_idx, s) is currently True.
                        lit_true = is_lit_true(var_idx, s, assignment)
                        if lit_true:
                            # The literal is currently True.
                            # If we flip var_idx, it becomes False.
                            # If this was the ONLY true literal in ci (sat_counts[ci] == 1),
                            # then ci becomes unsatisfied. This is DAMAGE.
                            if sat_counts[ci] == 1:
                                damage += 1
                        else:
                            # Literal is currently False.
                            # Flipping makes it True.
                            # ci stays satisfied (or becomes more satisfied). No damage.
                            pass
                
                score = gain - damage
                
                if score > best_score:
                    best_score = score
                    best_lit = lit
            
            # Decide flip
            if random.random() < noise:
                lit_to_flip = random.choice(unsat_clause)
            else:
                lit_to_flip = best_lit
            
            var_idx = abs(lit_to_flip) - 1
            
            # Update assignment
            assignment[var_idx] = not assignment[var_idx]
            
            # Update sat_counts incrementally
            # We need to update all clauses containing var_idx
            for (ci, s) in var_to_clauses[var_idx]:
                # Determine old truth value of the literal in clause ci
                # Before flip, assignment[var_idx] was the opposite of current.
                # So if current assignment[var_idx] is True, old was False.
                # Let's just recompute the count for this clause. It's O(3) which is fast.
                new_count = 0
                for l in clauses[ci]:
                    v = abs(l) - 1
                    sg = 1 if l > 0 else -1
                    if is_lit_true(v, sg, assignment):
                        new_count += 1
                sat_counts[ci] = new_count
                
            # Update unsat_clauses list
            # This is O(m) if we rebuild, which is too slow if m is large.
            # But m is typically ~250 for 50 vars. Rebuilding is 250 ops.
            # Steps are 50000. 50000 * 250 = 12.5M ops. Acceptable.
            # However, we can be smarter.
            # Only clauses containing var_idx changed.
            # We can update the unsat_clauses list by checking only affected clauses.
            # But maintaining a sorted list or set is complex.
            # Given m is small, rebuilding is simpler and likely fast enough in Python if optimized.
            # Let's try to rebuild only if steps % 10 == 0 or similar? No, we need accurate unsat_clauses for selection.
            
            # Optimization: Instead of rebuilding the whole list, we can just count.
            # But we need to pick a random one.
            # Let's just rebuild.
            unsat_clauses = get_unsat_clauses(sat_counts)
            unsat_count = len(unsat_clauses)
            
            if unsat_count == 0:
                return assignment
            
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]
            
            steps += 1
            
            # Adaptive noise adjustment
            if steps % 100 == 0:
                if unsat_count < 5:
                    noise = 0.05
                elif unsat_count < 10:
                    noise = 0.15
                else:
                    noise = base_noise
        
        # End of restart steps
        if time.time() > deadline - 5:
            break
            
    return best_assignment
# EVOLVE-BLOCK-END
