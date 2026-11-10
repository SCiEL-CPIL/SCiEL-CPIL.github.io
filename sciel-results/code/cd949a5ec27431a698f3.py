# EVOLVE-BLOCK-START
import time
import random
import math


def solve(instance):
    """
    Planted 3-SAT solver using a refined WalkSAT with:
    - Deadline guard (325s safety margin)
    - Adaptive noise parameter
    - Weighted clause selection (prefer most violated clauses)
    - Variable flip heuristic: prefer flips that satisfy more clauses
    - Restarts with fresh random assignments if stuck
    """
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    deadline = time.time() + 325.0  # safety margin before 330s kill
    
    # Pre-compute clause literal indices for speed
    # clause_lits[i] = list of (var_idx, is_positive)
    clause_lits = []
    for c in clauses:
        lits = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = (l > 0)
            lits.append((var_idx, is_pos))
        clause_lits.append(lits)
    
    # Build variable-to-clause index
    var_to_clauses = [[] for _ in range(n)]
    for ci, lits in enumerate(clause_lits):
        for vi, _ in lits:
            var_to_clauses[vi].append(ci)
    
    def is_satisfied(ci, assignment):
        """Check if clause ci is satisfied under assignment."""
        for vi, is_pos in clause_lits[ci]:
            if assignment[vi] == is_pos:
                return True
        return False
    
    def count_unsatisfied(assignment):
        """Count number of unsatisfied clauses."""
        count = 0
        for ci in range(num_clauses):
            if not is_satisfied(ci, assignment):
                count += 1
        return count
    
    def evaluate_flip(vi, new_val, assignment, unsat_clauses_set):
        """
        Evaluate flipping variable vi to new_val.
        Returns the change in number of unsatisfied clauses (negative = improvement).
        """
        delta = 0
        for ci in var_to_clauses[vi]:
            # Check current satisfaction status
            currently_satisfied = False
            for (v, p) in clause_lits[ci]:
                if v == vi:
                    continue
                if assignment[v] == p:
                    currently_satisfied = True
                    break
            
            # After flip, does this clause become satisfied?
            after_satisfied = False
            for (v, p) in clause_lits[ci]:
                if v == vi:
                    if new_val == p:
                        after_satisfied = True
                        break
                else:
                    if assignment[v] == p:
                        after_satisfied = True
                        break
            
            if currently_satisfied and not after_satisfied:
                delta += 1  # was satisfied, now not
            elif not currently_satisfied and after_satisfied:
                delta -= 1  # was not satisfied, now satisfied
        
        return delta
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random assignment
        rng = random.Random(42 + restart)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initial unsatisfied count
        unsat_count = 0
        unsat_clauses = []
        for ci in range(num_clauses):
            if not is_satisfied(ci, assignment):
                unsat_count += 1
                unsat_clauses.append(ci)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        
        # WalkSAT loop
        # Adaptive noise: start low, increase if stuck
        noise = 0.2
        steps_since_improvement = 0
        max_steps_per_restart = 50000
        
        for step in range(max_steps_per_restart):
            if time.time() > deadline - 5:
                break
            
            # Check if solved
            if unsat_count == 0:
                return assignment
            
            # Track best
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
            
            # Increase noise if stuck
            if steps_since_improvement > 500:
                noise = min(0.5, noise + 0.05)
            
            # Select an unsatisfied clause (weighted: prefer clauses with more "potential")
            # Simple: pick randomly from unsatisfied, but bias toward shorter effective clauses
            ci = rng.choice(unsat_clauses)
            
            # Choose variable to flip
            # Heuristic: for each literal in the clause, compute delta if we flip
            best_delta = float('inf')
            best_var = -1
            best_val = False
            
            for (vi, is_pos) in clause_lits[ci]:
                # Flipping vi to is_pos would satisfy this literal
                new_val = is_pos
                d = evaluate_flip(vi, new_val, assignment, set(unsat_clauses))
                if d < best_delta:
                    best_delta = d
                    best_var = vi
                    best_val = new_val
            
            # With probability noise, flip a random variable from the clause
            if rng.random() < noise:
                vi, is_pos = rng.choice(clause_lits[ci])
                best_var = vi
                best_val = not assignment[vi]
            else:
                # Use the best flip found
                if best_var == -1:
                    vi, is_pos = rng.choice(clause_lits[ci])
                    best_var = vi
                    best_val = not assignment[vi]
            
            # Apply flip
            old_val = assignment[best_var]
            new_val = best_val
            if old_val == new_val:
                continue
            
            # Update unsat_count by checking affected clauses
            # Re-evaluate affected clauses
            old_unsat_set = set(unsat_clauses)
            assignment[best_var] = new_val
            
            # Recompute unsat for affected clauses
            new_unsat_count = unsat_count
            for ci_affected in var_to_clauses[best_var]:
                was_unsat = ci_affected in old_unsat_set
                is_now_satisfied = is_satisfied(ci_affected, assignment)
                if was_unsat and is_now_satisfied:
                    new_unsat_count -= 1
                elif not was_unsat and not is_now_satisfied:
                    new_unsat_count += 1
            
            # Rebuild unsat_clauses list if needed
            if new_unsat_count != unsat_count:
                unsat_clauses = [ci for ci in range(num_clauses) if not is_satisfied(ci, assignment)]
                unsat_count = new_unsat_count
            
            if unsat_count == 0:
                return assignment
        
        # If we get here, restart
        if best_assignment is not None and best_unsat_count == 0:
            return best_assignment
    
    # Return best found
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: random
    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
