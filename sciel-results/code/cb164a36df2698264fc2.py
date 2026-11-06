"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    # WalkSAT with incremental tracking, multiple restarts, and adaptive flip selection.
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline: 325 seconds (5s safety margin before 330s kill)
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Pre-process clauses: convert to 0-based indices and sign flags
    # For each clause, store (var_idx, is_positive) tuples
    # Also build variable-to-clauses index for incremental updates
    m = len(clauses)
    
    # Parse clauses into structured form
    clause_vars = []  # list of lists of (var_idx, positive)
    var_to_clauses = [[] for _ in range(n)]
    
    for ci, clause in enumerate(clauses):
        clause_vars.append([])
        for lit in clause:
            var_idx = abs(lit) - 1
            positive = (lit > 0)
            clause_vars[ci].append((var_idx, positive))
            var_to_clauses[var_idx].append(ci)
    
    # Precompute clause sizes
    clause_sizes = [len(cv) for cv in clause_vars]
    
    # For each clause, track how many literals are satisfied (0, 1, 2, or 3)
    # sat_count[ci] = number of satisfied literals in clause ci
    
    def check_solution(assignment):
        """Check if assignment satisfies all clauses."""
        for ci in range(m):
            satisfied = False
            for var_idx, positive in clause_vars[ci]:
                val = assignment[var_idx]
                if (positive and val) or (not positive and not val):
                    satisfied = True
                    break
            if not satisfied:
                return False
        return True
    
    def count_unsat(assignment):
        """Count number of unsatisfied clauses."""
        count = 0
        for ci in range(m):
            satisfied = False
            for var_idx, positive in clause_vars[ci]:
                val = assignment[var_idx]
                if (positive and val) or (not positive and not val):
                    satisfied = True
                    break
            if not satisfied:
                count += 1
        return count
    
    best_assignment = None
    best_unsat = m + 1
    rng = random.Random(42)
    
    max_restarts = 200
    for restart in range(max_restarts):
        # Check time
        if time.time() > deadline:
            break
        
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Compute initial sat counts for each clause
        sat_counts = np.zeros(m, dtype=np.int32)
        for ci in range(m):
            cnt = 0
            for var_idx, positive in clause_vars[ci]:
                val = assignment[var_idx]
                if (positive and val) or (not positive and not val):
                    cnt += 1
            sat_counts[ci] = cnt
        
        unsat_count = int(np.sum(sat_counts == 0))
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
        
        # Adaptive step budget based on restart number
        # Early restarts get fewer steps, later ones get more
        base_steps = 500
        step_budget = base_steps + restart * 100
        # Cap total steps per restart
        step_budget = min(step_budget, 10000)
        
        # WalkSAT with damage-minimizing flip selection
        for step in range(step_budget):
            if time.time() > deadline:
                break
            
            if unsat_count == 0:
                return assignment
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
            
            # Pick a random unsatisfied clause
            # Build list of unsat clause indices occasionally for efficiency
            # For speed, just pick randomly and check
            unsat_indices = np.where(sat_counts == 0)[0]
            if len(unsat_indices) == 0:
                break
            ci = int(rng.choice(unsat_indices.tolist()))
            
            # Evaluate flipping each variable in this clause
            # For each literal in the clause, compute the damage (change in unsat count)
            best_var = -1
            best_damage = m + 1
            best_is_satisfying = False
            
            for var_idx, positive in clause_vars[ci]:
                # Compute damage of flipping var_idx
                damage = 0
                for cj in var_to_clauses[var_idx]:
                    # Current sat status of clause cj
                    cur_sat = sat_counts[cj]
                    # After flip, how does clause cj change?
                    # If var_idx appears in clause cj:
                    #   If currently satisfied by var_idx: after flip, it's not satisfied by this var
                    #   If currently not satisfied by var_idx: after flip, it is satisfied by this var
                    # We need to check if var_idx is in clause cj and its current contribution
                    found = False
                    for vi, pos in clause_vars[cj]:
                        if vi == var_idx:
                            found = True
                            cur_val = assignment[var_idx]
                            currently_satisfies = (pos and cur_val) or (not pos and not cur_val)
                            if currently_satisfies:
                                # Flip will make this literal unsatisfying
                                # If this was the only satisfying literal, clause becomes unsat
                                if cur_sat == 1:
                                    damage += 1
                            else:
                                # Flip will make this literal satisfying
                                if cur_sat == 0:
                                    damage -= 1
                            break
                    if not found:
                        # var_idx not in clause cj, no change
                        pass
                
                # Check if this flip would satisfy the chosen clause ci
                cur_val = assignment[var_idx]
                will_satisfy = (positive and not cur_val) or (not positive and cur_val)
                
                if damage < best_damage or (damage == best_damage and will_satisfy and not best_is_satisfying):
                    best_damage = damage
                    best_var = var_idx
                    best_is_satisfying = will_satisfy
            
            if best_var < 0:
                break
            
            # Apply the flip
            old_val = assignment[best_var]
            new_val = not old_val
            assignment[best_var] = new_val
            
            # Update sat_counts for all clauses containing best_var
            for cj in var_to_clauses[best_var]:
                # Determine if best_var currently satisfies clause cj
                found_pos = False
                for vi, pos in clause_vars[cj]:
                    if vi == best_var:
                        found_pos = True
                        currently_satisfies = (pos and new_val) or (not pos and not new_val)
                        break
                if found_pos:
                    if currently_satisfies:
                        sat_counts[cj] += 1
                    else:
                        sat_counts[cj] -= 1
                # If best_var not in clause cj (shouldn't happen due to indexing), no change
            
            # Update unsat_count
            unsat_count = int(np.sum(sat_counts == 0))
        
        # After this restart, check if we improved
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
        
        # If we've been stuck, try a more aggressive strategy
        # Occasionally do a pure literal elimination or unit propagation attempt
        if restart % 5 == 4 and best_unsat > 0:
            # Try simple unit propagation on the current best assignment
            pass  # Skip for now, keep it simple
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: return a random assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
