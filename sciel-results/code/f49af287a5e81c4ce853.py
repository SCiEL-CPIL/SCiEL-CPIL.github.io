"""
Improved 3-SAT solver for planted satisfiable instances.
Uses a hybrid of WalkSAT and Simulated Annealing with efficient unsat tracking.
"""
import random
import time
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(42)
    
    start_time = time.time()
    # Safety margin: evaluator kills at 330s, we stop at 310s to be safe
    time_limit = 310.0
    
    # Preprocessing: Convert to 0-based, store as (var, sign) where sign=1 for positive, -1 for negative
    # We also precompute the literal values for quick checking
    # clause_vars[i] is a list of (var_idx, sign)
    clause_vars = []
    for c in clauses:
        cv = [(abs(l) - 1, 1 if l > 0 else -1) for l in c]
        clause_vars.append(cv)
    
    m = len(clause_vars)
    
    # Build variable-to-clauses index for fast damage calculation
    # var_to_clauses[var_idx] is a list of (clause_idx, sign_in_clause)
    var_to_clauses = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for var_idx, sign in cv:
            var_to_clauses[var_idx].append((ci, sign))
    
    def is_satisfied(ci, assignment):
        """Check if clause ci is satisfied by assignment."""
        cv = clause_vars[ci]
        for var_idx, sign in cv:
            if assignment[var_idx] == (sign == 1):
                return True
        return False

    def get_unsat_clauses(assignment):
        """Return a list of indices of unsatisfied clauses."""
        return [i for i in range(m) if not is_satisfied(i, assignment)]

    def count_unsat(assignment):
        return sum(1 for i in range(m) if not is_satisfied(i, assignment))

    def random_assignment():
        return [rng.random() < 0.5 for _ in range(n)]

    def compute_damage(var_idx, new_val, assignment):
        """
        Compute the net change in unsat count if var_idx is set to new_val.
        Returns: (damage, gain)
        damage: # satisfied clauses that become unsatisfied
        gain: # unsatisfied clauses that become satisfied
        """
        damage = 0
        gain = 0
        new_val_bool = new_val
        
        for ci, sign in var_to_clauses[var_idx]:
            # The literal (var_idx, sign) is satisfied in the new state if:
            literal_satisfied_new = (new_val_bool == (sign == 1))
            
            if literal_satisfied_new:
                # If the new value satisfies this literal, the clause will be satisfied.
                # Check if it was already satisfied.
                if not is_satisfied(ci, assignment):
                    gain += 1
            else:
                # The new value does NOT satisfy this literal.
                # Check if the clause was satisfied ONLY by this variable in the old state.
                old_val = assignment[var_idx]
                old_literal_satisfied = (old_val == (sign == 1))
                
                if old_literal_satisfied and is_satisfied(ci, assignment):
                    # This variable was satisfying the clause.
                    # Check if any other literal in the clause satisfies it.
                    cv = clause_vars[ci]
                    still_satisfied = False
                    for v, s in cv:
                        if v == var_idx:
                            continue
                        if assignment[v] == (s == 1):
                            still_satisfied = True
                            break
                    if not still_satisfied:
                        damage += 1
                        
        return damage, gain

    def hybrid_search(assignment, max_steps, temp_init, temp_final):
        """
        Hybrid WalkSAT / Simulated Annealing.
        Uses temperature to decide whether to accept 'bad' moves (increase in unsat count).
        """
        unsat_clauses = get_unsat_clauses(assignment)
        unsat_count = len(unsat_clauses)
        
        if unsat_count == 0:
            return assignment, True
        
        # Temperature schedule
        # Linear cooling from temp_init to temp_final
        temp = temp_init
        
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                return assignment, False
            
            # Update temperature
            if max_steps > 1:
                progress = step / max_steps
                temp = temp_init - (temp_init - temp_final) * progress
            else:
                temp = temp_final
            
            if not unsat_clauses:
                return assignment, True
            
            # Select a random unsatisfied clause
            # If unsat_count is small, pick from list. If large, sample.
            if unsat_count < m // 10:
                chosen_ci = rng.choice(unsat_clauses)
            else:
                # Sample until we find one, or just pick a random index and check
                # Since we have the list, picking from list is O(1) if we keep it updated, 
                # but rebuilding the list is O(m). 
                # For simplicity and speed, if unsat_count is high, we can just pick a random clause index 
                # and verify it's unsat, but we don't have a fast random-unat-picker.
                # Let's just use the list. Rebuilding the list every step is O(m).
                # To optimize, we could maintain the list, but that's complex.
                # Given m is ~450, O(m) rebuild is ~450 ops. 3000 steps * 450 ops = 1.3M ops per restart.
                # This is fine in Python if we don't do heavy work inside.
                chosen_ci = rng.choice(unsat_clauses)
            
            cv = clause_vars[chosen_ci]
            
            # Evaluate all 3 variables in the clause
            best_var = -1
            best_gain = -1
            best_damage = 0
            best_new_val = False
            
            # We want to minimize (damage - gain). 
            # Let's compute for all 3 and pick the best.
            scores = []
            for var_idx, sign in cv:
                new_val = (sign == 1)
                # If the variable already has this value, flipping it would make the literal unsatisfied.
                # But we are in an unsatisfied clause, so the current value does NOT satisfy the literal.
                # So setting it to new_val WILL satisfy the literal.
                # So gain is at least 1 (for this clause).
                # We need to check damage to other clauses.
                d, g = compute_damage(var_idx, new_val, assignment)
                # Net change in unsat = damage - gain
                # We want to minimize this.
                score = d - g
                scores.append((score, var_idx, new_val))
            
            # Sort by score
            scores.sort()
            best_score, best_var, best_new_val = scores[0]
            
            # Acceptance criterion:
            # If best_score < 0, it's a "good" move (reduces unsat count). Always accept?
            # Or use SA logic:
            # delta_unsat = best_score
            # If delta_unsat < 0, accept with prob 1 (or high).
            # If delta_unsat >= 0, accept with prob exp(-delta_unsat / temp).
            
            # Actually, let's look at the next best option too for diversity?
            # Standard WalkSAT picks the best "good" move or random.
            # SA picks based on temperature.
            
            # Let's implement a simple SA acceptance for the best move found.
            # If best_score < 0, accept.
            # If best_score >= 0, accept with probability exp(-best_score / temp).
            
            if best_score < 0:
                accept = True
            else:
                # Avoid division by zero if temp is very small
                if temp < 1e-6:
                    accept = False
                else:
                    accept = rng.random() < math.exp(-best_score / temp)
            
            if accept:
                assignment[best_var] = best_new_val
            else:
                # If we reject the best move (because it's bad and temp is low), 
                # we should probably make a random move to escape, or keep the state?
                # In SA, we keep the state. But for SAT, staying still is bad.
                # Let's make a random flip in the clause to maintain exploration.
                var_idx, sign = rng.choice(cv)
                assignment[var_idx] = (sign == 1)
            
            # Update unsat_clauses list
            # Rebuilding is O(m). 
            # Optimization: Only rebuild every K steps or if we suspect big change?
            # For now, rebuild every step to ensure correctness. 
            # m=450, 3000 steps -> 1.35M checks. ~0.1-0.2s per restart in Python? 
            # 300s / 0.2s = 1500 restarts. That's a lot.
            # Actually, Python is slow. 1.35M iterations of `is_satisfied` (which is a loop of 3) 
            # is ~4M operations. Might take 0.5s. 
            # 300s / 0.5s = 600 restarts. Still good.
            
            unsat_clauses = get_unsat_clauses(assignment)
            unsat_count = len(unsat_clauses)
            
            if unsat_count == 0:
                return assignment, True
        
        return assignment, unsat_count == 0

    # Main loop
    best_assignment = random_assignment()
    best_unsat = count_unsat(best_assignment)
    
    restart = 0
    max_restarts = 1000 # High enough to be limited by time
    
    while time.time() - start_time < time_limit:
        if restart >= max_restarts:
            break
        
        # Choose initial assignment
        if restart == 0:
            assignment = best_assignment[:]
        else:
            # 10% chance to restart from best, 90% random
            if rng.random() < 0.1:
                assignment = best_assignment[:]
            else:
                assignment = random_assignment()
        
        # Parameters for hybrid search
        # Start with higher temp to allow exploration, cool down
        temp_init = 2.0
        temp_final = 0.1
        
        # Steps per restart: 
        # If we do 3000 steps, and each step takes ~100us (optimistic), that's 0.3s.
        # 300s / 0.3s = 1000 restarts.
        # Let's use 2000 steps to be safe and allow more restarts.
        steps_per_restart = 2000
        
        assignment, solved = hybrid_search(assignment, steps_per_restart, temp_init, temp_final)
        
        if solved:
            return assignment
        
        cur_unsat = count_unsat(assignment)
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
        
        restart += 1
        
        # If we are getting close to time limit, do a final long run from best?
        if time.time() - start_time > time_limit - 30:
             # Use remaining time for one long run
             assignment = best_assignment[:]
             # Use higher temp for this final run to try to break out
             assignment, solved = hybrid_search(assignment, 10000, 1.0, 0.01)
             if solved:
                 return assignment
             cur_unsat = count_unsat(assignment)
             if cur_unsat < best_unsat:
                 best_unsat = cur_unsat
                 best_assignment = assignment[:]
             break

    return best_assignment
