"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def solve(instance):
    start_time = time.time()
    time_limit = 290.0  # Safety margin
    
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Preprocess
    clause_vars = []
    var_to_clauses = [[] for _ in range(n)]
    
    for i, c in enumerate(clauses):
        cv = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            cv.append((var_idx, is_pos))
            var_to_clauses[var_idx].append(i)
        clause_vars.append(cv)
    
    def is_satisfied(clause_idx, assignment):
        cv = clause_vars[clause_idx]
        for var_idx, is_pos in cv:
            if assignment[var_idx] == is_pos:
                return True
        return False
    
    def count_unsat(assignment):
        count = 0
        for i in range(m):
            if not is_satisfied(i, assignment):
                count += 1
        return count

    best_assignment = None
    best_unsat_count = float('inf')
    
    # Adaptive restart strategy
    max_steps_per_restart = 500
    restarts = 0
    max_restarts = 5000
    
    while time.time() - start_time < time_limit and restarts < max_restarts:
        assignment = [random.random() < 0.5 for _ in range(n)]
        
        # Build initial unsat set
        unsat_set = set()
        for i in range(m):
            if not is_satisfied(i, assignment):
                unsat_set.add(i)
        
        if not unsat_set:
            return assignment
        
        current_unsat = len(unsat_set)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        steps = 0
        while steps < max_steps_per_restart and time.time() - start_time < time_limit:
            if not unsat_set:
                return assignment
            
            # Pick random unsat clause
            idx = random.choice(list(unsat_set))
            clause = clause_vars[idx]
            
            # Pick a literal to flip
            # Heuristic: Try all 3, pick the one that minimizes damage (creates fewest new unsat clauses)
            best_var_idx = -1
            best_damage = float('inf')
            best_is_pos = False
            
            for var_idx, is_pos in clause:
                # Simulate flip
                # Current state: assignment[var_idx] != is_pos (since clause is unsat)
                # New state: assignment[var_idx] == is_pos
                
                # Damage: Number of clauses containing var_idx that become unsat.
                # A clause becomes unsat if:
                # 1. It is currently satisfied ONLY by this literal (i.e., this literal is True in current assignment? No, wait.)
                # Wait, in an unsat clause, all literals are False.
                # So assignment[var_idx] != is_pos.
                # If we flip, assignment[var_idx] becomes is_pos.
                
                # Let's check other clauses containing var_idx.
                damage = 0
                for c_idx in var_to_clauses[var_idx]:
                    if c_idx == idx:
                        continue # We already know this becomes sat
                    
                    # Check if this clause is currently satisfied
                    # And if it becomes unsatisfied after the flip.
                    
                    # A clause becomes unsatisfied if:
                    # 1. It was satisfied before.
                    # 2. The only satisfied literal in it was the one corresponding to var_idx with the OLD polarity.
                    #    Wait. The flip changes assignment[var_idx].
                    #    The literal in the clause has some polarity p_c.
                    #    Before flip: sat if assignment[var_idx] == p_c.
                    #    After flip: sat if (not assignment[var_idx]) == p_c.
                    
                    # So, the clause loses satisfaction ONLY IF:
                    # 1. It was satisfied before.
                    # 2. The literal (var_idx, p_c) was the ONLY satisfied literal.
                    # 3. The flip makes this literal false.
                    
                    # Let's evaluate:
                    # Get the literal in c_idx that corresponds to var_idx.
                    # There might be multiple? No, standard CNF usually assumes no duplicate vars in a clause, 
                    # but even if so, we check if ANY literal in the clause remains satisfied.
                    
                    # Efficient check:
                    # 1. Is c_idx currently satisfied?
                    #    If not, it stays unsat (damage 0 for this clause, it was already bad).
                    # 2. If yes, does it remain satisfied after flipping var_idx?
                    #    Check if any OTHER literal in c_idx is satisfied.
                    #    OR if the literal for var_idx in c_idx is still satisfied (impossible if flip makes it false).
                    
                    # Let's find the polarity of var_idx in c_idx.
                    p_c = None
                    for v, p in clause_vars[c_idx]:
                        if v == var_idx:
                            p_c = p
                            break
                            
                    if p_c is None:
                        continue # Should not happen if var_to_clauses is correct
                    
                    # Was it satisfied before?
                    # assignment[var_idx] == p_c
                    was_sat = (assignment[var_idx] == p_c)
                    
                    if was_sat:
                        # It was satisfied. Does it stay satisfied?
                        # After flip, assignment[var_idx] becomes not assignment[var_idx].
                        # New value: not assignment[var_idx]
                        # Is not assignment[var_idx] == p_c?
                        # Since was_sat was True, assignment[var_idx] == p_c.
                        # So not assignment[var_idx] != p_c.
                        # So this literal becomes False.
                        
                        # Check if any OTHER literal in c_idx is True.
                        remains_sat = False
                        for v, p in clause_vars[c_idx]:
                            if v == var_idx:
                                continue
                            if assignment[v] == p:
                                remains_sat = True
                                break
                            
                        if not remains_sat:
                            damage += 1
                    # If it was NOT satisfied before, it stays unsatisfied (unless the flip makes it sat? 
                    # No, if it was unsat, all literals false. 
                    # If the var_idx literal was false (assignment != p_c), and we flip, 
                    # it becomes true (assignment == p_c). 
                    # So it becomes SATISFIED. This is a gain, not damage.)
                    
                if damage < best_damage:
                    best_damage = damage
                    best_var_idx = var_idx
                    best_is_pos = is_pos
            
            if best_var_idx == -1:
                # Fallback to random
                var_idx, is_pos = random.choice(clause)
            else:
                var_idx = best_var_idx
                is_pos = best_is_pos
            
            # Perform flip
            old_val = assignment[var_idx]
            assignment[var_idx] = not old_val
            
            # Update unsat_set
            # Only clauses in var_to_clauses[var_idx] can change status.
            for c_idx in var_to_clauses[var_idx]:
                was_unsat = c_idx in unsat_set
                is_now_sat = is_satisfied(c_idx, assignment)
                is_now_unsat = not is_now_sat
                
                if was_unsat and not is_now_unsat:
                    unsat_set.remove(c_idx)
                elif not was_unsat and is_now_unsat:
                    unsat_set.add(c_idx)
            
            steps += 1
            
            # Check for solution
            if not unsat_set:
                return assignment
            
            # Track best
            current_unsat = len(unsat_set)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
        
        restarts += 1
        
    if best_assignment:
        return best_assignment
    return assignment # Fallback to last attempt

# EVOLVE-BLOCK-END
