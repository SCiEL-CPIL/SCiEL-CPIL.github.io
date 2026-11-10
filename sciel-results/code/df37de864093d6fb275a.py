import time
import random
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Pre-process clauses into 0-based indices and signs
    # literal is 1-based signed. 
    # var = abs(l) - 1
    # sign = 1 if l > 0 else 0 (True for positive, False for negative)
    # Literal l is satisfied if assignment[var] == (l > 0)
    
    pre_clauses = []
    for c in clauses:
        pc = []
        for l in c:
            var = abs(l) - 1
            is_pos = 1 if l > 0 else 0
            pc.append((var, is_pos))
        pre_clauses.append(pc)
        
    m = len(pre_clauses)
    
    # Build variable-to-clause index for faster gain calculation
    # var_clause_map[var] = list of (clause_idx, is_pos)
    var_clause_map = [[] for _ in range(n)]
    for idx, c in enumerate(pre_clauses):
        for var, is_pos in c:
            var_clause_map[var].append((idx, is_pos))
            
    def count_unsat(assignment):
        count = 0
        for c in pre_clauses:
            sat = False
            for var, is_pos in c:
                if assignment[var] == (is_pos == 1):
                    sat = True
                    break
            if not sat:
                count += 1
        return count
        
    def is_satisfied(assignment):
        for c in pre_clauses:
            sat = False
            for var, is_pos in c:
                if assignment[var] == (is_pos == 1):
                    sat = True
                    break
            if not sat:
                return False
        return True

    def fast_walksat(assignment, max_flips, deadline, p_greedy=0.5):
        """
        Optimized WalkSAT with incremental unsat count and precomputed var-clause map.
        """
        # Maintain list of unsatisfied clause indices
        unsat_indices = []
        for i, c in enumerate(pre_clauses):
            sat = False
            for var, is_pos in c:
                if assignment[var] == (is_pos == 1):
                    sat = True
                    break
            if not sat:
                unsat_indices.append(i)
                
        current_unsat = len(unsat_indices)
        
        # Track satisfaction count for each clause to quickly check if it's the "only" true literal
        # sat_count[i] = number of true literals in clause i
        sat_count = np.zeros(m, dtype=int)
        for i, c in enumerate(pre_clauses):
            cnt = 0
            for var, is_pos in c:
                if assignment[var] == (is_pos == 1):
                    cnt += 1
            sat_count[i] = cnt
            if cnt == 0:
                if i not in unsat_indices:
                    unsat_indices.append(i)
            else:
                if i in unsat_indices:
                    unsat_indices.remove(i)
                    
        # Re-initialize unsat_indices correctly from scratch to be safe
        unsat_indices = [i for i in range(m) if sat_count[i] == 0]
        current_unsat = len(unsat_indices)
        
        for _ in range(max_flips):
            if time.time() > deadline:
                break
                
            if current_unsat == 0:
                return assignment, 0
            
            # Pick a random unsatisfied clause
            c_idx = random.choice(unsat_indices)
            c = pre_clauses[c_idx]
            
            # Calculate gain for each variable in the clause
            # Gain = (# unsat clauses this var satisfies) - (# sat clauses this var breaks)
            # A clause is broken if it is currently sat and this var is the ONLY true literal.
            
            best_gain = -100
            best_vars = []
            
            for var, is_pos in c:
                gain = 0
                # Check clauses containing this var
                for u_idx, u_is_pos in var_clause_map[var]:
                    # If u_is_pos != is_pos, flipping var doesn't change the truth value of this literal in this clause
                    # Wait, flipping var changes assignment[var].
                    # Literal (var, is_pos) is true if assignment[var] == (is_pos==1).
                    # If we flip var, assignment[var] becomes not assignment[var].
                    # So literal (var, is_pos) becomes true if (not assignment[var]) == (is_pos==1).
                    # Since we are considering flipping a var in an unsat clause, currently assignment[var] != (is_pos==1) for the specific literal in c?
                    # Not necessarily, the clause is unsat, so ALL literals in it are false.
                    # So for the specific literal in c, it is currently false. Flipping it makes it true.
                    
                    # For any other clause u containing var:
                    # Current state of literal (var, u_is_pos) in u:
                    # true if assignment[var] == (u_is_pos==1)
                    # New state after flip:
                    # true if (not assignment[var]) == (u_is_pos==1)
                    
                    # If u_is_pos == is_pos:
                    #   Current: false (since var is in unsat clause c with same polarity? No, var might appear with different polarity in u)
                    #   Actually, we just need to check current truth value.
                    
                    curr_true = (assignment[var] == (u_is_pos == 1))
                    new_true = (not assignment[var] == (u_is_pos == 1))
                    
                    if not curr_true and new_true:
                        # Literal becomes true.
                        # If clause u was unsat (sat_count[u] == 0), it becomes sat. Gain +1.
                        if sat_count[u_idx] == 0:
                            gain += 1
                    elif curr_true and not new_true:
                        # Literal becomes false.
                        # If clause u was sat and this was the ONLY true literal (sat_count[u] == 1), it becomes unsat. Gain -1.
                        if sat_count[u_idx] == 1:
                            gain -= 1
                            
                if gain > best_gain:
                    best_gain = gain
                    best_vars = [var]
                elif gain == best_gain:
                    best_vars.append(var)
                    
            # Choose flip
            if random.random() < p_greedy and best_vars:
                var_to_flip = random.choice(best_vars)
            else:
                var_to_flip = random.choice([v for v, _ in c])
                
            # Flip variable
            old_val = assignment[var_to_flip]
            assignment[var_to_flip] = not old_val
            
            # Update sat_counts and unsat_indices
            for u_idx, u_is_pos in var_clause_map[var_to_flip]:
                # Determine if truth value of literal changed
                # It always changes when we flip the variable
                old_true = (old_val == (u_is_pos == 1))
                new_true = (not old_val == (u_is_pos == 1))
                
                if old_true and not new_true:
                    # Lost a true literal
                    sat_count[u_idx] -= 1
                    if sat_count[u_idx] == 0:
                        # Became unsat
                        if u_idx not in unsat_indices:
                            unsat_indices.append(u_idx)
                        current_unsat += 1
                elif not old_true and new_true:
                    # Gained a true literal
                    sat_count[u_idx] += 1
                    if sat_count[u_idx] == 1:
                        # Became sat (was 0)
                        if u_idx in unsat_indices:
                            unsat_indices.remove(u_idx)
                        current_unsat -= 1
                            
        return assignment, current_unsat

    # Main Loop
    deadline = time.time() + 280
    best_assignment = None
    best_unsat = float('inf')
    
    # Strategy 1: Multiple random restarts with adaptive WalkSAT
    # Use a mix of greedy and random flips
    # Increase restarts if time permits
    
    max_restarts = 100
    flips_per_restart = 2000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        assignment = np.random.rand(n) < 0.5
        
        # Slightly bias p_greedy based on restart count? 
        # Early restarts: more greedy. Later: more random to escape local minima.
        p_greedy = 0.7 - (0.4 * restart / max_restarts)
        
        assignment, unsat = fast_walksat(assignment, flips_per_restart, deadline, p_greedy)
        
        if unsat == 0:
            return assignment.tolist()
            
        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = assignment.copy()
            
    # If we haven't solved it, try a "polish" on the best assignment found
    # with a different strategy (e.g., min-conflicts style or higher randomness)
    if best_assignment is not None and time.time() < deadline - 10:
        # Min-conflicts heuristic: flip variable in most unsat clauses
        assignment = best_assignment.copy()
        
        for _ in range(5000):
            if time.time() > deadline:
                break
            
            unsat_clauses = []
            for i, c in enumerate(pre_clauses):
                sat = False
                for var, is_pos in c:
                    if assignment[var] == (is_pos == 1):
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(i)
            
            if not unsat_clauses:
                return assignment.tolist()
            
            # Count how many unsat clauses each variable appears in
            var_count = [0] * n
            for c_idx in unsat_clauses:
                c = pre_clauses[c_idx]
                for var, is_pos in c:
                    var_count[var] += 1
            
            max_count = max(var_count)
            if max_count == 0:
                break
                
            candidates = [i for i in range(n) if var_count[i] == max_count]
            var_to_flip = random.choice(candidates)
            assignment[var_to_flip] = not assignment[var_to_flip]
            
        # Final check
        if is_satisfied(assignment):
            return assignment.tolist()
            
        if count_unsat(assignment) < best_unsat:
            best_unsat = count_unsat(assignment)
            best_assignment = assignment.copy()

    if best_assignment is not None:
        return best_assignment.tolist()
    return [False] * n
