import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290
    
    # Pre-compute variable to clauses mapping
    var_to_clauses = [[] for _ in range(n)]
    # clause_literal_map[c] = {var_idx: literal_value} is not strictly needed if we can find the literal sign quickly
    # But for speed, let's store for each clause, the sign of each variable? 
    # Or just iterate. Since clauses are size 3, iteration is cheap.
    
    # To speed up true_count updates, we need to know the sign of the literal for variable v in clause c.
    # Let's precompute: clause_var_sign[c][v] = 1 if positive, -1 if negative, 0 if not present.
    # Since n=50, we can use a list of dicts or a 2D array.
    # A 2D array [n_clauses][n_vars] might be memory heavy if n_clauses is large, but n=50 is small.
    # Max clauses for 3-SAT with 50 vars? Usually 4.25 * n = ~212.
    # So [212][50] is tiny.
    
    n_clauses = len(clauses)
    # clause_var_sign[c][v] = sign of literal v in clause c, or 0
    clause_var_sign = [[0] * n for _ in range(n_clauses)]
    
    for i, clause in enumerate(clauses):
        for l in clause:
            v = abs(l) - 1
            var_to_clauses[v].append(i)
            clause_var_sign[i][v] = 1 if l > 0 else -1

    rng = random.Random(0)
    best_assignment = None
    best_unsat = float('inf')
    
    # WalkSAT Parameters
    p_noise = 0.2
    steps_per_restart = 20000
    max_restarts = 2000
    
    # Activity tracking
    activity = [1.0] * n
    decay = 0.99
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize true_count and unsat structures
        true_count = [0] * n_clauses
        is_unsat = [False] * n_clauses
        unsat_list = []
        
        for i, c in enumerate(clauses):
            count = 0
            for l in c:
                v = abs(l) - 1
                if (l > 0 and assignment[v]) or (l < 0 and not assignment[v]):
                    count += 1
            true_count[i] = count
            if count == 0:
                is_unsat[i] = True
                unsat_list.append(i)
                
        cur_unsat = len(unsat_list)
        if cur_unsat == 0:
            return assignment
            
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
            
        # Local search
        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
                break
                
            if not unsat_list:
                return assignment
                
            # Pick a random unsat clause
            # O(1) selection
            c_idx = unsat_list[rng.randrange(len(unsat_list))]
            clause = clauses[c_idx]
            
            # Evaluate flip for each literal in the clause
            best_var = None
            best_score = float('inf')
            best_act = -1.0
            
            for l in clause:
                var_idx = abs(l) - 1
                sign = 1 if l > 0 else -1
                
                # Calculate score: number of clauses that will become unsat (broken)
                # We flip var_idx.
                # For each clause c' containing var_idx:
                #   Let s' be the sign of var_idx in c'.
                #   Current truth of literal in c': 
                #     if s' == 1: truth = assignment[var_idx]
                #     if s' == -1: truth = not assignment[var_idx]
                #   New truth of literal in c':
                #     if s' == 1: truth = not assignment[var_idx]
                #     if s' == -1: truth = assignment[var_idx]
                
                #   Change in true_count[c']:
                #   If current truth was True, new is False -> true_count[c'] -= 1
                #   If current truth was False, new is True -> true_count[c'] += 1
                
                #   Satisfaction status change:
                #   If true_count[c'] was 1 and becomes 0 -> Broken
                #   If true_count[c'] was 0 and becomes 1 -> Fixed
                #   Else -> No change in status
                
                broken = 0
                # We don't strictly need 'fixed' for the score if we stick to current logic, 
                # but minimizing broken is the goal.
                
                for other_c_idx in var_to_clauses[var_idx]:
                    if other_c_idx == c_idx:
                        continue
                    
                    # Get sign of var_idx in other_c_idx
                    s_other = clause_var_sign[other_c_idx][var_idx]
                    
                    # Determine current truth of this literal
                    # Current value of var_idx is assignment[var_idx]
                    # If s_other == 1, literal is var. Truth = assignment[var_idx]
                    # If s_other == -1, literal is ~var. Truth = not assignment[var_idx]
                    
                    if s_other == 1:
                        current_truth = assignment[var_idx]
                    else:
                        current_truth = not assignment[var_idx]
                    
                    # The new truth will be the opposite
                    new_truth = not current_truth
                    
                    tc = true_count[other_c_idx]
                    
                    if current_truth:
                        # Literal was True, becomes False
                        # true_count decreases by 1
                        if tc == 1:
                            # Was satisfied (only this literal true), becomes unsatisfied
                            broken += 1
                    else:
                        # Literal was False, becomes True
                        # true_count increases by 1
                        # If tc was 0, it becomes 1 (Fixed), but we don't count fixed in score
                        # If tc > 0, it stays satisfied.
                        pass
                    
                    if broken > best_score:
                        # Early exit if already worse than best
                        break

                if broken < best_score or (broken == best_score and activity[var_idx] > best_act):
                    best_score = broken
                    best_var = var_idx
                    best_act = activity[var_idx]
                    
            # Decide flip
            if rng.random() < p_noise:
                l = rng.choice(clause)
                var_idx = abs(l) - 1
            else:
                var_idx = best_var
                
            if var_idx is None:
                continue
                
            # Perform flip
            old_val = assignment[var_idx]
            assignment[var_idx] = not old_val
            
            # Update activity
            for l in clause:
                v = abs(l) - 1
                activity[v] *= decay
            activity[var_idx] += 1.0
            
            # Update true_count and unsat structures
            for c_idx in var_to_clauses[var_idx]:
                s = clause_var_sign[c_idx][var_idx]
                
                # Determine if the literal's truth value changed from True to False or False to True
                # Old truth:
                if s == 1:
                    old_truth = old_val
                else:
                    old_truth = not old_val
                
                tc = true_count[c_idx]
                
                if old_truth:
                    # Was True, now False
                    true_count[c_idx] = tc - 1
                    if tc == 1:
                        # Became unsatisfied
                        is_unsat[c_idx] = True
                        unsat_list.append(c_idx)
                else:
                    # Was False, now True
                    true_count[c_idx] = tc + 1
                    if tc == 0:
                        # Became satisfied
                        is_unsat[c_idx] = False
                        # Remove from unsat_list efficiently
                        # Find index in unsat_list
                        # Since we append to end, we can swap with last
                        # But we need to find the index.
                        # To make this O(1) amortized, we can keep an index map?
                        # Or just linear search since unsat_list size is usually small?
                        # Or, we can just let unsat_list have duplicates and clean it up?
                        # No, duplicates break the "random choice" logic if we assume uniqueness.
                        # Let's use a linear search for removal. 
                        # In 3-SAT, unsat count is usually small compared to total.
                        # If performance is an issue, we can use a different structure.
                        # For now, linear search.
                        try:
                            idx = unsat_list.index(c_idx)
                            unsat_list[idx] = unsat_list[-1]
                            unsat_list.pop()
                        except ValueError:
                            # Should not happen if is_unsat is consistent
                            pass

        # End local search loop
        if time.time() > deadline - 5:
            break

    if best_assignment:
        return best_assignment
    return [False] * n
