import time
import random
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Pre-process clauses for fast access
    # Convert to 0-based indices for variables
    # lit: -n .. n
    # var_idx: 0 .. n-1
    # We will store clauses as lists of signed 1-based literals for compatibility with standard logic,
    # but access variables via abs(lit)-1.
    
    # To speed up WalkSAT, we can pre-compute for each variable, which clauses contain it.
    # vars_in_clause[c_idx] = list of vars in clause c
    # clauses_for_var[v] = list of clause indices containing var v
    
    vars_in_clause = []
    clauses_for_var = [[] for _ in range(n)]
    
    for c_idx, c in enumerate(clauses):
        vs = []
        for l in c:
            v = abs(l) - 1
            vs.append(v)
            clauses_for_var[v].append(c_idx)
        vars_in_clause.append(vs)
        
    # Function to check if a clause is satisfied
    # values: list of 0/1 for variables
    def is_clause_satisfied(c_idx, values):
        for l in clauses[c_idx]:
            v = abs(l) - 1
            if (l > 0 and values[v] == 1) or (l < 0 and values[v] == 0):
                return True
        return False
        
    # Function to count unsatisfied clauses
    def count_unsat(values):
        count = 0
        for c_idx in range(len(clauses)):
            if not is_clause_satisfied(c_idx, values):
                count += 1
        return count

    # WalkSAT Solver
    # Parameters:
    # p: probability of random flip (vs greedy)
    # max_steps: max flips per restart
    # max_restarts: max restarts
    
    rng = random.Random(42)
    
    best_score = float('inf')
    best_assignment = None
    
    # We will try multiple restarts.
    # Each restart has a time budget.
    # If we find a solution (0 unsat), return immediately.
    
    # Initial random assignment
    # We can also try to start with a greedy initialization? 
    # Random is standard for WalkSAT.
    
    num_restarts = 0
    while True:
        if time.time() > deadline - 5.0: # 5s safety margin for final check/return
            break
            
        # Initialize random assignment
        values = [rng.randint(0, 1) for _ in range(n)]
        
        # Check if already satisfied
        if count_unsat(values) == 0:
            return values
            
        # Walk loop
        # To avoid infinite loops in one restart, we limit steps based on time or fixed count.
        # A fixed count like 1000 * n is often good, but time-based is safer.
        # Let's use a time slice per restart.
        restart_start = time.time()
        restart_time_limit = min(10.0, (deadline - time.time()) / 2) # At least some time, max 10s per restart initially
        
        # If we have little time left, use a small slice
        if time.time() > deadline - 20.0:
            restart_time_limit = 5.0
            
        max_steps = 10000 # Hard cap on steps per restart to prevent logic errors/hangs
        
        for step in range(max_steps):
            if time.time() > restart_start + restart_time_limit:
                break
            if time.time() > deadline:
                return best_assignment if best_assignment else values
                
            # Find unsatisfied clauses
            unsat_clauses = []
            for c_idx in range(len(clauses)):
                if not is_clause_satisfied(c_idx, values):
                    unsat_clauses.append(c_idx)
                    
            if not unsat_clauses:
                return values
                
            # Score each variable flip
            # For each variable, calculate:
            # gain = (unsat clauses satisfied by flip) - (sat clauses broken by flip)
            # We only care about variables in unsat clauses? 
            # Flipping a variable NOT in an unsat clause will only break sat clauses (or keep them sat if redundant, but usually breaks).
            # So we restrict candidate variables to those appearing in unsat clauses.
            
            candidate_vars = set()
            for c_idx in unsat_clauses:
                for v in vars_in_clause[c_idx]:
                    candidate_vars.add(v)
                    
            if not candidate_vars:
                break # Should not happen if unsat_clauses is not empty
                
            best_var = -1
            best_gain = -1000
            
            # Heuristic: Weighted random selection
            # Calculate score for each candidate
            scores = []
            for v in candidate_vars:
                # Calculate gain if we flip v
                gain = 0
                for c_idx in clauses_for_var[v]:
                    is_sat_before = is_clause_satisfied(c_idx, values)
                    # Simulate flip
                    # If v is flipped, does c_idx change satisfaction status?
                    # A clause's status changes only if the flipped variable was the *only* satisfying literal (if it was sat) 
                    # or is the *only* unsatisfied literal (if it was unsat).
                    
                    # Efficient check:
                    # Count true literals in c_idx under current values.
                    # If is_sat_before:
                    #   If count_true == 1 and the true literal is v, flipping v makes it unsat. (Loss)
                    # Else:
                    #   If count_true == 0 and flipping v makes it true, it becomes sat. (Gain)
                    
                    # This is O(len(c)) per clause. 
                    # For 3-SAT, len(c)=3. Very fast.
                    
                    true_count = 0
                    v_is_true = False
                    for l in clauses[c_idx]:
                        vi = abs(l) - 1
                        if (l > 0 and values[vi] == 1) or (l < 0 and values[vi] == 0):
                            true_count += 1
                            if vi == v:
                                v_is_true = True
                                
                    if is_sat_before:
                        # Clause is sat.
                        # If v is the ONLY reason it's sat, flipping v breaks it.
                        if true_count == 1 and v_is_true:
                            gain -= 1
                    else:
                        # Clause is unsat.
                        # If flipping v makes it true, it becomes sat.
                        # Flipping v makes it true if v is currently false in the clause's perspective?
                        # i.e., if (l > 0 and values[v]==0) or (l < 0 and values[v]==1)
                        # This is equivalent to: v is NOT true in the clause currently.
                        if true_count == 0:
                            # Check if flipping v satisfies this clause
                            # We need to check if the literal corresponding to v in this clause is currently false.
                            # Since true_count is 0, all literals are false.
                            # So flipping v will make the literal for v true.
                            gain += 1
                
                scores.append((v, gain))
                
            # Select variable
            # Use a weighted probability based on gain? 
            # Or pick best gain with some randomness?
            # Standard WalkSAT: 
            # 1. Pick a random unsat clause.
            # 2. Pick a variable in that clause.
            # 3. Flip it with prob p, otherwise flip the one with max gain.
            
            # Let's implement the standard heuristic for better performance:
            # 1. Pick random unsat clause c*
            c_star = rng.choice(unsat_clauses)
            
            # 2. Pick random variable from c_star
            v_star = rng.choice(vars_in_clause[c_star])
            
            # 3. Determine flip
            if rng.random() < 0.5: # p=0.5 is common
                flip_var = v_star
            else:
                # Find variable in c_star with max gain (among all variables? or just c_star?)
                # Standard is among all variables, but restricting to c_star is faster and often effective.
                # Let's use the scores we calculated for candidate_vars.
                # We need the score for v_star.
                # If v_star is not in candidate_vars (it must be, as it's in an unsat clause), find its score.
                
                # Re-calculate max gain for variables in c_star specifically? 
                # We already calculated scores for all candidate_vars.
                max_gain_val = -1000
                best_v_in_clause = v_star
                for v, g in scores:
                    if v in vars_in_clause[c_star]:
                        if g > max_gain_val:
                            max_gain_val = g
                            best_v_in_clause = v
                flip_var = best_v_in_clause
                
            # Flip
            values[flip_var] = 1 - values[flip_var]
            
            # Update best if improved (optional, usually just return on 0)
            
        # End of restart steps
        # If we didn't solve, check time and restart
        num_restarts += 1
        if num_restarts > 100: # Safety cap on restarts
             break

    # If we ran out of time, return the current best guess (last values)
    return values
