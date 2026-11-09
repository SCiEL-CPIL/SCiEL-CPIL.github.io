import time
import random
import sys

def solve(instance):
    """
    Solves a planted 3-SAT instance using an optimized WalkSAT algorithm
    with gain heuristics and restarts.
    """
    n = instance["n"]
    clauses_input = instance["clauses"]
    m = len(clauses_input)
    
    # Deadline guard: 300s limit, leave 10s margin
    start_time = time.time()
    deadline = start_time + 290.0

    # Preprocess clauses
    # Convert to 0-based variables and signs (1 for positive, -1 for negative)
    # clause_data[i] = (var0, sign0, var1, sign1, var2, sign2)
    clause_data = []
    var_to_clauses = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses_input):
        c_data = []
        for lit in clause:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            c_data.append((var, sign))
            var_to_clauses[var].append(i)
        clause_data.append(tuple(c_data))

    # State
    # assignment[v] is True or False
    # clause_sat_count[i] is the number of satisfied literals in clause i
    
    def init_state():
        assignment = [random.random() < 0.5 for _ in range(n)]
        clause_sat_count = [0] * m
        for i, c in enumerate(clause_data):
            count = 0
            for v, s in c:
                if assignment[v] == (s == 1):
                    count += 1
            clause_sat_count[i] = count
        return assignment, clause_sat_count

    def is_satisfied(clause_sat_count):
        return all(c > 0 for c in clause_sat_count)

    def walksat_restart(max_steps):
        assignment, clause_sat_count = init_state()
        
        # Initial check
        if is_satisfied(clause_sat_count):
            return assignment

        # Precompute clause indices for variables for faster gain calculation
        # Actually, var_to_clauses is already built.
        
        for step in range(max_steps):
            # Find an unsatisfied clause
            # Randomly pick an unsatisfied clause
            unsat_clauses = [i for i in range(m) if clause_sat_count[i] == 0]
            if not unsat_clauses:
                return assignment
            
            c_idx = random.choice(unsat_clauses)
            clause = clause_data[c_idx]
            
            # Calculate gain for each variable in the clause
            # Gain = (num unsat clauses containing var that will become sat) - 
            #        (num sat clauses containing var that will become unsat)
            
            best_var = -1
            best_gain = -1000
            best_loss = 1000
            
            # To avoid recalculating everything from scratch, we can just evaluate the 3 variables in the clause
            # But standard WalkSAT often picks the variable with max gain among all variables, or just within the clause.
            # Given n=150 and m is likely ~150-200 (planted 3-SAT is usually near threshold), 
            # evaluating all variables is O(n*m) per step which might be slow.
            # Evaluating only variables in the clause is O(3 * degree(var)).
            # Let's try the "min-conflicts" or "gain" heuristic restricted to the clause first, 
            # or full gain if it's fast enough.
            
            # With n=150, m~200, degree ~ 4. 
            # Checking gain for a variable v:
            # Iterate var_to_clauses[v] (size ~4-5).
            # For each clause, check if flipping v changes its status.
            
            # Let's implement a function to calculate gain/loss for a specific variable flip
            def calc_gain_loss(var, current_assignment):
                # current_assignment is the list before flip
                # We want to know:
                # 1. How many currently UNSAT clauses become SAT? (Gain)
                # 2. How many currently SAT clauses become UNSAT? (Loss)
                
                gain = 0
                loss = 0
                new_val = not current_assignment[var]
                
                for c_idx in var_to_clauses[var]:
                    c = clause_data[c_idx]
                    # Check if this clause contains the literal (var, sign)
                    # It does, because it's in var_to_clauses[var]
                    # Find the sign of var in this clause
                    v_sign = None
                    for v, s in c:
                        if v == var:
                            v_sign = s
                            break
                    if v_sign is None: continue # Should not happen
                    
                    # Current status of this clause
                    # We need to know if the literal (var, v_sign) is currently satisfied
                    # and if the clause is currently satisfied.
                    
                    # It's easier to just recount the clause satisfaction if we don't have 
                    # a fast way to check "is this specific literal the only one keeping it sat".
                    # However, we have clause_sat_count.
                    
                    # If clause_sat_count[c_idx] == 0, it is UNSAT.
                    # If we flip var, and the literal (var, v_sign) becomes satisfied, 
                    # then this clause becomes SAT (since it was 0, now at least 1).
                    # So gain += 1 if (new_val == (v_sign == 1)) is True.
                    
                    # If clause_sat_count[c_idx] > 0, it is SAT.
                    # If we flip var, and the literal (var, v_sign) was satisfied and becomes unsatisfied,
                    # we need to check if it was the ONLY satisfied literal.
                    # If clause_sat_count[c_idx] == 1, and the literal was sat, it becomes UNSAT.
                    # So loss += 1 if (current_assignment[var] == (v_sign == 1)) and clause_sat_count[c_idx] == 1.
                    
                    if clause_sat_count[c_idx] == 0:
                        if new_val == (v_sign == 1):
                            gain += 1
                    else:
                        if current_assignment[var] == (v_sign == 1):
                            if clause_sat_count[c_idx] == 1:
                                loss += 1
                return gain, loss

            # Evaluate variables in the chosen clause
            candidates = []
            for v, s in clause:
                g, l = calc_gain_loss(v, assignment)
                # We want to maximize gain and minimize loss
                # Score = gain - loss
                score = g - l
                candidates.append((score, v, g, l))
            
            # Pick the best candidate
            # To add some randomness, we can pick from top few or use a noise term
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Add noise to avoid getting stuck in deterministic loops
            # If best score is negative or 0, maybe pick randomly?
            # Standard WalkSAT: pick best, or pick randomly with probability p.
            # Let's stick to picking the best gain, but if multiple have same gain, pick randomly.
            
            max_score = candidates[0][0]
            best_candidates = [c for c in candidates if c[0] == max_score]
            chosen = random.choice(best_candidates)
            var_to_flip = chosen[1]
            
            # Perform the flip
            old_val = assignment[var_to_flip]
            new_val = not old_val
            assignment[var_to_flip] = new_val
            
            # Update clause_sat_count
            for c_idx in var_to_clauses[var_to_flip]:
                c = clause_data[c_idx]
                v_sign = None
                for v, s in c:
                    if v == var_to_flip:
                        v_sign = s
                        break
                if v_sign is None: continue
                
                # Did the literal status change? Yes, always.
                # Was it satisfied before?
                was_sat = old_val == (v_sign == 1)
                # Is it satisfied now?
                is_sat = new_val == (v_sign == 1)
                
                if was_sat and not is_sat:
                    clause_sat_count[c_idx] -= 1
                elif not was_sat and is_sat:
                    clause_sat_count[c_idx] += 1
            
            # Check if solved
            # We can check if min(clause_sat_count) > 0
            # But that's O(m). 
            # We can maintain a count of unsat clauses.
            # Let's just check periodically or if we suspect we are close.
            # Actually, with m~200, checking all is fast.
            if all(c > 0 for c in clause_sat_count):
                return assignment
            
            # Time check
            if time.time() > deadline:
                return assignment # Return best effort (current state)

        return assignment # Return best effort

    # Main loop with restarts
    best_assignment = None
    restart_count = 0
    max_steps_per_restart = 1000
    
    while time.time() < deadline:
        assignment = walksat_restart(max_steps_per_restart)
        
        # Verify if the returned assignment is actually a solution
        # walksat_restart returns early if solved, but might return unsat state if time runs out
        # We need to verify.
        
        # Fast verification
        is_sol = True
        for i, c in enumerate(clause_data):
            if not any(assignment[v] == (s == 1) for v, s in c):
                is_sol = False
                break
        
        if is_sol:
            return assignment
        
        # If not solved, we can try to save the "best" unsat assignment?
        # The problem asks for a satisfying assignment. If we don't find one, we return whatever.
        # Usually, returning the last attempt is fine.
        
        restart_count += 1
        # Increase steps slightly? Or keep fixed?
        # Fixed is safer for time management.
        
    # If we run out of time, return the last attempt
    # We need to ensure we return something.
    # The last assignment from the last walksat_restart call is in 'assignment' variable 
    # only if we scope it correctly. 
    # Let's restructure to keep the last attempt.
    
    # Re-run structure to capture last attempt
    last_attempt = None
    start_time = time.time()
    deadline = start_time + 290.0
    
    while time.time() < deadline:
        assignment = walksat_restart(max_steps_per_restart)
        last_attempt = assignment
        
        # Verify
        is_sol = True
        for i, c in enumerate(clause_data):
            if not any(assignment[v] == (s == 1) for v, s in c):
                is_sol = False
                break
        
        if is_sol:
            return assignment
            
    return last_attempt

# EVOLVE-BLOCK-START
# The code above is the complete implementation.
# EVOLVE-BLOCK-END
