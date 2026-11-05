import random
import time
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Preprocess
    # var_info[var_idx] = list of (clause_idx, sign, [(other_var_idx, other_sign), ...])
    # var_idx is 0-based
    var_info = [[] for _ in range(n)]
    
    for c_idx, clause in enumerate(clauses):
        literals = clause
        for l in literals:
            var_idx = abs(l) - 1
            sign = 1 if l > 0 else -1
            others = []
            for l2 in literals:
                if abs(l2) - 1 != var_idx:
                    others.append((abs(l2) - 1, 1 if l2 > 0 else -1))
            var_info[var_idx].append((c_idx, sign, others))
            
    # Helper to calculate current unsat count
    def get_unsat_count(assignment):
        count = 0
        for clause in clauses:
            sat = False
            for l in clause:
                v = abs(l) - 1
                expected = l > 0
                if assignment[v] == expected:
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def calc_delta(var_idx, assignment):
        """
        Returns the change in the number of UNSATISFIED clauses 
        if we flip var_idx.
        Negative delta means improvement (fewer unsat).
        """
        delta = 0
        for c_idx, sign, others in var_info[var_idx]:
            # Check if clause c_idx is currently satisfied
            # We need to check all 3 literals
            # Literal for var_idx:
            val_v = assignment[var_idx]
            lit_v_sat = (val_v == (sign > 0))
            
            # Literals for others:
            others_sat = False
            for v_idx, o_sign in others:
                val_o = assignment[v_idx]
                if val_o == (o_sign > 0):
                    others_sat = True
                    break
            
            if not (lit_v_sat or others_sat):
                # Clause is currently UNSAT
                # Flipping var_idx will make lit_v_sat True (since it was False)
                # So clause becomes SAT
                delta -= 1
            else:
                # Clause is currently SAT
                # If we flip var_idx:
                # If lit_v_sat was True:
                #   If others_sat was False: Clause becomes UNSAT (delta += 1)
                #   If others_sat was True: Clause stays SAT (delta += 0)
                # If lit_v_sat was False:
                #   Flipping makes it True. Clause stays SAT (delta += 0)
                
                if lit_v_sat:
                    if not others_sat:
                        delta += 1
                # else: no change
                
        return delta

    rng = random.Random(42)
    assignment = [rng.random() < 0.5 for _ in range(n)]
    current_unsat = get_unsat_count(assignment)
    best_unsat = current_unsat
    best_assignment = assignment[:]
    
    if current_unsat == 0:
        return assignment

    start_time = time.time()
    time_limit = 325.0  # Leave 5s margin for 330s limit
    
    # SA Parameters
    T = 1.0
    cooling_rate = 0.999
    min_T = 1e-4
    
    # We will restart if T drops too low or if we are stuck
    # But first, let's just run SA.
    
    # To pick a variable, we pick a random unsat clause and a random var from it.
    # We need a fast way to get unsat clauses.
    # Recalculating unsat clauses every step is O(M*3). M ~ 200. This is fine.
    
    while time.time() - start_time < time_limit:
        # Find unsat clauses
        unsat_clauses = []
        for i, clause in enumerate(clauses):
            sat = False
            for l in clause:
                v = abs(l) - 1
                if assignment[v] == (l > 0):
                    sat = True
                    break
            if not sat:
                unsat_clauses.append(clause)
                
        if not unsat_clauses:
            return assignment
            
        # Pick a random unsat clause
        clause = rng.choice(unsat_clauses)
        # Pick a random variable from it
        l = rng.choice(clause)
        var_idx = abs(l) - 1
        
        delta = calc_delta(var_idx, assignment)
        
        # Acceptance probability
        if delta < 0:
            accept = True
        else:
            # delta >= 0
            if T > 0:
                prob = math.exp(-delta / T)
            else:
                prob = 0
            accept = rng.random() < prob
            
        if accept:
            assignment[var_idx] = not assignment[var_idx]
            current_unsat += delta
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment[:]
                if current_unsat == 0:
                    return best_assignment
                    
        T *= cooling_rate
        if T < min_T:
            # Restart with a new random assignment? 
            # Or just keep going with T=min_T?
            # Restarting is better for escaping local minima.
            # But we must preserve the best found so far.
            # Let's restart the search state, but keep best_assignment.
            assignment = [rng.random() < 0.5 for _ in range(n)]
            current_unsat = get_unsat_count(assignment)
            T = 1.0
            
    return best_assignment
