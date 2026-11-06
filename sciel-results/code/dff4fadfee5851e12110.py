# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290
    
    # Preprocess
    m = len(clauses)
    
    # Convert to 0-based and build matrices
    # vars_matrix: shape (m, 3), values 0..n-1
    # signs_matrix: shape (m, 3), values 0 (negative) or 1 (positive)
    vars_matrix = np.zeros((m, 3), dtype=np.int32)
    signs_matrix = np.zeros((m, 3), dtype=np.int32)
    
    for i, c in enumerate(clauses):
        for j, l in enumerate(c):
            vars_matrix[i, j] = abs(l) - 1
            signs_matrix[i, j] = 1 if l > 0 else 0

    # Build variable index for quick access to clauses containing a variable
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            var_to_clauses[vars_matrix[i, j]].append(i)

    # Preprocess: Pure literal elimination and unit propagation
    # This can simplify the problem before search starts.
    # We will track which variables are fixed and their values.
    # If a variable is fixed, we can simplify clauses.
    
    # However, for a 150-variable problem, full simplification might be overkill or risky if not careful.
    # Let's do a light preprocessing:
    # 1. Check for empty clauses (UNSAT) - not expected for planted SAT.
    # 2. Check for tautologies (var and -var in same clause) - remove them.
    # 3. Unit propagation: If a clause has only one literal, that literal MUST be true.
    
    # We'll implement a simple unit propagation that fixes variables.
    # If we fix too many, we might get stuck. But for planted SAT, it should help.
    
    # Let's keep it simple: just remove tautologies and do a few rounds of unit propagation
    # to reduce m and n effectively.
    
    # Actually, modifying n and m complicates the mapping back to the original assignment.
    # Let's stick to the original n and m but use the preprocessing to initialize the assignment
    # with higher probability of success or to identify forced variables.
    
    # A better approach for "Preprocessing" in this context:
    # Use the structure to guide the initial assignment or the first few steps.
    
    # Let's implement a "Strong Start" using a greedy heuristic:
    # Iterate through variables. For each variable, choose the value that satisfies the most
    # currently unsatisfied clauses (or breaks the fewest satisfied ones).
    
    # But the directive says "Keep this design" (WalkSAT) and "investigate improvements within its family".
    # The inspiration mentions "Preprocessing (pure literal elimination + unit propagation) reduces the effective problem size".
    
    # Let's implement a simplified unit propagation that just sets initial values for variables
    # that appear in unit clauses (clauses with only one literal? No, clauses are size 3).
    # In 3-SAT, a "unit clause" in the context of local search often refers to a clause
    # where 2 literals are false and 1 is true (satisfied) or 3 are false (unsatisfied).
    # Standard unit propagation applies to partial assignments.
    
    # Let's try a different refinement: **Adaptive p_random** and **Better Min-Conflicts**.
    # The current min-conflicts calculation is O(degree * clause_size).
    # We can optimize the cost calculation.
    
    # Also, the inspiration suggests **Simulated Annealing**.
    # Let's combine WalkSAT with a Simulated Annealing acceptance criterion.
    # If the flip increases the unsat count, we accept it with probability exp(-delta/T).
    # This allows escaping local minima.
    
    rng = random.Random(42)
    
    def count_unsat(assignment):
        a = np.asarray(assignment, dtype=np.int32)
        matches = (a[vars_matrix] == signs_matrix)
        sat_counts = matches.sum(axis=1)
        return int(np.sum(sat_counts == 0))

    def get_unsat_clauses(assignment):
        a = np.asarray(assignment, dtype=np.int32)
        matches = (a[vars_matrix] == signs_matrix)
        sat_counts = matches.sum(axis=1)
        unsat_indices = np.where(sat_counts == 0)[0]
        return unsat_indices.tolist()

    best_assignment = None
    best_unsat_count = m + 1
    
    max_restarts = 1000
    base_steps = 1000
    step_growth = 1.1
    
    # Simulated Annealing parameters
    T_start = 5.0
    T_end = 0.1
    cooling_rate = 0.9995
    
    restart = 0
    while restart < max_restarts and time.time() < deadline:
        # Random initial assignment
        assignment = np.random.randint(0, 2, n, dtype=np.int32).tolist()
        
        unsat_count = count_unsat(assignment)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = list(assignment)
        
        steps = int(base_steps * (step_growth ** restart))
        steps = min(steps, 20000)
        
        # Adaptive p_random: start high, decrease
        p_random_start = 0.5
        p_random_end = 0.2
        T = T_start
        
        for step in range(steps):
            if time.time() >= deadline:
                break
            
            unsat_indices = get_unsat_clauses(assignment)
            
            if not unsat_indices:
                return assignment
            
            current_unsat_count = len(unsat_indices)
            if current_unsat_count < best_unsat_count:
                best_unsat_count = current_unsat_count
                best_assignment = list(assignment)
            
            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_indices)
            
            # Get the literals in this clause
            c_vars = vars_matrix[ci]
            c_signs = signs_matrix[ci]
            
            # Identify false literals in this clause
            false_vars = []
            for k in range(3):
                v = c_vars[k]
                s = c_signs[k]
                if assignment[v] != s:
                    false_vars.append((v, s))
            
            if not false_vars:
                continue
            
            # Choose which variable to flip
            # Calculate scores for min-conflicts
            best_var = -1
            min_cost = float('inf')
            
            for v, s in false_vars:
                # Calculate cost of flipping v
                cost = 0
                for ci2 in var_to_clauses[v]:
                    if ci2 == ci:
                        continue
                    
                    c2_vars = vars_matrix[ci2]
                    c2_signs = signs_matrix[ci2]
                    
                    # Check if clause ci2 is currently satisfied
                    sat_now = False
                    only_sat_var = -1
                    sat_count = 0
                    for k in range(3):
                        vv = c2_vars[k]
                        ss = c2_signs[k]
                        if assignment[vv] == ss:
                            sat_now = True
                            sat_count += 1
                            only_sat_var = vv
                            
                    if sat_now:
                        if sat_count == 1 and only_sat_var == v:
                            cost += 1
                    else:
                        # Clause is unsatisfied. Check if flipping v satisfies it.
                        for k in range(3):
                            if c2_vars[k] == v and c2_signs[k] == s:
                                cost -= 1
                                break
                                
                if cost < min_cost:
                    min_cost = cost
                    best_var = v
            
            if best_var == -1:
                var_to_flip = false_vars[0][0]
            else:
                var_to_flip = best_var
            
            # Calculate the change in unsat count if we flip var_to_flip
            # We need to know the new unsat count to apply SA.
            # But calculating it exactly is expensive.
            # Approximation: The cost calculated above is the net change in unsat count
            # for clauses other than ci. ci itself goes from unsat to sat (change -1).
            # So total change = cost - 1.
            
            delta = min_cost - 1
            
            # Simulated Annealing acceptance
            if delta <= 0:
                # Improvement or neutral, always accept
                accept = True
            else:
                # Worse state, accept with probability exp(-delta/T)
                # T cools down
                T = max(T_end, T * cooling_rate)
                if T > T_end:
                    prob = np.exp(-delta / T)
                    accept = rng.random() < prob
                else:
                    accept = False
            
            if accept:
                assignment[var_to_flip] = 1 - assignment[var_to_flip]
            else:
                # If we reject the best move, try a random move from the clause?
                # Or just skip?
                # Let's try a random move from the false_vars to maintain exploration
                var_to_flip, _ = rng.choice(false_vars)
                assignment[var_to_flip] = 1 - assignment[var_to_flip]
            
        restart += 1

    if best_assignment is not None:
        return best_assignment
    return assignment
# EVOLVE-BLOCK-END
