"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    # WalkSAT with restarts, cost-based variable selection, and a time budget.
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    rng = random.Random(42)

    # Pre-compute variable to clause indices mapping for cost calculation
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for lit in clause:
            var_idx = abs(lit) - 1
            var_to_clauses[var_idx].append(i)

    # Pre-compute literal sign and variable index for faster access
    # lit_vars[i] = (var_index, is_positive)
    lit_vars = []
    for clause in clauses:
        lit_vars.append([(abs(l) - 1, l > 0) for l in clause])

    max_time = 300.0
    start_time = time.time()
    
    best_assignment = [False] * n
    best_unsat_count = m  # Worst case is all clauses unsat

    def count_unsat(assignment):
        count = 0
        for i in range(m):
            sat = False
            for var_idx, is_pos in lit_vars[i]:
                if assignment[var_idx] == is_pos:
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def find_unsat_clauses(assignment):
        unsat_indices = []
        for i in range(m):
            sat = False
            for var_idx, is_pos in lit_vars[i]:
                if assignment[var_idx] == is_pos:
                    sat = True
                    break
            if not sat:
                unsat_indices.append(i)
        return unsat_indices

    def calculate_cost(var_idx, assignment):
        """
        Calculate the change in unsatisfied clause count if we flip var_idx.
        Cost = (New Unsat Count) - (Old Unsat Count)
        We want to minimize this.
        """
        cost = 0
        current_val = assignment[var_idx]
        
        for c_idx in var_to_clauses[var_idx]:
            clause = lit_vars[c_idx]
            
            # Check if the literal corresponding to var_idx is satisfied currently
            # Find the position of var_idx in this clause
            x_sat = False
            other_sat_count = 0
            other_unsat_count = 0
            
            for var_idx_c, is_pos_c in clause:
                if var_idx_c == var_idx:
                    # This is the literal for our variable
                    if assignment[var_idx] == is_pos_c:
                        x_sat = True
                else:
                    if assignment[var_idx_c] == is_pos_c:
                        other_sat_count += 1
                    else:
                        other_unsat_count += 1
            
            if x_sat:
                # Variable is currently satisfying this literal.
                # If we flip it, it becomes unsatisfying.
                # The clause becomes unsatisfied ONLY IF no other literal is satisfied.
                if other_sat_count == 0:
                    cost += 1
            else:
                # Variable is currently NOT satisfying this literal.
                # If we flip it, it becomes satisfying.
                # The clause becomes satisfied ONLY IF it was previously unsatisfied.
                # It was previously unsatisfied ONLY IF no other literal was satisfied.
                if other_sat_count == 0:
                    cost -= 1
                    
        return cost

    # Parameters
    p_walk = 0.5  # Probability to walk (choose random var in clause) vs walk (choose best var)
    # Actually, standard WalkSAT uses a heuristic. Let's use:
    # 1. Find unsat clauses.
    # 2. Pick one at random.
    # 3. Pick a variable in that clause.
    #    With prob p, pick randomly.
    #    With prob 1-p, pick the one with min cost (flip that reduces unsat count most).
    p_random = 0.5
    
    max_steps_per_restart = 10000
    restart_counter = 0

    while True:
        elapsed = time.time() - start_time
        if elapsed > max_time - 5:
            break

        # Restart
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        current_unsat = count_unsat(assignment)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
            
        if current_unsat == 0:
            return assignment

        # WalkSAT loop
        for step in range(max_steps_per_restart):
            if time.time() - start_time > max_time - 5:
                break

            unsat_indices = find_unsat_clauses(assignment)
            
            if not unsat_indices:
                return assignment

            current_unsat = len(unsat_indices)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]

            # Pick a random unsatisfied clause
            c_idx = rng.choice(unsat_indices)
            clause = lit_vars[c_idx]
            
            # Pick a variable to flip
            if rng.random() < p_random:
                # Random selection
                var_idx, is_pos = rng.choice(clause)
            else:
                # Heuristic selection: min cost
                min_cost = float('inf')
                best_var = None
                for var_idx, is_pos in clause:
                    cost = calculate_cost(var_idx, assignment)
                    if cost < min_cost:
                        min_cost = cost
                        best_var = var_idx
                var_idx = best_var
            
            # Flip the variable
            assignment[var_idx] = not assignment[var_idx]

    return best_assignment
# EVOLVE-BLOCK-END
