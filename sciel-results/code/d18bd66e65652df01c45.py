"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    num_clauses = len(clauses)
    
    # Pre-process clauses
    # clause_vars[i] = [(var_idx, polarity), ...]
    clause_vars = []
    for c in clauses:
        clause_vars.append([(abs(l) - 1, l > 0) for l in c])
    
    # Variable to clause index mapping
    var_to_clauses = [[] for _ in range(n)]
    for i, c_data in enumerate(clause_vars):
        for v, pol in c_data:
            var_to_clauses[v].append(i)

    start_time = time.time()
    deadline = start_time + 290.0  # 330s limit, leave 40s margin

    best_assignment = None
    best_unsat_count = num_clauses + 1

    # WalkSAT Parameters
    p_walk = 0.5
    max_steps_per_restart = 1000

    while time.time() < deadline:
        # Random Initialization
        assignment = [random.random() < 0.5 for _ in range(n)]
        
        # Calculate initial unsatisfied clauses
        # unsat_bool[i] is True if clause i is unsatisfied
        unsat_bool = [False] * num_clauses
        unsat_count = 0
        
        for i, c_data in enumerate(clause_vars):
            is_sat = False
            for v, pol in c_data:
                if assignment[v] == pol:
                    is_sat = True
                    break
            if not is_sat:
                unsat_bool[i] = True
                unsat_count += 1
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]

        # WalkSAT Loop
        for _ in range(max_steps_per_restart):
            if time.time() >= deadline:
                break
            
            if unsat_count == 0:
                return assignment

            # Pick a random unsatisfied clause
            # We need a list of unsat indices for random.choice
            # Rebuilding list every step is O(N_clauses).
            # N_clauses is small (~300).
            unsat_indices = [i for i in range(num_clauses) if unsat_bool[i]]
            
            if not unsat_indices:
                return assignment

            ci = random.choice(unsat_indices)
            c_data = clause_vars[ci]

            # Choose literal to flip
            if random.random() < p_walk:
                # Random literal from the clause
                v, pol = random.choice(c_data)
            else:
                # Walk Heuristic: Pick variable that appears in the most currently satisfied clauses
                best_v = None
                max_sat_clauses = -1
                
                for v, pol in c_data:
                    count_sat = 0
                    for ci2 in var_to_clauses[v]:
                        if not unsat_bool[ci2]:
                            count_sat += 1
                    
                    if count_sat > max_sat_clauses:
                        max_sat_clauses = count_sat
                        best_v = v
                
                v = best_v

            # Flip variable v
            assignment[v] = not assignment[v]
            
            # Update unsat_bool for all clauses containing v
            for ci2 in var_to_clauses[v]:
                c_data2 = clause_vars[ci2]
                is_sat = False
                for v2, pol2 in c_data2:
                    if assignment[v2] == pol2:
                        is_sat = True
                        break
                
                was_unsat = unsat_bool[ci2]
                is_unsat = not is_sat
                
                if was_unsat and not is_unsat:
                    unsat_count -= 1
                elif not was_unsat and is_unsat:
                    unsat_count += 1
                
                unsat_bool[ci2] = is_unsat

            if unsat_count == 0:
                return assignment

        # Restart
        if best_unsat_count == 0:
            break

    return best_assignment
# EVOLVE-BLOCK-END
