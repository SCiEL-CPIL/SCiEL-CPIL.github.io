"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n
    
    # Deadline guard: 325s safety margin out of 330s limit
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Pre-compute clause structure
    clause_vars = []
    for c in clauses:
        cv = [(abs(l) - 1, l > 0) for l in c]
        clause_vars.append(cv)
    
    # For each variable, store list of (clause_idx, is_positive)
    var_clauses = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for vi, pos in cv:
            var_clauses[vi].append((ci, pos))
    
    # State: assignment as list of bools
    # clause_unsat_count[ci] = number of falsified literals in clause ci
    # clause_weight[ci] = weight of the clause
    
    def init_state(rng):
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_unsat_count = [0] * m
        clause_weight = [1.0] * m  # Start with uniform weights
        for ci, cv in enumerate(clause_vars):
            count = 0
            for vi, pos in cv:
                if assignment[vi] != pos:
                    count += 1
            clause_unsat_count[ci] = count
        return assignment, clause_unsat_count, clause_weight
    
    def get_weighted_unsat(clause_unsat_count, clause_weight):
        total = 0.0
        for i in range(m):
            if clause_unsat_count[i] >= 3:
                total += clause_weight[i]
        return total
    
    def find_unsat_clause(clause_unsat_count, clause_weight, rng):
        # Find an unsatisfied clause. 
        # Strategy: Pick the unsatisfied clause with the highest weight.
        # If there are ties or to add stochasticity, we can pick randomly among top K or just the max.
        # Let's pick the one with max weight. If multiple, pick randomly.
        max_w = -1
        candidates = []
        for ci in range(m):
            if clause_unsat_count[ci] >= 3:
                if clause_weight[ci] > max_w:
                    max_w = clause_weight[ci]
                    candidates = [ci]
                elif clause_weight[ci] == max_w:
                    candidates.append(ci)
        if not candidates:
            return -1
        return rng.choice(candidates)
    
    def flip_var(vi, assignment, clause_unsat_count):
        """Flip variable vi and update clause_unsat_count incrementally."""
        old_val = assignment[vi]
        assignment[vi] = not old_val
        for ci, pos in var_clauses[vi]:
            if old_val == pos:
                # Was satisfied, now falsified
                clause_unsat_count[ci] += 1
            else:
                # Was falsified, now satisfied
                clause_unsat_count[ci] -= 1
    
    best_assignment = None
    best_weighted_unsat = float('inf')
    
    rng = random.Random(0)
    
    # Parameters for Weighted WalkSAT
    max_restarts = 100
    base_steps = 10000
    max_steps = 100000
    stagnation_limit = 2000
    weight_boost = 2.0
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        assignment, clause_unsat_count, clause_weight = init_state(rng)
        current_weighted_unsat = get_weighted_unsat(clause_unsat_count, clause_weight)
        
        if current_weighted_unsat == 0:
            return assignment
        
        if current_weighted_unsat < best_weighted_unsat:
            best_weighted_unsat = current_weighted_unsat
            best_assignment = list(assignment)
        
        steps_in_restart = base_steps + restart * 5000
        steps_in_restart = min(steps_in_restart, max_steps)
        
        steps_without_improvement = 0
        local_best_unsat = current_weighted_unsat
        
        for step in range(steps_in_restart):
            if time.time() > deadline:
                break
            
            # Find an unsatisfied clause
            unsat_idx = find_unsat_clause(clause_unsat_count, clause_weight, rng)
            if unsat_idx == -1:
                return assignment
            
            # Get the clause
            cv = clause_vars[unsat_idx]
            
            # For each variable in the clause, compute the change in weighted unsat count
            best_var = -1
            best_delta = float('inf')
            
            for vi, pos in cv:
                # Calculate delta if we flip vi
                delta = 0.0
                for cj, cpos in var_clauses[vi]:
                    if cj == unsat_idx:
                        # We are flipping a var in the unsat clause itself.
                        # This literal will become satisfied (since we picked an unsat clause, 
                        # and we assume we are trying to satisfy it).
                        # Wait, if we flip vi, the literal (vi, pos) in clause cj=unsat_idx changes.
                        # Currently unsat means all 3 are false. So (vi, pos) is false.
                        # After flip, it becomes true. So clause cj becomes satisfied (unless other vars change, which they don't).
                        # So unsat_count[cj] decreases by 1.
                        # If it was 3, it becomes 2. It was unsat, now sat.
                        # Weighted contribution goes from weight[cj] to 0.
                        # Delta = -weight[cj].
                        if clause_unsat_count[cj] == 3:
                            delta -= clause_weight[cj]
                        # If it was >3 (impossible for 3-SAT) or <3 (impossible for unsat clause), no change in sat status.
                        # Actually, if clause_unsat_count[cj] < 3, it's already sat. 
                        # But we picked an unsat clause, so clause_unsat_count[unsat_idx] is 3.
                        # So delta is definitely -weight[unsat_idx].
                    else:
                        # Clause cj is different from unsat_idx.
                        # Check if flipping vi changes the satisfaction status of cj.
                        # Literal (vi, cpos) in cj.
                        # If assignment[vi] == cpos, literal is true. After flip, false.
                        #   If cj was sat (count < 3), it might become unsat if count was 2.
                        #   If cj was unsat (count == 3), it stays unsat (count becomes 4? No, max 3).
                        #     Wait, if cj was unsat (count 3), all literals false. 
                        #     So assignment[vi] != cpos. 
                        #     After flip, assignment[vi] == cpos. Literal becomes true.
                        #     Count becomes 2. Clause becomes sat.
                        #     Delta = -weight[cj].
                        # If assignment[vi] != cpos, literal is false. After flip, true.
                        #   If cj was unsat (count 3), it becomes sat (count 2). Delta = -weight[cj].
                        #   If cj was sat (count < 3), it stays sat. Delta = 0.
                        
                        if assignment[vi] == cpos:
                            # Literal is currently TRUE.
                            # After flip, it becomes FALSE.
                            # Clause cj might become unsat.
                            # It becomes unsat if current count is 2 (so new count is 3).
                            if clause_unsat_count[cj] == 2:
                                delta += clause_weight[cj]
                            # If count < 2, it stays sat.
                            # If count == 3, it was already unsat. New count 4? No, max 3. 
                            # Actually, if count==3, all are false. So assignment[vi] != cpos. 
                            # So this branch (assignment[vi] == cpos) is only possible if count < 3.
                        else:
                            # Literal is currently FALSE.
                            # After flip, it becomes TRUE.
                            # Clause cj might become sat.
                            # It becomes sat if current count is 3.
                            if clause_unsat_count[cj] == 3:
                                delta -= clause_weight[cj]
                
                # Add a small random tiebreaker to avoid cycles
                score = delta + rng.random() * 0.01
                if score < best_delta:
                    best_delta = score
                    best_var = vi
            
            if best_var == -1:
                best_var = rng.choice(cv)[0]
            
            # Flip the variable
            flip_var(best_var, assignment, clause_unsat_count)
            
            # Update weighted unsat
            current_weighted_unsat = get_weighted_unsat(clause_unsat_count, clause_weight)
            
            if current_weighted_unsat < local_best_unsat:
                local_best_unsat = current_weighted_unsat
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
            
            if current_weighted_unsat == 0:
                return assignment
            
            if current_weighted_unsat < best_weighted_unsat:
                best_weighted_unsat = current_weighted_unsat
                best_assignment = list(assignment)
            
            # Stagnation handling: Boost weights of unsatisfied clauses
            if steps_without_improvement > stagnation_limit:
                for ci in range(m):
                    if clause_unsat_count[ci] >= 3:
                        clause_weight[ci] *= weight_boost
                steps_without_improvement = 0
        
        # End of restart
        if time.time() > deadline:
            break
    
    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: random assignment
    rng2 = random.Random(42)
    return [rng2.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
