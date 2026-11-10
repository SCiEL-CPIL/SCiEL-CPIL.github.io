"""Simulated Annealing based 3-SAT solver."""
# EVOLVE-BLOCK-START
import random
import math
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Pre-compute variable to clause mapping for efficient delta calculation
    # var_to_clauses[var] = list of (clause_idx, sign) where sign is 1 for pos, -1 for neg
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for lit in clause:
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_to_clauses[var_idx].append((i, sign))
    
    def initial_assignment(rng):
        return [rng.random() < 0.5 for _ in range(n)]

    def compute_sat_state(assignment):
        # Returns (clause_satisfied_list, unsat_count)
        clause_satisfied = [False] * len(clauses)
        unsat_count = 0
        for i, clause in enumerate(clauses):
            is_sat = False
            for lit in clause:
                idx = abs(lit) - 1
                if assignment[idx] == (lit > 0):
                    is_sat = True
                    break
            clause_satisfied[i] = is_sat
            if not is_sat:
                unsat_count += 1
        return clause_satisfied, unsat_count

    # SA Parameters
    # We will try multiple restarts
    start_time = time.time()
    time_limit = 280.0  # Leave buffer for overhead
    
    best_assignment = None
    best_unsat = float('inf')
    
    def run_sa(seed_offset):
        rng = random.Random(42 + seed_offset)
        assignment = initial_assignment(rng)
        clause_satisfied, unsat_count = compute_sat_state(assignment)
        
        if unsat_count == 0:
            return assignment, 0
        
        # SA Parameters
        T_start = 5.0
        T_end = 0.01
        cooling_rate = 0.9995
        max_iterations = 20000
        
        T = T_start
        
        for _ in range(max_iterations):
            if unsat_count == 0:
                break
            
            # Pick a random variable
            var = rng.randint(0, n - 1)
            
            # Calculate delta and potential new states
            delta = 0
            old_val = assignment[var]
            new_val = not old_val
            
            # We need to know which clauses change state
            # If a clause is currently satisfied, and the var literal is the *only* true literal, it becomes unsatisfied.
            # If a clause is currently unsatisfied, and the var literal becomes true, it becomes satisfied.
            
            # To determine if var literal is the only true literal in a satisfied clause,
            # we need to check other literals.
            
            # We will simulate the flip, update state, and revert if rejected.
            
            # Save old states of affected clauses
            affected_clause_indices = [c_idx for c_idx, _ in var_to_clauses[var]]
            old_states = [clause_satisfied[c_idx] for c_idx in affected_clause_indices]
            
            # Perform flip
            assignment[var] = new_val
            
            # Update clause_satisfied and unsat_count
            for c_idx, sign in var_to_clauses[var]:
                # Determine if this literal is true under new_val
                lit_true = (new_val if sign == 1 else not new_val)
                
                if lit_true:
                    # Clause is definitely satisfied now
                    if not clause_satisfied[c_idx]:
                        clause_satisfied[c_idx] = True
                        unsat_count -= 1
                    # else: already satisfied, no change
                else:
                    # This literal is false.
                    # If clause was satisfied, we need to check if it remains satisfied by other literals.
                    # If clause was unsatisfied, it remains unsatisfied (since this lit is false and others were false).
                    if clause_satisfied[c_idx]:
                        # Check if any other literal is true
                        is_still_sat = False
                        for lit in clauses[c_idx]:
                            idx = abs(lit) - 1
                            if idx == var:
                                continue
                            if assignment[idx] == (lit > 0):
                                is_still_sat = True
                                break
                        if not is_still_sat:
                            clause_satisfied[c_idx] = False
                            unsat_count += 1
                    # else: unsatisfied stays unsatisfied
            
            # Calculate delta for acceptance
            # We can just compare old unsat_count vs new, but we modified unsat_count.
            # Let's recalculate delta from old_states and new states?
            # Or simpler: store old_unsat_count.
            
            # Actually, the standard way is to compute delta before modifying.
            # Let's revert and do it properly.
            
            # Revert
            assignment[var] = old_val
            for c_idx, old_state in zip(affected_clause_indices, old_states):
                clause_satisfied[c_idx] = old_state
            
            # Recalculate delta properly
            delta = 0
            for c_idx, sign in var_to_clauses[var]:
                was_sat = clause_satisfied[c_idx]
                
                # Check if clause is satisfied with new_val
                lit_true = (new_val if sign == 1 else not new_val)
                
                if lit_true:
                    is_sat_now = True
                else:
                    # Check other literals
                    is_sat_now = False
                    if was_sat:
                        # It was sat, so at least one other lit was true.
                        # But wait, if was_sat is True and lit_true is False, we need to check if *another* lit is true.
                        # If was_sat is True, and we flip var to make its lit false, the clause is sat iff some other lit is true.
                        for lit in clauses[c_idx]:
                            idx = abs(lit) - 1
                            if idx == var:
                                continue
                            if assignment[idx] == (lit > 0):
                                is_sat_now = True
                                break
                    else:
                        # It was unsat, so all other lits are false.
                        # If lit_true is false, it remains unsat.
                        is_sat_now = False
                
                if was_sat and not is_sat_now:
                    delta += 1
                elif not was_sat and is_sat_now:
                    delta -= 1
            
            # Acceptance
            if delta <= 0:
                # Commit the flip
                assignment[var] = new_val
                # Update states
                for c_idx, sign in var_to_clauses[var]:
                    lit_true = (new_val if sign == 1 else not new_val)
                    if lit_true:
                        if not clause_satisfied[c_idx]:
                            clause_satisfied[c_idx] = True
                            unsat_count -= 1
                    else:
                        if clause_satisfied[c_idx]:
                            # Check if still sat
                            is_still_sat = False
                            for lit in clauses[c_idx]:
                                idx = abs(lit) - 1
                                if idx == var:
                                    continue
                                if assignment[idx] == (lit > 0):
                                    is_still_sat = True
                                    break
                            if not is_still_sat:
                                clause_satisfied[c_idx] = False
                                unsat_count += 1
            else:
                if T > T_end:
                    prob = math.exp(-delta / T)
                    if rng.random() < prob:
                        # Commit the flip
                        assignment[var] = new_val
                        for c_idx, sign in var_to_clauses[var]:
                            lit_true = (new_val if sign == 1 else not new_val)
                            if lit_true:
                                if not clause_satisfied[c_idx]:
                                    clause_satisfied[c_idx] = True
                                    unsat_count -= 1
                            else:
                                if clause_satisfied[c_idx]:
                                    is_still_sat = False
                                    for lit in clauses[c_idx]:
                                        idx = abs(lit) - 1
                                        if idx == var:
                                            continue
                                        if assignment[idx] == (lit > 0):
                                            is_still_sat = True
                                            break
                                    if not is_still_sat:
                                        clause_satisfied[c_idx] = False
                                        unsat_count += 1
            
            T *= cooling_rate
            
            if _ % 1000 == 0 and time.time() - start_time > time_limit:
                break
        
        return assignment, unsat_count

    # Run multiple attempts
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        assignment, unsat = run_sa(i)
        if unsat == 0:
            return assignment
        
        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = assignment[:]
            
    # If not solved, return best found
    if best_assignment is None:
        return [False] * n
    return best_assignment

# EVOLVE-BLOCK-END
