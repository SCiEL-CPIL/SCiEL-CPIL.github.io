"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Safety margin for the 330s limit
    deadline = time.time() + 310
    
    # 1. Preprocessing: Unit Propagation and Pure Literal Elimination
    # This often solves easy instances or simplifies hard ones.
    
    def preprocess(cnf):
        # cnf is a list of lists of literals
        # Returns (solved_assignment, reduced_cnf, partial_assignment)
        # solved_assignment is a list of bools or None
        # partial_assignment is a list of bool/None
        
        assignment = [None] * n
        cnf_work = [list(c) for c in cnf]
        
        changed = True
        while changed:
            changed = False
            
            # Remove satisfied clauses
            new_cnf = []
            for c in cnf_work:
                sat = False
                for lit in c:
                    vi = abs(lit) - 1
                    if assignment[vi] is not None:
                        if assignment[vi] == (lit > 0):
                            sat = True
                            break
                if not sat:
                    new_cnf.append(c)
            cnf_work = new_cnf
            
            # Unit Propagation
            for i, c in enumerate(cnf_work):
                if len(c) == 1:
                    lit = c[0]
                    vi = abs(lit) - 1
                    if assignment[vi] is None:
                        assignment[vi] = (lit > 0)
                        changed = True
                    elif assignment[vi] != (lit > 0):
                        return None, None, assignment # Contradiction
                    else:
                        cnf_work[i] = []
                        changed = True
                elif len(c) == 0:
                    return None, None, assignment # Contradiction
            
            # Pure Literal Elimination
            # Count occurrences of pos and neg literals
            pos_count = [0] * n
            neg_count = [0] * n
            for c in cnf_work:
                seen_vars = set()
                for lit in c:
                    vi = abs(lit) - 1
                    if vi in seen_vars:
                        continue
                    seen_vars.add(vi)
                    if lit > 0:
                        pos_count[vi] += 1
                    else:
                        neg_count[vi] += 1
            
            for vi in range(n):
                if assignment[vi] is not None:
                    continue
                if pos_count[vi] == 0 and neg_count[vi] > 0:
                    assignment[vi] = False
                    changed = True
                elif neg_count[vi] == 0 and pos_count[vi] > 0:
                    assignment[vi] = True
                    changed = True
        
        # Check if all assigned
        if all(a is not None for a in assignment):
            final_assign = [bool(a) for a in assignment]
            # Verify
            is_sat = True
            for c in cnf:
                sat = False
                for lit in c:
                    vi = abs(lit) - 1
                    if final_assign[vi] == (lit > 0):
                        sat = True
                        break
                if not sat:
                    is_sat = False
                    break
            if is_sat:
                return final_assign, None, final_assign
            return None, None, assignment
        
        return None, cnf_work, assignment

    solved, reduced_cnf, partial_assign = preprocess(clauses)
    if solved is not None:
        return solved
    
    # If reduced_cnf is None, it means contradiction or full assignment failed, 
    # but usually we just use the original if reduced is empty or None.
    # If reduced_cnf is not None, use it.
    if reduced_cnf is not None and len(reduced_cnf) > 0:
        active_cnf = [tuple(c) for c in reduced_cnf if len(c) > 0]
    else:
        active_cnf = [tuple(c) for c in clauses]
        
    m = len(active_cnf)
    
    # Build index
    var_to_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(active_cnf):
        for lit in c:
            vi = abs(lit) - 1
            var_to_clauses[vi].append(ci)
    
    # MinConfl / WalkSAT Hybrid
    # State:
    # assignment: list of bools
    # clause_sat_count: list of ints, number of satisfied literals in each clause
    # unsat_count: int
    
    def init_state(assign):
        clause_sat_count = [0] * m
        for ci, c in enumerate(active_cnf):
            count = 0
            for lit in c:
                vi = abs(lit) - 1
                if assign[vi] == (lit > 0):
                    count += 1
            clause_sat_count[ci] = count
        
        unsat_count = 0
        for count in clause_sat_count:
            if count == 0:
                unsat_count += 1
        return clause_sat_count, unsat_count

    def min_confl_step(assign, clause_sat_count, unsat_count, rng):
        if unsat_count == 0:
            return assign, clause_sat_count, 0, True
        
        # Calculate delta for each variable
        # delta[i] = (clauses that become sat) - (clauses that become unsat)
        # We want to maximize delta (minimize unsat)
        
        best_delta = -1000
        best_var = -1
        candidates = []
        
        for vi in range(n):
            # Determine current and new literal masks
            current_val = assign[vi]
            new_val = not current_val
            
            gain = 0
            loss = 0
            
            for ci in var_to_clauses[vi]:
                c = active_cnf[ci]
                # Find the literal in c corresponding to vi
                lit_val = None
                for lit in c:
                    if abs(lit) - 1 == vi:
                        lit_val = lit
                        break
                
                if lit_val is None:
                    continue
                
                current_lit_sat = (current_val == (lit_val > 0))
                new_lit_sat = (new_val == (lit_val > 0))
                
                current_count = clause_sat_count[ci]
                
                if current_count == 0:
                    # Clause is currently unsat
                    if new_lit_sat:
                        # Flipping vi will satisfy this clause
                        gain += 1
                else:
                    # Clause is currently sat
                    if current_lit_sat and not new_lit_sat:
                        # vi is currently satisfying this clause
                        # If we flip, we lose this satisfaction
                        # Check if it was the ONLY satisfaction
                        if current_count == 1:
                            # It becomes unsat
                            loss += 1
                    # If current_lit_sat is False, flipping doesn't affect satisfaction status 
                    # (it remains sat due to others, or was already unsat due to others 
                    # but wait, if current_count > 0 and current_lit_sat is False, 
                    # then other literals are satisfying it. Flipping vi to new_lit_sat 
                    # might make it 2 sat, but it was already sat. So no change in unsat status.)
                    # Actually, if current_count > 0, the clause is sat.
                    # If we flip vi:
                    # - If current_lit_sat was True and new_lit_sat is False:
                    #   - If current_count == 1, it becomes 0 (loss)
                    #   - If current_count > 1, it remains > 0 (no change)
                    # - If current_lit_sat was False:
                    #   - If new_lit_sat is True:
                    #     - Count increases by 1. Clause remains sat. (No change in unsat status)
                    #   - If new_lit_sat is False:
                    #     - Count unchanged. (No change)
                    
            delta = gain - loss
            if delta > best_delta:
                best_delta = delta
                best_var = vi
                candidates = [vi]
            elif delta == best_delta:
                candidates.append(vi)
        
        # If best_delta is negative, we are stuck in a local minimum.
        # We should still pick a variable to escape, but maybe randomly or the "least bad".
        # Standard MinConfl: pick the one with max delta. If all are negative, pick one randomly 
        # from those with max delta (which is still negative) or just random variable.
        # To ensure progress, if best_delta < 0, we can pick a random variable 
        # to act as a "kick" (WalkSAT style).
        
        if best_delta < 0:
            # Random kick
            vi = rng.randint(0, n - 1)
        else:
            # Pick randomly from best candidates
            vi = rng.choice(candidates)
        
        # Perform the flip
        old_val = assign[vi]
        new_val = not old_val
        assign[vi] = new_val
        
        # Update clause_sat_count and unsat_count
        # We only need to update clauses containing vi
        for ci in var_to_clauses[vi]:
            c = active_cnf[ci]
            lit_val = None
            for lit in c:
                if abs(lit) - 1 == vi:
                    lit_val = lit
                    break
            
            old_lit_sat = (old_val == (lit_val > 0))
            new_lit_sat = (new_val == (lit_val > 0))
            
            old_count = clause_sat_count[ci]
            
            if old_lit_sat and not new_lit_sat:
                # We lost a satisfaction
                clause_sat_count[ci] = old_count - 1
                if old_count == 1:
                    # Was sat, now unsat
                    unsat_count += 1
            elif not old_lit_sat and new_lit_sat:
                # We gained a satisfaction
                clause_sat_count[ci] = old_count + 1
                if old_count == 0:
                    # Was unsat, now sat
                    unsat_count -= 1
            
        return assign, clause_sat_count, unsat_count, unsat_count == 0

    # Main Loop
    # Multiple restarts
    # Initial assignment: based on partial_assign if available, else random
    
    best_assign = None
    best_unsat = m + 1
    
    # Get a base assignment
    base_assign = []
    for i in range(n):
        if partial_assign[i] is not None:
            base_assign.append(bool(partial_assign[i]))
        else:
            base_assign.append(False) # Will be randomized below
    
    restarts = 0
    max_restarts = 5000 # Limit restarts to avoid infinite loops if time runs out
    steps_per_restart = 2000
    
    while time.time() < deadline and restarts < max_restarts:
        # Randomize the unassigned variables in base_assign
        assign = base_assign[:]
        for i in range(n):
            if partial_assign[i] is None:
                assign[i] = random.random() < 0.5
        
        # Initialize state
        clause_sat_count, unsat_count = init_state(assign)
        
        if unsat_count == 0:
            return assign
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assign = assign[:]
        
        # Run MinConfl for a fixed number of steps
        rng = random.Random(restarts)
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            assign, clause_sat_count, unsat_count, solved = min_confl_step(assign, clause_sat_count, unsat_count, rng)
            
            if solved:
                return assign
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assign[:]
        
        restarts += 1
        
        # If we are stuck (best_unsat hasn't improved in many restarts), 
        # we might need to increase steps or change strategy.
        # For now, simple restarts work well for planted SAT.
    
    # If we didn't solve it, return the best assignment found
    if best_assign is not None:
        return best_assign
    else:
        return [False] * n

# EVOLVE-BLOCK-END
