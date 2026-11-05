"""Synthetic planted 3-SAT solver (50 variables) with optimized incremental local search."""
import random
import time
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    # Precompute clause structure and incidence
    # clause_info[i] = list of (var_idx, polarity) where polarity 1 is positive, -1 is negative
    # var_to_clauses[j] = list of (clause_idx, polarity) for variable j
    clause_info = []
    var_to_clauses = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses):
        info = []
        for l in clause:
            var_idx = abs(l) - 1
            polarity = 1 if l > 0 else -1
            info.append((var_idx, polarity))
            var_to_clauses[var_idx].append((i, polarity))
        clause_info.append(info)
    
    # State variables for incremental updates
    # sat_count[i] = number of satisfied literals in clause i
    # clause_sat[i] = True if sat_count[i] > 0
    sat_count = [0] * num_clauses
    clause_sat = [False] * num_clauses
    total_sat = 0
    
    def update_clause_state(ci):
        """Update sat_count and clause_sat for a specific clause index."""
        cnt = 0
        for var_idx, pol in clause_info[ci]:
            # assignment[var_idx] is True/False
            # pol is 1 (positive) or -1 (negative)
            if pol == 1:
                if assignment[var_idx]:
                    cnt += 1
            else:
                if not assignment[var_idx]:
                    cnt += 1
        sat_count[ci] = cnt
        was_sat = clause_sat[ci]
        is_sat = cnt > 0
        clause_sat[ci] = is_sat
        
        # Return delta in total_sat
        if is_sat and not was_sat:
            return 1
        elif not is_sat and was_sat:
            return -1
        else:
            return 0

    # Initial random assignment
    # We will use a main assignment array that gets modified
    # We need to be careful with restarts, so we'll re-initialize state on restart
    
    def initialize_state(assignment):
        nonlocal sat_count, clause_sat, total_sat
        sat_count = [0] * num_clauses
        clause_sat = [False] * num_clauses
        total_sat = 0
        for i in range(num_clauses):
            delta = update_clause_state(i)
            total_sat += delta
        return total_sat

    def flip_var(var_idx):
        """Flip a variable and update state. Returns delta in total_sat."""
        nonlocal total_sat
        assignment[var_idx] = not assignment[var_idx]
        delta = 0
        # Update all clauses containing this variable
        for ci, pol in var_to_clauses[var_idx]:
            # We need to re-evaluate the clause because the variable value changed
            # However, we can optimize by just adjusting the count based on the change
            # Old value contributed: 1 if satisfied, 0 otherwise
            # New value contributes: 1 if satisfied, 0 otherwise
            
            # Let's just recount for simplicity first, but we can optimize if needed.
            # Given n=50, degree is likely low (<20), recounting 3 literals is fast.
            old_sat = clause_sat[ci]
            old_cnt = sat_count[ci]
            
            # Recalculate
            new_cnt = 0
            for v, p in clause_info[ci]:
                if p == 1:
                    if assignment[v]: new_cnt += 1
                else:
                    if not assignment[v]: new_cnt += 1
            
            sat_count[ci] = new_cnt
            new_sat = new_cnt > 0
            clause_sat[ci] = new_sat
            
            if new_sat and not old_sat:
                delta += 1
            elif not new_sat and old_sat:
                delta -= 1
                
        total_sat += delta
        return delta

    # Helper for WalkSAT damage calculation (optimized)
    def get_damage(var_idx, candidate_clause_idx):
        """Calculate damage if we flip var_idx. 
        Damage = number of clauses that become unsatisfied.
        We only look at clauses containing var_idx.
        """
        damage = 0
        for ci, pol in var_to_clauses[var_idx]:
            if ci == candidate_clause_idx:
                continue # Skip the clause we are trying to satisfy
            
            # Check if this clause is currently satisfied ONLY by var_idx
            # If yes, and flipping var_idx makes it unsatisfied, damage += 1
            
            # Current state:
            # Is it satisfied by var_idx currently?
            # assignment[var_idx] is current value.
            # pol is polarity in this clause.
            current_satisfies_var = (assignment[var_idx] if pol == 1 else not assignment[var_idx])
            
            if not current_satisfies_var:
                # Flipping will not break satisfaction via this var (it wasn't satisfying)
                # But wait, if it was unsatisfied, flipping might satisfy it (good) or leave it unsat (neutral)
                # We only care about BREAKING satisfied clauses.
                # So if it wasn't satisfying this clause via this var, and the clause is satisfied,
                # it must be satisfied by OTHER literals. Flipping this var won't break it.
                continue
                
            # It IS satisfying this clause via this var.
            # Is it the ONLY literal satisfying this clause?
            # Check sat_count[ci]
            if sat_count[ci] == 1:
                # Yes, only this literal satisfies it.
                # If we flip, this literal becomes false.
                # Are there other literals that will become true? No, we only flip one var.
                # So the clause becomes unsatisfied.
                damage += 1
                
        return damage

    def walksat_step(assignment, rng, unsat_clauses):
        """Optimized WalkSAT step."""
        if not unsat_clauses:
            return assignment, 0
        
        clause_idx = rng.choice(unsat_clauses)
        
        best_var = None
        best_damage = float('inf')
        
        # Try each literal in the chosen clause
        for var_idx, pol in clause_info[clause_idx]:
            damage = get_damage(var_idx, clause_idx)
            if damage < best_damage:
                best_damage = damage
                best_var = var_idx
            # Early exit if perfect
            if best_damage == 0:
                break
        
        if best_var is not None:
            delta = flip_var(best_var)
            return assignment, delta
        return assignment, 0

    def get_unsat_clauses_list():
        """Return list of indices of unsatisfied clauses."""
        return [i for i in range(num_clauses) if not clause_sat[i]]

    # Main Solver Logic
    deadline = time.time() + 290 # Safety margin
    best_assignment = None
    best_score = -1
    
    # Strategy 1: WalkSAT with restarts (Primary, as it's good for planted SAT)
    # Strategy 2: Simulated Annealing (Secondary, for escaping local minima)
    
    # We'll interleave or run sequentially based on time.
    # WalkSAT is usually faster to converge if the instance is easy/planted.
    
    max_restarts = 50
    steps_per_restart = 2000 # Increased steps due to O(1) update
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        rng = random.Random(1000 + restart)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        current_score = initialize_state(assignment)
        
        if current_score == num_clauses:
            return assignment
        
        best_local_assignment = assignment[:]
        best_local_score = current_score
        
        unsat_clauses = get_unsat_clauses_list()
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
                break
                
            if not unsat_clauses:
                return assignment
                
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_assignment = assignment[:]
                
            # WalkSAT Step
            assignment, delta = walksat_step(assignment, rng, unsat_clauses)
            current_score += delta
            
            # Update unsat_clauses list efficiently?
            # For n=50, M~100-500, rebuilding the list is O(M).
            # 2000 steps * 500 ops = 1M ops. Very fast.
            unsat_clauses = get_unsat_clauses_list()
            
            if not unsat_clauses:
                return assignment
                
        if best_local_score > best_score:
            best_score = best_local_score
            best_assignment = best_local_assignment[:]
            
        # If we are close to solution, maybe try SA polish?
        # For now, just keep restarting WalkSAT.
        
    # If WalkSAT didn't solve it, try Simulated Annealing from the best point found
    if best_assignment is not None and time.time() < deadline - 10:
        assignment = best_assignment[:]
        current_score = initialize_state(assignment)
        
        if current_score == num_clauses:
            return assignment
            
        # SA parameters
        start_temp = 1.0
        end_temp = 0.01
        sa_steps = 5000
        
        best_sa_assignment = assignment[:]
        best_sa_score = current_score
        
        for step in range(sa_steps):
            if time.time() > deadline - 5:
                break
                
            # Pick random var
            var_idx = random.randint(0, n - 1)
            old_val = assignment[var_idx]
            
            # Flip and calculate delta
            # We need to simulate the flip to get delta without committing yet?
            # Or just commit and rollback if rejected.
            # Committing is easier with the state machine.
            
            # To get delta, we can just flip, record delta, then unflip if rejected.
            # But flip_var modifies state.
            # Let's just flip.
            delta = flip_var(var_idx)
            new_score = current_score + delta
            
            # Temperature
            progress = step / sa_steps
            temp = start_temp * (end_temp / start_temp) ** progress
            
            accepted = False
            if delta >= 0:
                accepted = True
            else:
                # Accept with prob exp(delta/temp)
                # delta is negative.
                if temp > 0:
                    # Avoid underflow
                    if delta / temp > -700: 
                        prob = math.exp(delta / temp)
                    else:
                        prob = 0.0
                    if random.random() < prob:
                        accepted = True
            
            if accepted:
                current_score = new_score
                if current_score > best_sa_score:
                    best_sa_score = current_score
                    best_sa_assignment = assignment[:]
                    if best_sa_score == num_clauses:
                        return assignment
            else:
                # Rollback
                flip_var(var_idx) # Flip back
                # current_score remains unchanged (we didn't update it)
        
        if best_sa_score > best_score:
            best_score = best_sa_score
            best_assignment = best_sa_assignment[:]

    if best_assignment is not None:
        return best_assignment
    
    # Fallback
    return [False] * n
