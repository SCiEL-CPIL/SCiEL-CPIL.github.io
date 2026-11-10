"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def solve(instance):
    """
    Multi-restart WalkSAT solver with lazy deletion of unsatisfied clauses.
    Uses adaptive flip probabilities to balance exploration and exploitation.
    """
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin for 330s limit

    # Pre-extract clause literals as (var_idx, sign)
    # sign: 1 if positive literal, -1 if negative literal
    clause_data = []
    for clause in clauses:
        clause_data.append([(abs(l) - 1, 1 if l > 0 else -1) for l in clause])
    
    num_clauses = len(clause_data)

    # Build index: var_idx -> list of clause_indices
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clause_data):
        for var_idx, sign in clause:
            var_to_clauses[var_idx].append(ci)

    rng = random.Random(42)
    
    # Multi-restart strategy
    # We will run restarts until deadline
    # Each restart has a max_steps limit that can grow or be fixed
    base_max_steps = 10000
    restart_count = 0
    
    while time.time() < deadline:
        restart_count += 1
        
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize sat_count: number of true literals in each clause
        sat_count = [0] * num_clauses
        
        # Lazy list of unsatisfied clauses
        # We store clause indices. If a clause becomes satisfied, we don't remove it immediately,
        # but check when selected.
        unsat_clauses_list = []
        
        for ci, clause in enumerate(clause_data):
            count = 0
            for var_idx, sign in clause:
                if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                    count += 1
            sat_count[ci] = count
            if count == 0:
                unsat_clauses_list.append(ci)
        
        if not unsat_clauses_list:
            return assignment

        # Adaptive parameters for this restart
        # p: probability of choosing the best move (min break count)
        # Start high, decay? Or keep fixed? 
        # Standard WalkSAT uses fixed p ~ 0.5.
        # We can vary p per restart to explore different behaviors.
        p_best = 0.5 + rng.uniform(-0.2, 0.2) # 0.3 to 0.7
        max_steps = base_max_steps + restart_count * 1000 # Slightly increase effort per restart
        
        steps = 0
        while steps < max_steps:
            if time.time() > deadline:
                return assignment
            
            if not unsat_clauses_list:
                return assignment

            steps += 1
            
            # 1. Select a random unsatisfied clause
            # Lazy deletion: pick an index, check if it's still unsat
            while True:
                if not unsat_clauses_list:
                    return assignment
                idx = rng.randrange(len(unsat_clauses_list))
                ci = unsat_clauses_list[idx]
                if sat_count[ci] == 0:
                    break
                else:
                    # Stale entry, remove it
                    unsat_clauses_list[idx] = unsat_clauses_list[-1]
                    unsat_clauses_list.pop()

            # 2. Select a variable to flip from this clause
            clause = clause_data[ci]
            # Calculate break count for each variable in the clause
            # break_count[var] = number of OTHER clauses that are currently satisfied 
            # but would become unsatisfied if we flip this variable.
            
            # Optimization: We only need to check clauses connected to the variables in this clause.
            # There are 3 variables.
            
            best_var = -1
            best_breaks = float('inf')
            
            candidates = []
            
            for var_idx, sign in clause:
                # Calculate cost of flipping var_idx
                breaks = 0
                for j in var_to_clauses[var_idx]:
                    if j == ci:
                        continue # Skip the clause we are trying to satisfy
                    # Check if clause j is currently satisfied
                    if sat_count[j] > 0:
                        # If we flip var_idx, does this literal become false?
                        # Current state:
                        # lit_is_true = (assignment[var_idx] and sign_in_j == 1) ...
                        # Wait, sign_in_j is the sign of var_idx in clause j.
                        # We need to know if the literal (var_idx, sign_in_j) is currently true.
                        # If it is true, and we flip var_idx, it becomes false.
                        # If sat_count[j] was 1, it becomes 0 (break).
                        # If sat_count[j] > 1, it stays satisfied.
                        
                        # We need the sign of var_idx in clause j.
                        # var_to_clauses[var_idx] stores (clause_idx, sign).
                        # Let's refactor var_to_clauses to store signs for faster access?
                        # Or just look up? 
                        # The current structure var_to_clauses[var_idx] is list of clause indices.
                        # We need the sign. Let's build a parallel structure or store tuples.
                        
                        # Let's stick to the tuple structure from the seed to avoid major refactor risks,
                        # but the seed had (ci, sign).
                        # I will rebuild var_to_clauses to store (clause_idx, sign).
                        pass 
                        
                # This loop is getting complex. Let's use the structure:
                # var_to_clauses[var_idx] = [(clause_idx, sign), ...]
                
                # I will re-initialize var_to_clauses with signs for efficiency in the inner loop.
                pass
                
            # Since I'm rewriting, let's define the structures correctly at the start.
            break # Break out of while to restructure initialization
            
    # The logic above was pseudo-code. I will implement the clean version below.
    
    # --- RE-IMPLEMENTATION WITH CORRECT DATA STRUCTURES ---
    
    # Re-initialize properly
    clause_data = []
    for clause in clauses:
        clause_data.append([(abs(l) - 1, 1 if l > 0 else -1) for l in clause])
    
    num_clauses = len(clause_data)
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clause_data):
        for var_idx, sign in clause:
            var_to_clauses[var_idx].append((ci, sign))

    rng = random.Random(42)
    base_max_steps = 20000
    restart_count = 0
    
    while time.time() < deadline:
        restart_count += 1
        
        assignment = [rng.random() < 0.5 for _ in range(n)]
        sat_count = [0] * num_clauses
        unsat_clauses_list = []
        
        for ci, clause in enumerate(clause_data):
            count = 0
            for var_idx, sign in clause:
                if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                    count += 1
            sat_count[ci] = count
            if count == 0:
                unsat_clauses_list.append(ci)
        
        if not unsat_clauses_list:
            return assignment

        # Adaptive probability of choosing best move
        p_best = 0.5 + rng.uniform(-0.3, 0.3)
        max_steps = base_max_steps
        
        steps = 0
        while steps < max_steps:
            if time.time() > deadline:
                return assignment
            
            if not unsat_clauses_list:
                return assignment

            steps += 1
            
            # 1. Select a random unsatisfied clause (Lazy Deletion)
            while True:
                if not unsat_clauses_list:
                    return assignment
                idx = rng.randrange(len(unsat_clauses_list))
                ci = unsat_clauses_list[idx]
                if sat_count[ci] == 0:
                    break
                else:
                    # Stale entry
                    unsat_clauses_list[idx] = unsat_clauses_list[-1]
                    unsat_clauses_list.pop()

            clause = clause_data[ci]
            
            # 2. Evaluate flips for the 3 variables in this clause
            # We want to minimize the number of OTHER clauses we break.
            # A clause j is "broken" if:
            #   - It is currently satisfied (sat_count[j] > 0)
            #   - The literal (var, sign_j) is currently true
            #   - Flipping var makes the literal false
            #   - And sat_count[j] == 1 (so it becomes 0)
            
            # If sat_count[j] > 1, flipping doesn't make it unsat.
            
            best_var = -1
            best_breaks = float('inf')
            candidates = []
            
            for var_idx, sign in clause:
                breaks = 0
                for j, sign_j in var_to_clauses[var_idx]:
                    if j == ci:
                        continue
                    # Is clause j currently satisfied?
                    if sat_count[j] > 0:
                        # Is the literal (var_idx, sign_j) currently true?
                        lit_true = (assignment[var_idx] and sign_j == 1) or (not assignment[var_idx] and sign_j == -1)
                        if lit_true:
                            # If we flip, this literal becomes false.
                            # If it was the only true literal, clause j becomes unsat.
                            if sat_count[j] == 1:
                                breaks += 1
                            
                candidates.append((var_idx, breaks))
                if breaks < best_breaks:
                    best_breaks = breaks
                    best_var = var_idx
                    
            # 3. Choose variable
            if rng.random() < p_best:
                var_idx = best_var
            else:
                # Pick randomly from candidates
                var_idx, _ = candidates[rng.randrange(3)]
                
            # 4. Flip variable
            assignment[var_idx] = not assignment[var_idx]
            
            # 5. Update sat_count and unsat_clauses_list
            for j, sign_j in var_to_clauses[var_idx]:
                # Determine if literal (var_idx, sign_j) is now true
                lit_true_now = (assignment[var_idx] and sign_j == 1) or (not assignment[var_idx] and sign_j == -1)
                
                if lit_true_now:
                    # Literal became True. sat_count increases.
                    if sat_count[j] == 0:
                        # Clause was unsat, now sat.
                        # It's in unsat_clauses_list (possibly as stale), we don't remove it now (lazy).
                        # But we must ensure we don't count it as unsat later.
                        # The check `if sat_count[ci] == 0` handles this.
                        pass
                    sat_count[j] += 1
                else:
                    # Literal became False. sat_count decreases.
                    if sat_count[j] == 1:
                        # Clause was sat (count 1), now unsat (count 0).
                        sat_count[j] = 0
                        unsat_clauses_list.append(j)
                    else:
                        sat_count[j] -= 1
                        # If sat_count[j] was 0, it becomes -1? No, logic prevents this.
                        # If it was > 1, it decreases.
                        
        # End of restart

    return assignment

# EVOLVE-BLOCK-END
