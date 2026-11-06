import time
import random
import numpy as np

def solve(instance):
    n = instance["n"]
    orig_clauses = instance["clauses"]
    m = len(orig_clauses)
    
    # 1. Preprocessing: Unit Propagation
    assignment = [None] * n
    active_clauses = [c[:] for c in orig_clauses]
    
    # Unit Propagation
    changed = True
    while changed:
        changed = False
        for i in range(len(active_clauses)):
            if len(active_clauses[i]) == 0:
                return [random.random() < 0.5 for _ in range(n)]
            
            if len(active_clauses[i]) == 1:
                lit = active_clauses[i][0]
                var = abs(lit) - 1
                val = lit > 0
                
                if assignment[var] is not None:
                    if assignment[var] != val:
                        return [random.random() < 0.5 for _ in range(n)]
                    else:
                        active_clauses[i] = []
                        changed = True
                        continue
                
                assignment[var] = val
                changed = True
                
                for j in range(len(active_clauses)):
                    if len(active_clauses[j]) == 0:
                        continue
                    new_clause = []
                    clause_satisfied = False
                    for l in active_clauses[j]:
                        v = abs(l) - 1
                        if v == var:
                            if (l > 0 and val) or (l < 0 and not val):
                                clause_satisfied = True
                                break
                            else:
                                continue
                        else:
                            new_clause.append(l)
                    
                    if clause_satisfied:
                        active_clauses[j] = []
                    else:
                        active_clauses[j] = new_clause
                break
    
    if all(a is not None for a in assignment):
        for c in orig_clauses:
            sat = False
            for l in c:
                v = abs(l) - 1
                if assignment[v] == (l > 0):
                    sat = True
                    break
            if not sat:
                return [random.random() < 0.5 for _ in range(n)]
        return assignment

    # 2. Vectorized WalkSAT
    # Initialize assignment
    for i in range(n):
        if assignment[i] is None:
            assignment[i] = random.random() < 0.5
            
    # Convert to NumPy arrays for vectorized operations
    # clause_vars: shape (m, 3), values are 0-indexed variable indices
    # clause_signs: shape (m, 3), values are +1 or -1
    clause_vars = np.empty((m, 3), dtype=np.int32)
    clause_signs = np.empty((m, 3), dtype=np.int8)
    
    for i, c in enumerate(orig_clauses):
        for j, l in enumerate(c):
            clause_vars[i, j] = abs(l) - 1
            clause_signs[i, j] = 1 if l > 0 else -1
            
    # Initial assignment vector: 1 for True, 0 for False
    # We need to check if (sign * (2*val - 1)) > 0 for satisfaction
    # Let's use val_vec where True=1, False=0.
    # A literal is satisfied if (lit>0 and val) or (lit<0 and not val)
    # This is equivalent to: (sign == 1 and val == 1) or (sign == -1 and val == 0)
    # Which is: sign * (2*val - 1) > 0? 
    # If sign=1, val=1 -> 1*(1) = 1 > 0 (Sat)
    # If sign=1, val=0 -> 1*(-1) = -1 < 0 (Unsat)
    # If sign=-1, val=1 -> -1*(1) = -1 < 0 (Unsat)
    # If sign=-1, val=0 -> -1*(-1) = 1 > 0 (Sat)
    # Yes.
    
    val_vec = np.array([1 if a else 0 for a in assignment], dtype=np.int8)
    
    # Pre-compute polarity contribution
    # polarity[i, j] = clause_signs[i, j]
    # satisfaction[i, j] = 1 if (clause_signs[i, j] == 1 and val_vec[clause_vars[i, j]] == 1) 
    #                    or (clause_signs[i, j] == -1 and val_vec[clause_vars[i, j]] == 0)
    
    # Let's pre-compute a matrix that makes it easy to update.
    # We want to know for each clause, how many literals are satisfied.
    
    # var_to_clauses: list of lists of clause indices
    var_to_clauses = [[] for _ in range(n)]
    # var_pos_in_clause: dict mapping (var, clause_idx) to position j (0,1,2)
    # Or better, for each var, store the clause indices and the positions.
    var_clause_info = [[] for _ in range(n)] # list of (clause_idx, pos)
    
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_to_clauses[v].append(i)
            var_clause_info[v].append((i, j))
            
    # Current satisfaction state
    # sat_count[i] = number of satisfied literals in clause i
    # We can compute this initially.
    
    # Vectorized initial computation
    # For each clause i, check each literal j
    # is_sat[i, j] = 1 if satisfied else 0
    
    # val_vec is 0/1.
    # condition: (sign == 1 and val == 1) or (sign == -1 and val == 0)
    # This is equivalent to: (sign == 1) * (val == 1) + (sign == -1) * (val == 0)
    
    # Let's create a lookup table for the current values.
    # Actually, just compute it directly.
    
    # Get values for all literals in all clauses
    # clause_vars is (m, 3)
    # current_vals = val_vec[clause_vars] -> shape (m, 3)
    current_vals = val_vec[clause_vars]
    
    # Compute satisfaction
    # signs is (m, 3)
    # sat = (signs == 1) * (current_vals == 1) + (signs == -1) * (current_vals == 0)
    # Note: in numpy, boolean comparisons yield 0/1 arrays.
    sat_matrix = ((clause_signs == 1) & (current_vals == 1)) | ((clause_signs == -1) & (current_vals == 0))
    sat_matrix = sat_matrix.astype(np.int8)
    
    sat_count = sat_matrix.sum(axis=1)
    num_unsat = np.sum(sat_count == 0)
    
    if num_unsat == 0:
        return assignment
        
    # Prepare for search
    start_time = time.time()
    deadline = start_time + 290.0
    
    rng = random.Random(42)
    
    best_num_unsat = num_unsat
    best_assignment = assignment[:]
    
    # Phase saving
    phase = assignment[:]
    
    steps = 0
    current_restart_limit = 10000 # Increased due to speed
    max_restart_limit = 500000
    
    while time.time() < deadline:
        if num_unsat == 0:
            return assignment
        
        if num_unsat < best_num_unsat:
            best_num_unsat = num_unsat
            best_assignment = assignment[:]
            
        if steps > current_restart_limit:
            # Restart with phase saving
            for i in range(n):
                assignment[i] = phase[i]
            val_vec = np.array([1 if a else 0 for a in assignment], dtype=np.int8)
            
            # Recompute sat_count
            current_vals = val_vec[clause_vars]
            sat_matrix = ((clause_signs == 1) & (current_vals == 1)) | ((clause_signs == -1) & (current_vals == 0))
            sat_matrix = sat_matrix.astype(np.int8)
            sat_count = sat_matrix.sum(axis=1)
            num_unsat = np.sum(sat_count == 0)
            
            steps = 0
            current_restart_limit = int(current_restart_limit * 1.5)
            if current_restart_limit > max_restart_limit:
                current_restart_limit = max_restart_limit
            continue
        
        # Pick a random unsatisfied clause
        # Find indices where sat_count == 0
        unsat_indices = np.where(sat_count == 0)[0]
        if len(unsat_indices) == 0:
            return assignment
            
        # Random choice from unsat
        unsat_clause_idx = unsat_indices[rng.randrange(len(unsat_indices))]
        
        # Get variables in this clause
        clause_vars_local = clause_vars[unsat_clause_idx]
        
        # Heuristic: Weighted random
        # Calculate score for each variable in the clause
        # Score = gain - loss
        # Gain: clauses that become satisfied (currently 0 sat, will be 1)
        # Loss: clauses that become unsatisfied (currently 1 sat, will be 0)
        
        # We need to evaluate the impact of flipping each variable in the clause.
        # Since the clause has 3 variables, we can just check the 3 candidates.
        
        best_score = -float('inf')
        best_var = None
        best_pos = None
        
        for k in range(3):
            var_idx = clause_vars_local[k]
            
            # Get all clauses containing this variable
            c_indices, c_positions = zip(*var_clause_info[var_idx])
            c_indices = np.array(c_indices, dtype=np.int32)
            c_positions = np.array(c_positions, dtype=np.int8)
            
            # Current values of these literals
            # The literal value in the clause is determined by clause_signs[c, pos] and val_vec[var]
            # Current val: val_vec[var_idx]
            # Current literal satisfaction:
            # sat_lit = (sign == 1 and val == 1) or (sign == -1 and val == 0)
            
            signs_local = clause_signs[c_indices, c_positions]
            vals_local = val_vec[var_idx]
            
            # Current satisfaction of these specific literals
            # is_sat = (signs_local == 1 and vals_local == 1) or (signs_local == -1 and vals_local == 0)
            is_sat_current = ((signs_local == 1) & (vals_local == 1)) | ((signs_local == -1) & (vals_local == 0))
            
            # New value after flip
            new_val = 1 - vals_local
            
            # New satisfaction
            is_sat_new = ((signs_local == 1) & (new_val == 1)) | ((signs_local == -1) & (new_val == 0))
            
            # Change in satisfaction for these literals
            # delta_sat = is_sat_new - is_sat_current
            
            # Impact on clauses:
            # For each clause c in c_indices:
            # old_count = sat_count[c]
            # new_count = old_count + delta_sat[c]
            
            old_counts = sat_count[c_indices]
            delta = (is_sat_new.astype(np.int8) - is_sat_current.astype(np.int8))
            new_counts = old_counts + delta
            
            # Gains: clauses that were 0 and become > 0
            gains = np.sum((old_counts == 0) & (new_counts > 0))
            # Losses: clauses that were > 0 and become 0
            losses = np.sum((old_counts > 0) & (new_counts == 0))
            
            score = gains * 1.0 - losses * 1.5
            
            if score > best_score:
                best_score = score
                best_var = var_idx
                best_pos = k
                
        # Flip the variable
        if best_var is None:
            best_var = int(clause_vars_local[0])
            
        old_val = val_vec[best_var]
        new_val = 1 - old_val
        
        assignment[best_var] = bool(new_val)
        val_vec[best_var] = new_val
        phase[best_var] = bool(new_val)
        
        # Update sat_count and num_unsat
        # Re-evaluate clauses containing best_var
        c_indices, c_positions = zip(*var_clause_info[best_var])
        c_indices = np.array(c_indices, dtype=np.int32)
        c_positions = np.array(c_positions, dtype=np.int8)
        
        signs_local = clause_signs[c_indices, c_positions]
        
        # Current satisfaction of these literals (before flip, but val_vec is already updated? No, we updated val_vec)
        # Wait, we updated val_vec. So we need to be careful.
        # Let's re-calculate based on the OLD val_vec? 
        # Actually, it's easier to just re-compute the sat_count for these clauses from scratch using the NEW val_vec.
        # Since we only changed one variable, only clauses containing that variable change.
        
        # New values for the literals in these clauses
        new_vals = val_vec[c_indices] # This is wrong. val_vec is indexed by var.
        # The literals are in clauses c_indices, at positions c_positions.
        # The variable is best_var.
        # The sign is signs_local.
        # The value is new_val.
        
        # is_sat_new_literal = (signs_local == 1 and new_val == 1) or (signs_local == -1 and new_val == 0)
        is_sat_new = ((signs_local == 1) & (new_val == 1)) | ((signs_local == -1) & (new_val == 0))
        
        # We need the OLD sat_count for these clauses to update.
        # But sat_count has already been used? No, we are in the update phase.
        # We stored old_counts in the heuristic phase? No, that was for a specific candidate.
        # We need to update sat_count for all clauses in c_indices.
        
        # Let's get the current sat_count for these clauses.
        # Note: sat_count might have been modified if we did something else? No.
        # However, we need to know the contribution of the OTHER literals in the clause to get the new total.
        # This is getting complicated to do incrementally without storing the full sat_matrix.
        
        # Alternative: Store sat_matrix (m, 3) and update it.
        # Let's refactor to use sat_matrix.
        
        # Re-implementation of update using sat_matrix
        
        # Actually, let's just re-compute sat_count for the affected clauses by summing the sat_matrix.
        # But we don't have sat_matrix updated yet.
        
        # Let's update sat_matrix first.
        # The literal at (c, pos) for c in c_indices, pos in c_positions corresponds to variable best_var.
        # We need to set sat_matrix[c_indices, c_positions] = is_sat_new
        
        sat_matrix[c_indices, c_positions] = is_sat_new.astype(np.int8)
        
        # Now update sat_count for these clauses
        # new_sat_count = sat_matrix[c_indices].sum(axis=1)
        new_sat_counts = sat_matrix[c_indices].sum(axis=1)
        
        # Update num_unsat
        # For each affected clause:
        # old_sat = sat_count[c]
        # new_sat = new_sat_counts[i]
        
        old_sat_counts = sat_count[c_indices]
        
        # Clauses that became unsat: old > 0, new == 0
        became_unsat = (old_sat_counts > 0) & (new_sat_counts == 0)
        num_became_unsat = np.sum(became_unsat)
        
        # Clauses that became sat: old == 0, new > 0
        became_sat = (old_sat_counts == 0) & (new_sat_counts > 0)
        num_became_sat = np.sum(became_sat)
        
        num_unsat += num_became_unsat - num_became_sat
        
        # Update sat_count array
        sat_count[c_indices] = new_sat_counts
        
        steps += 1
        
    # Final check
    if num_unsat == 0:
        return assignment
        
    return best_assignment
