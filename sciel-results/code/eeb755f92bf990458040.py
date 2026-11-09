import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 128
    
    # Precompute all 4-APs (indices 0-based)
    # A 4-AP is (i, i+d, i+2d, i+3d)
    aps = []
    for d in range(1, n // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    # For each variable, store indices of constraints it participates in
    # Actually, for forward checking, we need to know which constraints involve which variables.
    # Let's store for each variable x, the list of (constraint_index, position_in_constraint)
    var_constraints = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in range(4):
            var = ap[pos]
            var_constraints[var].append((idx, pos))

    def solve(k):
        # domains[x] is a bitmask of available colors
        full_mask = (1 << k) - 1
        domains = [full_mask for _ in range(n)]
        assigned = [-1] * n
        unassigned_count = n
        
        # Heuristic: pick variable with smallest domain size
        # We can just iterate 0..n-1 or use a priority queue.
        # Given n=128, simple scan is fast enough.
        
        # Propagation logic:
        # When a variable is assigned, we update domains of other variables in the same APs.
        # Specifically, if we have a constraint (v1, v2, v3, v4) and we assign v1=c1:
        # We need to ensure there exist c2, c3, c4 such that |{c1,c2,c3,c4}| >= 3.
        # This is a complex constraint to propagate directly.
        # Simpler: Unit propagation only when 3 variables in a constraint are assigned.
        # If v1, v2, v3 are assigned, we can restrict v4.
        # If v1, v2 are assigned, we can restrict (v3, v4) pairs? Too expensive.
        
        # Let's use a simpler backtracking with "fail-first" (MRV) and basic forward checking:
        # When assigning v=c, for each constraint containing v, if 3 other vars are assigned, 
        # check if the constraint is still satisfiable. If not, fail.
        # If 2 other vars are assigned, we can prune the domain of the 3rd?
        # Let's just do:
        # 1. Pick unassigned var with min domain size.
        # 2. Try each color in domain.
        # 3. Check consistency with all constraints where 3 vars are already assigned.
        # 4. Recurse.
        
        # To speed up, we can precompute for each constraint the status.
        
        def get_min_domain_var():
            min_size = k + 1
            min_var = -1
            for i in range(n):
                if assigned[i] == -1:
                    size = bin(domains[i]).count('1')
                    if size < min_size:
                        min_size = size
                        min_var = i
                        if size == 1: # Fail fast if forced
                            break
            return min_var, min_size

        def check_constraint(idx, ap):
            # Check if constraint ap is consistent with current assignments and domains
            c1, c2, c3, c4 = ap
            vals = [assigned[c1], assigned[c2], assigned[c3], assigned[c4]]
            
            # Count assigned
            num_assigned = sum(1 for v in vals if v != -1)
            
            if num_assigned == 4:
                # All assigned, check if valid
                distinct = len(set(vals))
                return distinct >= 3
            elif num_assigned == 3:
                # One unassigned. Let's say c4 is unassigned.
                # We need to see if there is a value in domains[c4] that makes distinct >= 3.
                assigned_vals = [v for v in vals if v != -1]
                unassigned_var = None
                for i, v in enumerate(vals):
                    if v == -1:
                        unassigned_var = ap[i]
                        break
                
                # If the 3 assigned values already have >= 3 distinct colors, any value works.
                if len(set(assigned_vals)) >= 3:
                    return True
                
                # If the 3 assigned values have exactly 2 distinct colors, say {A, B}.
                # We need the 4th value to be NOT A and NOT B?
                # No, if we have A, A, B, and we add C (C!=A,B), we have {A,B,C} size 3. OK.
                # If we add A, we have {A,B} size 2. Bad.
                # If we add B, we have {A,B} size 2. Bad.
                # So we need a color in domains[unassigned_var] that is not in set(assigned_vals).
                
                # If the 3 assigned values have exactly 1 distinct color, say {A}.
                # We need the 4th value to be NOT A?
                # If we add B (B!=A), we have {A,B} size 2. Bad.
                # We need at least 3 distinct. So with A,A,A, we can never reach 3 distinct with just one more variable.
                # So this is a conflict.
                
                distinct_set = set(assigned_vals)
                if len(distinct_set) == 1:
                    return False
                elif len(distinct_set) == 2:
                    # Need a color not in distinct_set
                    mask_to_check = domains[unassigned_var]
                    forbidden_mask = 0
                    for c in distinct_set:
                        forbidden_mask |= (1 << c)
                    # If there is any bit set in mask_to_check that is not in forbidden_mask, it's OK
                    if (mask_to_check & ~forbidden_mask) != 0:
                        return True
                    else:
                        return False
                else:
                    # len >= 3
                    return True
            elif num_assigned == 2:
                # Two assigned, two unassigned.
                # It's harder to prune. We can just return True (optimistic) or do deeper check.
                # For now, return True to avoid complex propagation.
                # This might lead to more backtracking but is safe.
                return True
            else:
                return True

        def backtrack(depth):
            if time.time() > deadline:
                return False
            
            if depth == n:
                return True
            
            var, min_size = get_min_domain_var()
            if var == -1:
                return True # All assigned
            
            # If domain is empty, fail
            if min_size == 0:
                return False
            
            # Try each color in the domain
            mask = domains[var]
            while mask:
                lsb = mask & -mask
                c = lsb.bit_length() - 1
                mask ^= lsb
                
                # Assign
                old_assigned = assigned[var]
                old_domain = domains[var]
                assigned[var] = c
                domains[var] = lsb
                
                # Check consistency of all constraints involving var
                consistent = True
                for c_idx, pos in var_constraints[var]:
                    if not check_constraint(c_idx, aps[c_idx]):
                        consistent = False
                        break
                
                if consistent:
                    # Optional: Forward checking on neighbors?
                    # For now, just recurse.
                    if backtrack(depth + 1):
                        return True
                
                # Undo
                assigned[var] = old_assigned
                domains[var] = old_domain
            
            return False

        return backtrack(0)

    # Try k=3, 4, 5...
    # It is highly likely 3 colors are not enough for n=128 for 4-APs.
    # Rado's number for 4-APs with 3 colors is known to be small?
    # Let's just iterate.
    
    best_result = None
    
    for k in range(3, 7):
        if time.time() > deadline - 5:
            break
            
        # Reset state for new k
        # The solve function uses closures, so we need to be careful or re-define.
        # Better to make solve(k) self-contained.
        
        # Re-define solve inside loop or pass k properly.
        # The current solve(k) captures n, aps, var_constraints. That's fine.
        
        if solve(k):
            # Reconstruct colors from assigned array?
            # The solve function modifies 'assigned' in the closure.
            # But 'assigned' is local to solve(k) in my draft above?
            # No, I defined assigned inside solve. So I need to return it.
            pass

    # Let's rewrite solve to return the assignment.
    
    def solve_with_result(k):
        full_mask = (1 << k) - 1
        domains = [full_mask for _ in range(n)]
        assigned = [-1] * n
        
        def get_min_domain_var():
            min_size = k + 1
            min_var = -1
            for i in range(n):
                if assigned[i] == -1:
                    size = bin(domains[i]).count('1')
                    if size < min_size:
                        min_size = size
                        min_var = i
                        if size == 1:
                            break
            return min_var, min_size

        def check_constraint(idx, ap):
            c1, c2, c3, c4 = ap
            vals = [assigned[c1], assigned[c2], assigned[c3], assigned[c4]]
            num_assigned = sum(1 for v in vals if v != -1)
            
            if num_assigned == 4:
                return len(set(vals)) >= 3
            elif num_assigned == 3:
                assigned_vals = [v for v in vals if v != -1]
                unassigned_var = None
                for i, v in enumerate(vals):
                    if v == -1:
                        unassigned_var = ap[i]
                        break
                
                distinct_set = set(assigned_vals)
                if len(distinct_set) >= 3:
                    return True
                elif len(distinct_set) == 2:
                    forbidden_mask = 0
                    for c in distinct_set:
                        forbidden_mask |= (1 << c)
                    if (domains[unassigned_var] & ~forbidden_mask) != 0:
                        return True
                    else:
                        return False
                else: # len == 1
                    return False
            else:
                return True

        def backtrack(depth):
            if time.time() > deadline:
                return False
            
            if depth == n:
                return True
            
            var, min_size = get_min_domain_var()
            if var == -1:
                return True
            
            if min_size == 0:
                return False
            
            mask = domains[var]
            while mask:
                lsb = mask & -mask
                c = lsb.bit_length() - 1
                mask ^= lsb
                
                old_assigned = assigned[var]
                old_domain = domains[var]
                assigned[var] = c
                domains[var] = lsb
                
                consistent = True
                for c_idx, pos in var_constraints[var]:
                    if not check_constraint(c_idx, aps[c_idx]):
                        consistent = False
                        break
                
                if consistent:
                    if backtrack(depth + 1):
                        return True
                
                assigned[var] = old_assigned
                domains[var] = old_domain
            
            return False

        if backtrack(0):
            return assigned
        else:
            return None

    for k in range(3, 8):
        if time.time() > deadline - 10:
            break
        res = solve_with_result(k)
        if res:
            # Verify just in case
            valid = True
            for ap in aps:
                if len(set(res[i] for i in ap)) < 3:
                    valid = False
                    break
            if valid:
                return res
            else:
                # Should not happen if solver is correct
                pass

    # Fallback: Greedy
    colors = [0] * n
    for i in range(n):
        for c in range(128):
            ok = True
            # Check all 4-APs containing i
            for ap in aps:
                if i in ap:
                    # Replace i with c, others with current colors
                    temp_colors = []
                    for j in ap:
                        if j == i:
                            temp_colors.append(c)
                        else:
                            temp_colors.append(colors[j])
                    if len(set(temp_colors)) < 3:
                        ok = False
                        break
            if ok:
                colors[i] = c
                break
    return colors

