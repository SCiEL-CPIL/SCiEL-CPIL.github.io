import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 295.0  # Safety margin

    # Precomputation
    # clause_vars[i] = list of (var_index, is_positive)
    clause_vars = []
    for c in clauses:
        clause_vars.append([(abs(l) - 1, l > 0) for l in c])

    # Bitmasks for variable occurrences
    # var_pos_clauses[v]: bitmask of clauses where variable v appears positively
    # var_neg_clauses[v]: bitmask of clauses where variable v appears negatively
    var_pos_clauses = [0] * n
    var_neg_clauses = [0] * n
    for i, c in enumerate(clauses):
        for l in c:
            if l > 0:
                var_pos_clauses[l - 1] |= (1 << i)
            else:
                var_neg_clauses[-l - 1] |= (1 << i)

    # Precompute literal masks for each clause to quickly check "only literal" conditions
    # clause_pos_mask[i]: bitmask of variables in clause i that appear positively
    # clause_neg_mask[i]: bitmask of variables in clause i that appear negatively
    clause_pos_mask = [0] * m
    clause_neg_mask = [0] * m
    for i, c in enumerate(clauses):
        for l in c:
            if l > 0:
                clause_pos_mask[i] |= (1 << (l - 1))
            else:
                clause_neg_mask[i] |= (1 << (-l - 1))

    def assign_to_int(assign):
        v = 0
        for i, b in enumerate(assign):
            if b:
                v |= (1 << i)
        return v

    def int_to_assign(v):
        return [bool(v & (1 << i)) for i in range(n)]

    def compute_unsat_mask(v):
        unsat = 0
        for i in range(m):
            # Clause i is satisfied if (v & pos_mask) != 0 OR ((~v) & neg_mask) != 0
            # ~v needs to be masked to n bits, but since pos/neg masks only have n bits, 
            # (~v) & neg_mask works correctly in Python's infinite bit logic 
            # as long as we don't care about bits > n, which we don't.
            if not ((v & clause_pos_mask[i]) or ((~v) & clause_neg_mask[i])):
                unsat |= (1 << i)
        return unsat

    def get_unsat_indices(unsat_mask):
        indices = []
        while unsat_mask:
            lsb = unsat_mask & (-unsat_mask)
            indices.append(lsb.bit_length() - 1)
            unsat_mask ^= lsb
        return indices

    def calculate_gain(v, var, unsat_mask):
        """
        Calculate the net change in satisfied clauses if we flip variable 'var' in assignment 'v'.
        Returns: (gain, new_unsat_mask)
        """
        # Clauses affected by flipping 'var'
        affected = var_pos_clauses[var] | var_neg_clauses[var]
        
        # Clauses that are currently unsatisfied and affected
        unsat_affected = affected & unsat_mask
        # Clauses that are currently satisfied and affected
        # We need sat_mask, but we can derive it: full_mask ^ unsat_mask
        # However, we only care about the change.
        # Let's iterate through affected clauses.
        
        gain = 0
        new_unsat = unsat_mask
        
        aff = affected
        while aff:
            lsb = aff & (-aff)
            ci = lsb.bit_length() - 1
            aff ^= lsb
            
            is_unsat = (unsat_mask >> ci) & 1
            is_pos_var = (var_pos_clauses[var] >> ci) & 1
            
            # Check current status of other literals in this clause
            # If var is positive in clause ci:
            #   Currently False (since it's in unsat_mask? No, unsat_mask means clause is unsat).
            #   Wait, if clause is unsat, ALL literals are False.
            #   If var is positive, it is False.
            #   If var is negative, it is True (because ~var is False? No).
            #   Let's be precise.
            
            # Assignment v: bit 'var' is b = (v >> var) & 1.
            # Literal is positive: value is b.
            # Literal is negative: value is 1-b.
            
            # Clause is unsat => all literals are False.
            # Clause is sat => at least one literal is True.
            
            # We are flipping 'var'.
            # If clause was UNSAT:
            #   All literals were False.
            #   After flip, 'var' literal becomes True.
            #   So the clause becomes SATISFIED.
            #   Gain +1.
            #   (No need to check other literals, because if they were True, clause would have been Sat already).
            #   Wait, this is only true if the clause was UNSAT.
            #   If clause was UNSAT, all literals are False. Flipping any literal in it makes it True.
            #   So any flip of a variable in an unsat clause SATISFIES that clause.
            
            # If clause was SAT:
            #   At least one literal was True.
            #   We are flipping 'var'.
            #   If 'var' was the ONLY True literal, it becomes False, and clause becomes UNSAT.
            #   If 'var' was False, it becomes True (clause stays Sat).
            #   If 'var' was True, but there were other True literals, it becomes False (clause stays Sat).
            
            if is_unsat:
                # Clause was unsat. Flipping any variable in it satisfies it.
                gain += 1
                new_unsat &= ~(1 << ci)
            else:
                # Clause was sat.
                # Check if flipping 'var' breaks it.
                # This happens if 'var' was the ONLY true literal.
                
                # Determine if 'var' is currently True in this clause context.
                # If var is positive: lit_val = (v >> var) & 1
                # If var is negative: lit_val = 1 - ((v >> var) & 1)
                
                if is_pos_var:
                    var_val = (v >> var) & 1
                else:
                    var_val = 1 - ((v >> var) & 1)
                    
                if var_val == 1:
                    # 'var' is currently True.
                    # Check if any OTHER literal is True.
                    # If no other literal is True, this clause will become UNSAT.
                    # Gain -1.
                    
                    # Check other literals
                    other_true = False
                    for var2, is_pos2 in clause_vars[ci]:
                        if var2 == var:
                            continue
                        if is_pos2:
                            val2 = (v >> var2) & 1
                        else:
                            val2 = 1 - ((v >> var2) & 1)
                        if val2 == 1:
                            other_true = True
                            break
                    
                    if not other_true:
                        gain -= 1
                        new_unsat |= (1 << ci)
                # If var_val == 0, it becomes 1, clause stays Sat. No change.
                
        return gain, new_unsat

    best_assign = None
    best_sat = -1

    # Phase 0: Random Initializations
    for attempt_seed in range(16):
        r = random.Random(attempt_seed * 31 + 7)
        assign = [r.random() < 0.5 for _ in range(n)]
        v = assign_to_int(assign)
        unsat_mask = compute_unsat_mask(v)
        s = m - unsat_mask.bit_count()
        if s > best_sat:
            best_sat = s
            best_assign = assign
        if s == m:
            return assign
        if time.time() > deadline:
            return best_assign if best_assign else [False] * n

    # Main Search Loop: Greedy with Escape
    # We will run multiple "runs" from random starts, each doing local search.
    max_restarts = 1000
    steps_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        r = random.Random(restart * 7919 + 13)
        assign = [r.random() < 0.5 for _ in range(n)]
        v = assign_to_int(assign)
        unsat_mask = compute_unsat_mask(v)
        cur_sat = m - unsat_mask.bit_count()
        
        if cur_sat > best_sat:
            best_sat = cur_sat
            best_assign = int_to_assign(v)
        if cur_sat == m:
            return best_assign

        local_best_sat = cur_sat
        local_best_v = v
        
        # Adaptive step limit: if we are close to solution, search longer?
        # Or just fixed budget per restart.
        for step in range(steps_per_restart):
            if time.time() > deadline:
                return best_assign if best_assign else [False] * n
            
            if unsat_mask == 0:
                best_assign = int_to_assign(v)
                best_sat = m
                return best_assign

            # Identify candidate variables: those in unsatisfied clauses
            unsat_indices = get_unsat_indices(unsat_mask)
            candidate_vars = set()
            for ci in unsat_indices:
                for var, _ in clause_vars[ci]:
                    candidate_vars.add(var)
            
            if not candidate_vars:
                break # Should not happen if unsat_mask != 0

            # Greedy: Find best gain
            best_gain = -100
            best_var = -1
            best_new_unsat = unsat_mask
            
            # To speed up, we might not need to check ALL candidates if we find a +1 gain early?
            # No, we need max gain. But gains are small.
            # Let's just iterate.
            for var in candidate_vars:
                gain, new_unsat = calculate_gain(v, var, unsat_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_var = var
                    best_new_unsat = new_unsat
                # Early exit if we found a gain of 3? Max possible gain for one flip in 3-SAT is 3?
                # If var is in 3 unsat clauses, and none are broken, gain is 3.
                if best_gain >= 3:
                    break

            # Decision:
            # If best_gain > 0, take it.
            # If best_gain == 0, take it with some prob (neutral move), else random.
            # If best_gain < 0, we are at a local min.
            #   Escape strategy: 
            #   1. If local_best_sat hasn't improved for a while, restart?
            #   2. Or take a random move from candidates.
            
            if best_gain > 0:
                v ^= (1 << best_var)
                unsat_mask = best_new_unsat
                cur_sat = m - unsat_mask.bit_count()
                
                if cur_sat > local_best_sat:
                    local_best_sat = cur_sat
                    local_best_v = v
                    if cur_sat > best_sat:
                        best_sat = cur_sat
                        best_assign = int_to_assign(v)
            
            elif best_gain == 0:
                # Neutral move. Accept with 50% prob, else random.
                if r.random() < 0.5:
                    v ^= (1 << best_var)
                    unsat_mask = best_new_unsat
                    cur_sat = m - unsat_mask.bit_count()
                else:
                    var = r.choice(list(candidate_vars))
                    v ^= (1 << var)
                    # Recalculate unsat mask for random move
                    # Optimization: we can update incrementally, but calculate_gain already did for best_var.
                    # For random var, we just recompute or use a helper.
                    # Let's recompute unsat_mask for simplicity/safety, or use the same logic.
                    # Actually, let's just use the general update.
                    # To avoid code duplication, let's just recompute unsat_mask if random.
                    # But recompute is O(m). m is small.
                    unsat_mask = compute_unsat_mask(v)
                    cur_sat = m - unsat_mask.bit_count()
                    
            else: # best_gain < 0
                # Local minimum. Escape.
                # Randomly pick a variable from candidates.
                var = r.choice(list(candidate_vars))
                v ^= (1 << var)
                unsat_mask = compute_unsat_mask(v)
                cur_sat = m - unsat_mask.bit_count()
                
                # If we drop too much, maybe restart?
                # For now, just continue. The random walk will eventually find a way out.
                
                # Track local best
                if cur_sat > local_best_sat:
                    local_best_sat = cur_sat
                    local_best_v = v
                    if cur_sat > best_sat:
                        best_sat = cur_sat
                        best_assign = int_to_assign(v)

        # End of restart loop
        if local_best_sat > best_sat:
            best_sat = local_best_sat
            best_assign = int_to_assign(local_best_v)
            
        if best_sat == m:
            return best_assign

    if best_assign is None:
        return [False] * n
    return best_assign
