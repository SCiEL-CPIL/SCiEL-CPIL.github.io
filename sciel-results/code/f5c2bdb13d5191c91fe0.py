"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    start_time = time.time()
    timeout = 315.0  # Leave margin for overhead (330s limit)
    
    m = len(clauses)
    if m == 0:
        return [True] * n
    
    # Preprocess clauses into numpy arrays for fast vectorized evaluation
    # clause_signs: shape (m, 3), values are +1 or -1
    # clause_vars: shape (m, 3), values are 0-based variable indices
    clause_signs = np.empty((m, 3), dtype=np.int8)
    clause_vars = np.empty((m, 3), dtype=np.int32)
    
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_signs[i, j] = 1 if lit > 0 else -1
            clause_vars[i, j] = abs(lit) - 1
    
    # Build variable-to-clause incidence for fast counting
    # For each variable, track which clauses contain it (as positive or negative)
    var_pos_clauses = [[] for _ in range(n)]  # clauses where var appears positively
    var_neg_clauses = [[] for _ in range(n)]  # clauses where var appears negatively
    
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            if clause_signs[i, j] == 1:
                var_pos_clauses[v].append(i)
            else:
                var_neg_clauses[v].append(i)
    
    def time_left():
        return timeout - (time.time() - start_time)
    
    def evaluate(assignment):
        """Return number of satisfied clauses using vectorized ops."""
        # assignment: numpy array of bools, length n
        # For each clause, check if any literal is satisfied
        # lit_val = assignment[var] if sign==+1 else not assignment[var]
        # satisfied if any lit_val is True
        
        # Get assignment values for all clause literals
        # signs: +1 means we want assignment[var] to be True
        #         -1 means we want assignment[var] to be False
        vals = assignment[clause_vars]  # shape (m, 3)
        # A literal is satisfied if: (sign==1 AND val==True) OR (sign==-1 AND val==False)
        # = (sign * (val ? 1 : -1)) > 0  ... let's think differently
        # satisfied_literal = (clause_signs == 1) & vals | (clause_signs == -1) & ~vals
        sat_lits = ((clause_signs == 1) & vals) | ((clause_signs == -1) & (~vals))
        # A clause is satisfied if any literal in it is satisfied
        sat_clauses = sat_lits.any(axis=1)
        return sat_clauses.sum()
    
    def is_satisfied(assignment):
        return evaluate(assignment) == m
    
    # Build lookup structures for vectorized flip-impact computation
    # For each variable, precompute:
    #   var_incident_clauses[v]: list of clause indices containing v
    #   var_incident_signs[v]:    corresponding signs (+1/-1)
    # This lets us compute, for a given assignment, the gain of flipping v
    # by looking only at clauses incident to v.
    var_incident_clauses = [[] for _ in range(n)]
    var_incident_signs = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_incident_clauses[v].append(i)
            var_incident_signs[v].append(clause_signs[i, j])
    
    # Convert to numpy arrays with padding for vectorized ops
    # Max clauses per variable is small (typically < 20 for 150 vars, m~450)
    max_inc = max(len(l) for l in var_incident_clauses) if n > 0 else 1
    var_inc_clauses_np = np.zeros((n, max_inc), dtype=np.int32)
    var_inc_signs_np = np.zeros((n, max_inc), dtype=np.int8)
    var_inc_counts = np.zeros(n, dtype=np.int32)
    for v in range(n):
        cnt = len(var_incident_clauses[v])
        var_inc_counts[v] = cnt
        if cnt > 0:
            var_inc_clauses_np[v, :cnt] = var_incident_clauses[v]
            var_inc_signs_np[v, :cnt] = var_incident_signs[v]
    
    def compute_flip_gains(assignment, sat_clauses):
        """
        For each variable v, compute the gain of flipping v:
          gain = (# unsat clauses that become sat) - (# sat clauses that become unsat)
        
        A clause containing literal (v, s) becomes satisfied by flipping v
        iff: the clause is currently unsat AND no other literal in the clause
        is already satisfied.
        
        A clause containing literal (v, s) becomes unsatisfied by flipping v
        iff: the clause is currently sat AND the literal (v,s) is the ONLY
        satisfied literal in that clause.
        
        We use vectorized numpy ops over all variables simultaneously.
        """
        # assignment: (n,) bool
        # sat_clauses: (m,) bool
        
        # For each variable v and each incident clause ci:
        #   current_lit_val = assignment[v] if sign==+1 else not assignment[v]
        #   after flip:      not assignment[v] if sign==+1 else assignment[v]
        
        # Gather assignment values for all (v, incident_clause) pairs
        # var_inc_clauses_np: (n, max_inc)
        # We need assignment[var] for each v, but the clause tells us which var
        # Actually, var_inc_clauses_np[v, k] = clause index, and the var is v itself.
        # So we need to know the sign and check if other literals in that clause are sat.
        
        # Alternative approach: for each variable v, compute:
        #   pos_incident: clauses where v appears positively
        #   neg_incident: clauses where v appears negatively
        #
        # Flipping v from False to True:
        #   - Pos clauses: v becomes True. Clause becomes sat if it was unsat and
        #     no other literal was already sat.
        #   - Neg clauses: v becomes True. Clause may become unsat if v was the
        #     only sat literal.
        #
        # Flipping v from True to False:
        #   - Pos clauses: v becomes False. Clause may become unsat if v was the
        #     only sat literal.
        #   - Neg clauses: v becomes False. Clause becomes sat if it was unsat and
        #     no other literal was already sat.
        
        # For efficiency, precompute for each clause: number of satisfied literals
        # and which literals are satisfied.
        vals = assignment[clause_vars]  # (m, 3)
        lit_sat = ((clause_signs == 1) & vals) | ((clause_signs == -1) & (~vals))  # (m, 3)
        lit_sat_counts = lit_sat.sum(axis=1)  # (m,)
        
        # For each variable v, we need to know, for each incident clause ci:
        #   - Is ci currently sat? (sat_clauses[ci])
        #   - Is the literal (v, s) currently sat? (lit_sat[ci, j] where j is position of v in ci)
        #   - How many other literals in ci are sat? (lit_sat_counts[ci] - lit_sat[ci, j])
        
        # To vectorize over v, we need to map (v, ci) -> (ci, j) where j is v's position in ci.
        # Precompute: for each variable v and each incident slot k, store (clause_idx, pos_in_clause)
        var_inc_pos = np.zeros((n, max_inc), dtype=np.int8)  # position of v in the clause (0,1,2)
        for v in range(n):
            cnt = var_inc_counts[v]
            if cnt > 0:
                # Find positions
                for k in range(cnt):
                    ci = var_incident_clauses[v][k]
                    # Find which position j in clause ci has variable v
                    for j in range(3):
                        if clause_vars[ci, j] == v:
                            var_inc_pos[v, k] = j
                            break
        
        # Now vectorized computation:
        # For variable v, incident clause ci at slot k, position j:
        #   lit_sat_for_v = lit_sat[ci, j]
        #   other_sat = lit_sat_counts[ci] - lit_sat[ci, j]
        
        # Gather: (n, max_inc) arrays
        ci_arr = var_inc_clauses_np  # (n, max_inc)
        pos_arr = var_inc_pos  # (n, max_inc)
        sign_arr = var_inc_signs_np  # (n, max_inc)
        cnt_arr = var_inc_counts  # (n,)
        
        # lit_sat for (v, ci, pos): shape (n, max_inc)
        # We need lit_sat[ci, pos] for each (v, k)
        # lit_sat is (m, 3), so lit_sat[ci_arr, pos_arr] gives (n, max_inc)
        lit_sat_v = lit_sat[ci_arr, pos_arr]  # (n, max_inc)
        
        # other_sat = lit_sat_counts[ci] - lit_sat[ci, pos]
        other_sat = lit_sat_counts[ci_arr] - lit_sat_v  # (n, max_inc)
        
        # sat_clauses for incident clauses
        ci_sat = sat_clauses[ci_arr]  # (n, max_inc)
        
        # Create mask for valid entries
        mask = np.arange(max_inc)[None, :] < cnt_arr[:, None]  # (n, max_inc)
        
        # Gain computation:
        # For variable v with current value cur_val:
        #   If cur_val == False (flipping to True):
        #     +1 for each pos clause that is unsat and other_sat == 0
        #     -1 for each neg clause that is sat and lit_sat_v == True and other_sat == 0
        #   If cur_val == True (flipping to False):
        #     -1 for each pos clause that is sat and lit_sat_v == True and other_sat == 0
        #     +1 for each neg clause that is unsat and other_sat == 0
        
        cur_val = assignment  # (n,)
        
        # For pos clauses (sign == +1):
        is_pos = (sign_arr == 1) & mask  # (n, max_inc)
        is_neg = (sign_arr == -1) & mask  # (n, max_inc)
        
        # Unsatisfied clauses with no other sat literal
        unsat_no_other = (~ci_sat) & (other_sat == 0) & mask  # (n, max_inc)
        # Satisfied clauses where this literal is the only sat literal
        sat_only_this = ci_sat & lit_sat_v & (other_sat == 0) & mask  # (n, max_inc)
        
        # Flipping False -> True:
        #   +1: pos & unsat_no_other
        #   -1: neg & sat_only_this
        gain_false_to_true = ((is_pos & unsat_no_other).sum(axis=1) - 
                              (is_neg & sat_only_this).sum(axis=1))
        
        # Flipping True -> False:
        #   -1: pos & sat_only_this
        #   +1: neg & unsat_no_other
        gain_true_to_false = ((is_neg & unsat_no_other).sum(axis=1) - 
                              (is_pos & sat_only_this).sum(axis=1))
        
        # Select based on current value
        # cur_val == True -> use gain_true_to_false
        # cur_val == False -> use gain_false_to_true
        gains = np.where(cur_val, gain_true_to_false, gain_false_to_true)
        
        return gains  # (n,)
    
    # Restart-based WalkSAT with adaptive parameters
    best_assignment = None
    best_score = -1
    
    restart = 0
    while time_left() > 2.0:
        restart += 1
        
        # Random initial assignment
        rng_seed = restart * 12345 + 67890
        rng = np.random.RandomState(rng_seed)
        assignment = rng.rand(n) < 0.5
        
        score = evaluate(assignment)
        if score > best_score:
            best_score = score
            best_assignment = assignment.copy()
        
        if score == m:
            return [bool(x) for x in assignment]
        
        # WalkSAT loop
        flip_budget = min(5000 + restart * 500, 50000)
        step = 0
        while step < flip_budget and time_left() > 1.0:
            step += 1
            
            # Check if satisfied
            if score == m:
                return [bool(x) for x in assignment]
            
            # Find unsatisfied clauses
            vals = assignment[clause_vars]
            sat_lits = ((clause_signs == 1) & vals) | ((clause_signs == -1) & (~vals))
            sat_clauses = sat_lits.any(axis=1)
            unsat_idx = np.where(~sat_clauses)[0]
            
            if len(unsat_idx) == 0:
                return [bool(x) for x in assignment]
            
            # Pick a random unsatisfied clause
            clause_idx = unsat_idx[rng.randint(len(unsat_idx))]
            clause_lits_vars = clause_vars[clause_idx]
            
            # Compute flip gains for ALL variables (vectorized)
            gains = compute_flip_gains(assignment, sat_clauses)
            
            # Candidate variables: the 3 literals in the chosen clause
            # Use score-based selection among candidates
            candidate_vars = clause_lits_vars  # (3,)
            candidate_gains = gains[candidate_vars]  # (3,)
            
            # Weighted selection: prefer higher gain, but allow random exploration
            # Use softmax-like weighting for smooth selection
            # Normalize gains to get probabilities
            max_gain = np.max(candidate_gains)
            exp_gains = np.exp((candidate_gains - max_gain) * 2.0)  # temperature-scaled
            probs = exp_gains / exp_gains.sum()
            
            # With some probability, pick randomly to escape local optima
            if rng.rand() < 0.15:
                best_var = candidate_vars[rng.randint(3)]
            else:
                best_var = candidate_vars[rng.choice(3, p=probs)]
            
            # Flip
            assignment[best_var] = not assignment[best_var]
            
            # Update score
            score = evaluate(assignment)
            if score > best_score:
                best_score = score
                best_assignment = assignment.copy()
    
    # Return best found
    if best_assignment is not None:
        return [bool(x) for x in best_assignment]
    return [bool(x) for x in assignment]
# EVOLVE-BLOCK-END
