import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [True] * n

    start_time = time.time()
    deadline = start_time + 320.0   # safety margin before 330s kill

    # ---- Preprocessing: vectorize clause literals ----
    lits = np.empty((m, 3), dtype=np.int32)
    for i, c in enumerate(clauses):
        for j, l in enumerate(c):
            lits[i, j] = l

    var_idx = np.abs(lits) - 1              # [m,3] 0-based var
    sign = (lits > 0).astype(np.int32)      # [m,3] 1 if positive

    # ---- Variable -> (clauses, positions) lookup, sorted by clause id ----
    var_clauses = [[] for _ in range(n)]
    var_pos = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = int(var_idx[i, j])
            var_clauses[v].append(i)
            var_pos[v].append(j)

    for v in range(n):
        if var_clauses[v]:
            order = np.argsort(var_clauses[v], kind="stable")
            var_clauses[v] = [var_clauses[v][k] for k in order]
            var_pos[v] = [var_pos[v][k] for k in order]
    var_clauses_np = [np.array(vc, dtype=np.int32) if vc else np.empty(0, dtype=np.int32)
                      for vc in var_clauses]
    var_pos_np = [np.array(vp, dtype=np.int32) if vp else np.empty(0, dtype=np.int32)
                  for vp in var_pos]

    # Precompute, per variable, the literal sign of v within each incident clause
    var_signs = [np.array([sign[c, p] for c, p in zip(var_clauses[v], var_pos[v])],
                          dtype=np.int32) if var_clauses[v] else np.empty(0, dtype=np.int32)
                 for v in range(n)]

    # Precompute clause-level data for vectorized delta computation
    # For each clause c, precompute which variables it contains and their positions
    clause_vars = var_idx  # [m, 3]
    clause_signs = sign    # [m, 3]

    def init_assignment():
        return (np.random.rand(n) < 0.5).astype(np.int8)

    def init_num_true(assignment):
        # satisfied iff assignment[var] == sign
        satisfied = (assignment[var_idx] == sign)
        return satisfied.sum(axis=1).astype(np.int32)

    def check_solved(num_true):
        return num_true.min() > 0

    def update_var(assignment, num_true, v, new_val):
        """Flip/set variable v to new_val and update num_true for incident clauses."""
        c_list = var_clauses_np[v]
        p_list = var_pos_np[v]
        if c_list.size == 0:
            return
        s = sign[c_list, p_list]
        now_sat = (new_val == s)
        delta = (2 * now_sat.astype(np.int32)) - 1
        num_true[c_list] += delta

    def unit_propagate(assignment, num_true):
        changed = False
        # Use a worklist approach: repeatedly find unit clauses and propagate
        for _ in range(10 * n + 20):
            # Find all unit clauses (exactly one literal satisfied)
            unit = np.where(num_true == 1)[0]
            if unit.size == 0:
                break
            progress = False
            for c in unit:
                # Find which position in this clause is satisfied
                sat_pos = np.where((assignment[var_idx[c]] == sign[c]))[0]
                if sat_pos.size == 0:
                    continue
                p = int(sat_pos[0])
                v = int(var_idx[c, p])
                target = int(sign[c, p])
                if int(assignment[v]) == target:
                    continue
                assignment[v] = target
                changed = True
                progress = True
                update_var(assignment, num_true, v, target)
            if not progress:
                break
        return assignment, num_true, changed

    def var_unsat_counts(num_true):
        # number of unsatisfied clauses each variable appears in
        unsat = (num_true == 0)
        if not unsat.any():
            return None
        counts = np.zeros(n, dtype=np.int32)
        for c in np.where(unsat)[0]:
            counts[var_idx[c]] += 1
        return counts

    def compute_all_deltas(assignment, num_true):
        """
        Vectorized computation of delta for flipping each variable.
        Returns an array of shape [n] where delta[v] is the change in total
        unsatisfied clauses if we flip variable v.
        
        For variable v, flipping it affects all clauses containing v.
        For each such clause c at position p:
          - If assignment[v] == sign[c,p], the literal is currently satisfied.
            Flipping makes it unsatisfied: delta contribution = -1 for this clause.
          - If assignment[v] != sign[c,p], the literal is currently unsatisfied.
            Flipping makes it satisfied: delta contribution = +1 for this clause.
        
        But we only care about the net change in the NUMBER OF UNSATISFIED CLAUSES.
        A clause becomes satisfied if all its literals become satisfied.
        A clause becomes unsatisfied if any literal becomes unsatisfied.
        
        Simpler approach: compute the change in num_true for each clause, then
        determine which clauses transition between 0 and >0.
        
        Actually, let's use the direct approach:
        For each variable v, compute how many clauses it appears in where:
          - It's currently the ONLY satisfied literal (num_true==1 and this lit is sat)
            -> flipping makes clause unsat: -1 contribution
          - It's currently unsatisfied and all other lits are unsat (num_true==0)
            -> flipping may satisfy clause: +1 contribution (if it was the only unsat)
        
        This is complex. Let's use a simpler vectorized method:
        Compute new_num_true after flipping v, then count transitions.
        
        But that's O(m) per variable. Instead, let's precompute:
        For each variable v, precompute:
          - sat_clauses: clauses where v's literal is currently satisfied
          - unsat_clauses: clauses where v's literal is currently unsatisfied
        
        Then delta[v] = (number of clauses in unsat_clauses that have num_true==0 
                        and v is the only unsat literal) 
                      - (number of clauses in sat_clauses that have num_true==1 
                        and v is the only sat literal)
        
        Actually, the cleanest vectorized approach:
        For each variable v, the change in num_true for clause c (where v appears at pos p) is:
          +1 if assignment[v] != sign[c,p] (was unsat, now sat)
          -1 if assignment[v] == sign[c,p] (was sat, now unsat)
        
        So delta_num_true[c] for flipping v is:
          +1 if (assignment[v] != sign[c,p])
          -1 if (assignment[v] == sign[c,p])
        
        Then the change in total unsat count is:
          +1 for each clause c where num_true[c]==0 and delta_num_true[c]==+1 (becomes sat)
          -1 for each clause c where num_true[c]==1 and delta_num_true[c]==-1 (becomes unsat)
        
        Let's compute this efficiently using precomputed structures.
        """
        # For each variable, we need:
        # 1. The clauses it appears in (var_clauses_np[v])
        # 2. Whether its literal is currently satisfied in each such clause
        # 3. The current num_true of those clauses
        
        deltas = np.zeros(n, dtype=np.int32)
        
        for v in range(n):
            c_list = var_clauses_np[v]
            if c_list.size == 0:
                continue
            
            s = sign[c_list, var_pos_np[v]]  # sign of v's literal in each clause
            cur_val = assignment[v]
            
            # Which clauses have v's literal currently satisfied?
            v_sat_mask = (cur_val == s)  # bool array
            v_unsat_mask = ~v_sat_mask
            
            cur_num_true = num_true[c_list]
            
            # Clauses that will become satisfied: num_true==0 and v_unsat (v's lit becomes sat)
            becomes_sat = (cur_num_true == 0) & v_unsat_mask
            # Clauses that will become unsatisfied: num_true==1 and v_sat (v's lit becomes unsat)
            becomes_unsat = (cur_num_true == 1) & v_sat_mask
            
            deltas[v] = int(becomes_sat.sum()) - int(becomes_unsat.sum())
        
        return deltas

    def vectorized_pick(assignment, num_true, var_counts, rng):
        """
        Vectorized variable selection: compute deltas for all variables,
        then pick one that minimizes total unsat (i.e., maximizes delta).
        With some randomness for exploration.
        """
        deltas = compute_all_deltas(assignment, num_true)
        
        # Focus on variables that are in at least one unsatisfied clause
        unsat_mask = (num_true == 0)
        if not unsat_mask.any():
            return None
        
        # Only consider variables that appear in unsatisfied clauses
        unsat_vars = np.zeros(n, dtype=bool)
        for c in np.where(unsat_mask)[0]:
            unsat_vars[var_idx[c]] = True
        
        if not unsat_vars.any():
            return None
        
        # Among variables in unsat clauses, pick based on delta
        # Prefer variables with positive delta (reduce unsat count)
        # Add some randomness for exploration
        
        candidate_vars = np.where(unsat_vars)[0]
        candidate_deltas = deltas[candidate_vars]
        
        # Weight: prefer higher delta, but allow exploration
        # Use softmax-like weighting with temperature
        if var_counts is not None:
            vc = var_counts[candidate_vars].astype(np.float64) + 0.5
            weights = (candidate_deltas.astype(np.float64) + 1.0) * vc
        else:
            weights = candidate_deltas.astype(np.float64) + 1.0
        
        # Ensure non-negative weights
        weights = np.maximum(weights, 0.1)
        
        total_w = weights.sum()
        if total_w <= 0:
            return int(candidate_vars[rng.integers(candidate_vars.size)])
        
        r = rng.random() * total_w
        cum = 0.0
        for i in range(len(candidate_vars)):
            cum += weights[i]
            if r <= cum:
                return int(candidate_vars[i])
        return int(candidate_vars[-1])

    rng = np.random.default_rng(0)
    best_unsat = None
    best_assign = None

    # Adaptive restart: start with moderate steps, increase if making progress
    base_steps = 5000
    max_restarts = 100000
    current_steps = base_steps
    consecutive_failures = 0

    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break

        assignment = init_assignment()
        num_true = init_num_true(assignment)
        if check_solved(num_true):
            return assignment.astype(bool).tolist()

        assignment, num_true, _ = unit_propagate(assignment, num_true)
        if check_solved(num_true):
            return assignment.astype(bool).tolist()

        unsat_before = int((num_true == 0).sum())
        
        for step in range(current_steps):
            if time.time() > deadline - 5.0:
                break

            unsat_mask = (num_true == 0)
            if not unsat_mask.any():
                return assignment.astype(bool).tolist()

            unsat_count = int(unsat_mask.sum())
            if best_unsat is None or unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assignment.copy()

            # Update var_counts periodically
            if step % 10 == 0:
                var_counts = var_unsat_counts(num_true)
            else:
                var_counts = None

            # Use vectorized pick
            v = vectorized_pick(assignment, num_true, var_counts, rng)
            
            if v is None:
                # Fallback: random variable from an unsat clause
                unsat_idx = np.where(unsat_mask)[0]
                c = int(unsat_idx[rng.integers(unsat_idx.size)])
                p = int(rng.integers(3))
                v = int(var_idx[c, p])
            
            new = 1 - int(assignment[v])
            assignment[v] = new
            update_var(assignment, num_true, v, new)

        unsat_after = int((num_true == 0).sum())
        
        # Adaptive restart: if we made progress, keep going; else restart
        if unsat_after < unsat_before:
            consecutive_failures = 0
            current_steps = min(current_steps * 2, 30000)
        else:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                current_steps = base_steps
                consecutive_failures = 0

        if time.time() > deadline - 5.0:
            break

    # ---- Final high-precision polish pass on the best assignment found ----
    if best_assign is not None:
        assignment = best_assign.copy()
        num_true = init_num_true(assignment)
        assignment, num_true, _ = unit_propagate(assignment, num_true)
        if check_solved(num_true):
            return assignment.astype(bool).tolist()

        polish_deadline = deadline - 1.0
        var_counts = var_unsat_counts(num_true)
        
        # Use vectorized approach for polish
        for _ in range(500000):
            if time.time() > polish_deadline:
                break
            unsat_mask = (num_true == 0)
            if not unsat_mask.any():
                return assignment.astype(bool).tolist()
            unsat_count = int(unsat_mask.sum())
            if best_unsat is None or unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assignment.copy()
            
            # Update var_counts every few steps
            if _ % 5 == 0:
                var_counts = var_unsat_counts(num_true)
            
            v = vectorized_pick(assignment, num_true, var_counts, rng)
            
            if v is None:
                unsat_idx = np.where(unsat_mask)[0]
                c = int(unsat_idx[rng.integers(unsat_idx.size)])
                p = int(rng.integers(3))
                v = int(var_idx[c, p])
            
            new = 1 - int(assignment[v])
            assignment[v] = new
            update_var(assignment, num_true, v, new)

    if best_assign is not None:
        return best_assign.astype(bool).tolist()

    return init_assignment().astype(bool).tolist()
