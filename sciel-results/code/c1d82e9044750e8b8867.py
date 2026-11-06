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
        for _ in range(3 * n + 10):
            unit = np.where(num_true == 1)[0]
            if unit.size == 0:
                break
            progress = False
            for c in unit:
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

    def smooth_var_score(assignment, num_true, v):
        """Smooth surrogate score for flipping variable v.
        Uses soft penalty based on how many unsatisfied clauses v appears in,
        and whether flipping would help or hurt."""
        c_list = var_clauses_np[v]
        if c_list.size == 0:
            return 0.0
        s = var_signs[v]
        current_val = assignment[v]
        new_val = 1 - current_val
        
        # For each clause containing v:
        # - If clause is unsat (num_true==0) and v's new value satisfies it: +1
        # - If clause is sat with exactly 1 (num_true==1) and v currently satisfies it: -1 (would break)
        # - Otherwise: 0
        unsat = (num_true[c_list] == 0)
        v_satisfies_new = (new_val == s)
        gain = unsat & v_satisfies_new
        
        was_v_sat = (current_val == s)
        critical = (num_true[c_list] == 1)
        loss = was_v_sat & critical
        
        # Smooth score: use sigmoid-like weighting to avoid sharp discontinuities
        # Scale by 0.5 to keep values in a reasonable range
        score = 0.5 * (gain.astype(np.float64).sum() - loss.astype(np.float64).sum())
        
        # Add small bonus for variables in more unsatisfied clauses (global pressure)
        if unsat.any():
            score += 0.1 * unsat.sum()
            
        return float(score)

    def targeted_pick(c, num_true, var_counts, rng, assignment):
        """Pick a variable to flip from unsatisfied clause c using smooth scoring."""
        cands = var_idx[c]
        
        # Compute smooth scores for each candidate
        scores = np.array([smooth_var_score(assignment, num_true, int(v)) 
                          for v in cands], dtype=np.float64)
        
        # Combine with clause frequency if available
        if var_counts is not None:
            freq_weight = var_counts[cands].astype(np.float64) * 0.3
            w = scores + freq_weight + 0.05
        else:
            w = scores + 0.05
        
        # Ensure positive weights for sampling
        w = np.maximum(w, 0.01)
        total = w.sum()
        if total <= 0:
            return int(cands[rng.integers(3)])
        
        # Weighted random selection
        r = rng.random() * total
        cum = 0.0
        for k in range(3):
            cum += w[k]
            if r <= cum:
                return int(cands[k])
        return int(cands[2])

    rng = np.random.default_rng(0)
    best_unsat = None
    best_assign = None

    max_restarts = 100000
    base_steps = 15000

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

        # Adaptive steps: more steps when close to solution
        current_unsat = int((num_true == 0).sum())
        if best_unsat is not None:
            if current_unsat < best_unsat * 0.5:
                max_steps = base_steps * 3
            elif current_unsat < best_unsat:
                max_steps = base_steps * 2
            else:
                max_steps = base_steps
        else:
            max_steps = base_steps

        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break

            unsat_mask = (num_true == 0)
            if not unsat_mask.any():
                return assignment.astype(bool).tolist()

            unsat_count = int(unsat_mask.sum())
            if best_unsat is None or unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assignment.copy()

            if step % 8 == 0:
                var_counts = var_unsat_counts(num_true)
            else:
                var_counts = None

            unsat_idx = np.where(unsat_mask)[0]
            c = int(unsat_idx[rng.integers(unsat_idx.size)])

            # Targeted repair: only flip variables in the unsatisfied clause
            v = targeted_pick(c, num_true, var_counts, rng, assignment)
            v = int(v)

            new = 1 - int(assignment[v])
            assignment[v] = new
            update_var(assignment, num_true, v, new)

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
        
        # Targeted repair polish: only flip variables in unsatisfied clauses
        # with greedy acceptance (no annealing) for faster convergence
        for _ in range(300000):
            if time.time() > polish_deadline:
                break
            unsat_mask = (num_true == 0)
            if not unsat_mask.any():
                return assignment.astype(bool).tolist()
            
            unsat_count = int(unsat_mask.sum())
            if best_unsat is None or unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assignment.copy()
            
            # Recompute variable unsat counts every few iterations for efficiency
            if _ % 10 == 0:
                var_counts = var_unsat_counts(num_true)
            else:
                var_counts = None
            
            unsat_idx = np.where(unsat_mask)[0]
            c = int(unsat_idx[rng.integers(unsat_idx.size)])
            v = targeted_pick(c, num_true, var_counts, rng, assignment)
            v = int(v)
            
            new = 1 - int(assignment[v])
            old_unsat = unsat_count
            assignment[v] = new
            update_var(assignment, num_true, v, new)
            new_unsat = int((num_true == 0).sum())
            
            # Greedy acceptance: only accept if it doesn't make things worse
            if new_unsat > old_unsat:
                # Reject: undo
                assignment[v] = 1 - new
                update_var(assignment, num_true, v, 1 - new)

    if best_assign is not None:
        return best_assign.astype(bool).tolist()

    return init_assignment().astype(bool).tolist()
