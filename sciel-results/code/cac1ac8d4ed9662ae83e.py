import time
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    start_time = time.time()
    deadline = start_time + 325.0   # safety margin before 330s kill

    # ---- Preprocessing: vectorize clause literals ----
    lits = np.empty((m, 3), dtype=np.int32)
    for i, c in enumerate(clauses):
        for j, l in enumerate(c):
            lits[i, j] = l

    var_idx = np.abs(lits) - 1            # [m,3]
    sign = (lits > 0).astype(np.int32)    # [m,3]  1 if positive

    # ---- Build variable -> (clauses, positions) lookup, sorted by clause id ----
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

    # Variable frequency (degree) for weighted selection
    var_freq = np.array([len(var_clauses[v]) for v in range(n)], dtype=np.float64)
    max_freq = var_freq.max() if n > 0 else 1.0
    if max_freq > 0:
        var_freq_norm = var_freq / max_freq
    else:
        var_freq_norm = np.ones(n, dtype=np.float64)

    def init_assignment():
        return (np.random.rand(n) < 0.5).astype(np.int8)

    def init_num_true(assignment):
        satisfied = (assignment[var_idx] == sign)
        return satisfied.sum(axis=1).astype(np.int32)

    def check_solved(num_true):
        return num_true.min() > 0

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
                c_list = var_clauses_np[v]
                p_list = var_pos_np[v]
                if c_list.size == 0:
                    continue
                s = sign[c_list, p_list]
                now_sat = (assignment[v] == s)
                delta = (2 * now_sat.astype(np.int32)) - 1
                num_true[c_list] += delta
            if not progress:
                break
        return assignment, num_true, changed

    def gain_flip(assignment, num_true, v):
        c_list = var_clauses_np[v]
        p_list = var_pos_np[v]
        if c_list.size == 0:
            return 0
        s = sign[c_list, p_list]
        now_sat = (assignment[v] == s)
        unsat_mask = (num_true[c_list] == 0)
        gain = int(np.sum(unsat_mask & ~now_sat))
        return gain

    rng = np.random.default_rng(0)
    best_unsat = None
    best_assign = None

    max_restarts = 100000
    max_steps = 15000

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        assignment = init_assignment()
        num_true = init_num_true(assignment)
        if check_solved(num_true):
            return assignment.astype(bool).tolist()

        assignment, num_true, _ = unit_propagate(assignment, num_true)
        if check_solved(num_true):
            return assignment.astype(bool).tolist()

        for step in range(max_steps):
            if time.time() > deadline:
                if best_unsat is None or num_true.min() < best_unsat:
                    best_unsat = int(num_true.min())
                    best_assign = assignment.copy()
                break

            unsat_mask = (num_true == 0)
            if not unsat_mask.any():
                return assignment.astype(bool).tolist()

            unsat_count = int(unsat_mask.sum())
            if best_unsat is None or unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assignment.copy()

            # Pick random unsatisfied clause
            unsat_idx = np.where(unsat_mask)[0]
            c = int(unsat_idx[rng.integers(unsat_idx.size)])

            # Compute gains for all 3 variables in the clause
            gains = np.empty(3, dtype=np.int32)
            vars_in_clause = np.empty(3, dtype=np.int32)
            for p in range(3):
                v = int(var_idx[c, p])
                vars_in_clause[p] = v
                gains[p] = gain_flip(assignment, num_true, v)

            # Weighted selection: combine gain with variable frequency
            # Higher gain and higher frequency => more likely to be selected
            weights = gains.astype(np.float64) + 0.5 * var_freq_norm[vars_in_clause]
            weights = weights / weights.sum()
            p = int(rng.choice(3, p=weights))
            v = int(vars_in_clause[p])

            # Perform flip
            old = int(assignment[v])
            new = 1 - old
            assignment[v] = new
            c_list = var_clauses_np[v]
            p_list = var_pos_np[v]
            if c_list.size > 0:
                s = sign[c_list, p_list]
                now_sat = (new == s)
                delta = (2 * now_sat.astype(np.int32)) - 1
                num_true[c_list] += delta

        if time.time() > deadline:
            break

    if best_assign is not None:
        return best_assign.astype(bool).tolist()

    return init_assignment().astype(bool).tolist()
