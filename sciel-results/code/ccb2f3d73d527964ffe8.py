import time
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    start_time = time.time()
    deadline = start_time + 325.0   # safety margin before 330s kill

    # ---- Preprocessing: vectorize clause literals ----
    # lits[m,3] = signed literals (1-based). lit > 0 means positive.
    lits = np.empty((m, 3), dtype=np.int32)
    for i, c in enumerate(clauses):
        for j, l in enumerate(c):
            lits[i, j] = l

    # var index (0-based) and sign (1 = positive literal, 0 = negative)
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

    def init_assignment():
        return (np.random.rand(n) < 0.5).astype(np.int8)

    def init_num_true(assignment):
        # satisfied iff assignment[var] == sign
        satisfied = (assignment[var_idx] == sign)
        return satisfied.sum(axis=1).astype(np.int32)

    def check_solved(num_true):
        return num_true.min() > 0

    def unit_propagate(assignment, num_true):
        # Iteratively set variables that appear in a unit clause to satisfy it.
        # Returns (assignment, num_true, changed_any)
        changed = False
        for _ in range(3 * n + 10):
            unit = np.where(num_true == 1)[0]
            if unit.size == 0:
                break
            progress = False
            for c in unit:
                # find the unique true literal position
                sat_pos = np.where((assignment[var_idx[c]] == sign[c]))[0]
                if sat_pos.size == 0:
                    continue
                p = int(sat_pos[0])
                v = int(var_idx[c, p])
                target = int(sign[c, p])
                if int(assignment[v]) == target:
                    continue
                # flip v to target
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

    def gain_flip(assignment, num_true, c_idx, v):
        # Number of currently-unsatisfied clauses (other than c_idx) that would
        # become satisfied if we flip v.
        c_list = var_clauses_np[v]
        p_list = var_pos_np[v]
        if c_list.size == 0:
            return 0
        s = sign[c_list, p_list]
        now_sat = (assignment[v] == s)
        unsat_mask = (num_true[c_list] == 0)
        # literal currently false == not now_sat
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

        # unit propagation on start
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

            # pick random unsatisfied clause
            unsat_idx = np.where(unsat_mask)[0]
            c = int(unsat_idx[rng.integers(unsat_idx.size)])

            # pick a literal in the clause
            p = int(rng.integers(3))
            v = int(var_idx[c, p])

            # compute gain for this variable
            g = gain_flip(assignment, num_true, c, v)

            # decide flip with probability proportional to gain (WalkSAT)
            # Use p = 1 - 0.5^(gain) style or simple: flip if gain high or random
            # Standard: flip with prob 1/(1+gain) else random among clause vars.
            # Here we use: with probability p_w flip best-gain var, else random.
            # Simpler robust heuristic: flip with prob min(1, gain/3 + 0.2)
            flip_prob = min(1.0, 0.15 + 0.3 * g)
            do_flip = rng.random() < flip_prob
            if not do_flip:
                # random variable from the clause
                p = int(rng.integers(3))
                v = int(var_idx[c, p])

            # perform flip
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

    # fallback: last random
    return init_assignment().astype(bool).tolist()
