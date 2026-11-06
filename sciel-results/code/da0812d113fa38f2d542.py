"""Planted 3-SAT solver (50 vars): optimized WalkSAT + planted-aware init + exact polish."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)

    # Deadline: leave a safety margin from the 330s kill
    deadline = time.time() + 325.0

    # Precompute structures
    # clause_vars[i]: list of (var_index, sign) where sign=+1 if lit>0 else -1
    clause_vars = []
    for clause in clauses:
        cv = []
        for lit in clause:
            v = abs(lit) - 1
            cv.append((v, 1 if lit > 0 else -1))
        clause_vars.append(cv)

    # var_to_clauses[v]: list of (clause_idx, sign)
    var_to_clauses = [[] for _ in range(n)]
    for i, cv in enumerate(clause_vars):
        for v, s in cv:
            var_to_clauses[v].append((i, s))

    def is_satisfied(assignment):
        for i in range(m):
            ok = False
            for v, s in clause_vars[i]:
                if (assignment[v] and s > 0) or (not assignment[v] and s < 0):
                    ok = True
                    break
            if not ok:
                return False
        return True

    def init_planted():
        """Heuristic: set each var to satisfy its most-frequent literal across its clauses."""
        # Count, per variable, how many times it appears positively vs negatively
        pos = [0] * n
        neg = [0] * n
        for v, s in [(v, s) for cv in clause_vars for (v, s) in cv]:
            if s > 0:
                pos[v] += 1
            else:
                neg[v] += 1
        a = [pos[v] >= neg[v] for v in range(n)]
        return a

    def walk_sat(assignment, max_steps, rng, p, use_weights, p_decay):
        """
        Optimized WalkSAT with incremental updates, O(1) unsat selection,
        and optional clause weighting.
        """
        clause_sat_count = [0] * m
        unsat = [False] * m
        unsat_list = []  # explicit list of unsat clause indices
        unsat_in_list = [False] * m
        unsat_count = 0

        for i in range(m):
            count = 0
            for v, s in clause_vars[i]:
                if (assignment[v] and s > 0) or (not assignment[v] and s < 0):
                    count += 1
            clause_sat_count[i] = count
            if count == 0:
                unsat[i] = True
                unsat_in_list[i] = True
                unsat_list.append(i)
                unsat_count += 1

        if unsat_count == 0:
            return True

        # Clause weights for biasing selection toward persistent conflicts
        w = [1.0] * m if use_weights else None

        # Local helper to pick an unsat clause index
        def pick_unsat():
            if use_weights:
                # Weighted selection over unsat_list
                total = 0.0
                for ci in unsat_list:
                    total += w[ci]
                if total <= 0.0:
                    return unsat_list[rng.randrange(len(unsat_list))]
                r = rng.random() * total
                acc = 0.0
                for ci in unsat_list:
                    acc += w[ci]
                    if acc >= r:
                        return ci
                return unsat_list[-1]
            else:
                return unsat_list[rng.randrange(len(unsat_list))]

        # Current greediness
        cur_p = p

        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                return False

            if unsat_count == 0:
                return True

            # Periodically decay greediness to escape local minima
            if p_decay and step % 500 == 0 and step > 0:
                cur_p = max(0.1, cur_p * 0.85)

            c_idx = pick_unsat()
            cv = clause_vars[c_idx]

            # Evaluate flips for the 3 literals in this clause
            best_gain = -1000
            best_var = -1
            flip_vars = []

            for v, s in cv:
                gain = 1  # flipping v satisfies lit in c_idx
                for (ci2, s2) in var_to_clauses[v]:
                    if ci2 == c_idx:
                        continue
                    # Is clause ci2 currently satisfied by v?
                    is_sat2 = (assignment[v] and s2 > 0) or (not assignment[v] and s2 < 0)
                    if is_sat2:
                        if clause_sat_count[ci2] == 1:
                            gain -= 1
                    else:
                        if clause_sat_count[ci2] == 0:
                            gain += 1
                flip_vars.append(v)
                if gain > best_gain:
                    best_gain = gain
                    best_var = v

            # Select variable to flip
            if best_var == -1:
                var = flip_vars[rng.randrange(len(flip_vars))]
            elif rng.random() < cur_p:
                var = flip_vars[rng.randrange(len(flip_vars))]
            else:
                var = best_var

            # Apply flip
            assignment[var] = not assignment[var]

            # Update clause_sat_count and unsat tracking
            for (ci2, s2) in var_to_clauses[var]:
                old_count = clause_sat_count[ci2]
                new_is_sat = (assignment[var] and s2 > 0) or (not assignment[var] and s2 < 0)
                old_is_sat = not new_is_sat
                if new_is_sat and not old_is_sat:
                    clause_sat_count[ci2] = old_count + 1
                    if old_count == 0:
                        # clause became satisfied
                        unsat[ci2] = False
                        unsat_in_list[ci2] = False
                        unsat_count -= 1
                        # remove from unsat_list (swap-pop)
                        try:
                            pos = unsat_list.index(ci2)
                        except ValueError:
                            pos = -1
                        if pos >= 0:
                            last = unsat_list[-1]
                            unsat_list[pos] = last
                            unsat_in_list[last] = True
                            unsat_list.pop()
                elif not new_is_sat and old_is_sat:
                    clause_sat_count[ci2] = old_count - 1
                    if old_count == 1:
                        # clause became unsatisfied
                        unsat[ci2] = True
                        if not unsat_in_list[ci2]:
                            unsat_in_list[ci2] = True
                            unsat_list.append(ci2)
                        unsat_count += 1

            # Update clause weights: boost unsat, decay sat
            if use_weights:
                for ci in unsat_list:
                    w[ci] = w[ci] * 1.3 + 1.0
                # decay satisfied clauses (only a sample to save time)
                if step % 10 == 0:
                    for i in range(m):
                        if not unsat[i]:
                            w[i] = w[i] * 0.9
                # Periodic renormalization to avoid overflow
                if step % 300 == 0 and step > 0:
                    mx = 0.0
                    for i in range(m):
                        if w[i] > mx:
                            mx = w[i]
                    if mx > 1e6:
                        for i in range(m):
                            w[i] /= mx

        return unsat_count == 0

    def polish(assignment, max_steps, rng):
        """
        Final exact local polish: only accept non-worsening moves (gain >= 0),
        with a small random perturbation to escape plateaus.
        """
        clause_sat_count = [0] * m
        unsat_count = 0
        for i in range(m):
            count = 0
            for v, s in clause_vars[i]:
                if (assignment[v] and s > 0) or (not assignment[v] and s < 0):
                    count += 1
            clause_sat_count[i] = count
            if count == 0:
                unsat_count += 1

        if unsat_count == 0:
            return True

        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                return False
            if unsat_count == 0:
                return True

            # Pick a random unsat clause
            for _ in range(20):
                ci = rng.randrange(m)
                if clause_sat_count[ci] == 0:
                    break
            else:
                ci = rng.randrange(m)
                while clause_sat_count[ci] != 0:
                    ci = rng.randrange(m)

            cv = clause_vars[ci]
            best_gain = -1000
            best_var = -1
            flip_vars = []
            for v, s in cv:
                gain = 1
                for (ci2, s2) in var_to_clauses[v]:
                    if ci2 == ci:
                        continue
                    is_sat2 = (assignment[v] and s2 > 0) or (not assignment[v] and s2 < 0)
                    if is_sat2:
                        if clause_sat_count[ci2] == 1:
                            gain -= 1
                    else:
                        if clause_sat_count[ci2] == 0:
                            gain += 1
                flip_vars.append(v)
                if gain > best_gain:
                    best_gain = gain
                    best_var = v

            # Accept only if non-worsening, else random
            if best_gain >= 0:
                var = best_var
            else:
                var = flip_vars[rng.randrange(len(flip_vars))]

            assignment[var] = not assignment[var]
            for (ci2, s2) in var_to_clauses[var]:
                old_count = clause_sat_count[ci2]
                new_is_sat = (assignment[var] and s2 > 0) or (not assignment[var] and s2 < 0)
                old_is_sat = not new_is_sat
                if new_is_sat and not old_is_sat:
                    clause_sat_count[ci2] = old_count + 1
                    if old_count == 0:
                        unsat_count -= 1
                elif not new_is_sat and old_is_sat:
                    clause_sat_count[ci2] = old_count - 1
                    if old_count == 1:
                        unsat_count += 1

        return unsat_count == 0

    # Strategy: planted-aware init first, then random restarts with varied WalkSAT params
    base_steps = max(3000, n * 120)

    # First, try planted-aware initialization with a few WalkSAT runs
    planted_a = init_planted()
    if is_satisfied(planted_a):
        return planted_a

    rng0 = random.Random(12345)
    if walk_sat(planted_a[:], base_steps, rng0, 0.5, True, True):
        return planted_a

    # Try a few variations of planted init (flip some random vars)
    for trial in range(3):
        a = planted_a[:]
        r = random.Random(900 + trial)
        for _ in range(n // 5):
            a[r.randrange(n)] = not a[r.randrange(n)]
        if is_satisfied(a):
            return a
        rng_t = random.Random(90000 + trial)
        if walk_sat(a, base_steps, rng_t, 0.4, True, True):
            return a

    # Now random restarts with varied parameters
    restart_count = 0
    p_values = [0.2, 0.3, 0.5, 0.6, 0.8, 0.25, 0.45]
    weight_flags = [True, False, True, True, False, True, False]

    while time.time() < deadline - 5.0:
        restart_count += 1
        rng = random.Random(restart_count * 1000003 + 42)
        assignment = [rng.random() < 0.5 for _ in range(n)]

        if is_satisfied(assignment):
            return assignment

        p = p_values[restart_count % len(p_values)]
        use_weights = weight_flags[restart_count % len(weight_flags)]

        if walk_sat(assignment, base_steps, rng, p, use_weights, True):
            return assignment

        # If close, try a polish pass
        # (walk_sat already returns True if solved; if it returned False, we can
        #  still try polish on the resulting assignment)
        if time.time() < deadline - 5.0:
            if polish(assignment, base_steps // 2, rng):
                return assignment

    # Fallback: return planted-aware or random
    if is_satisfied(planted_a):
        return planted_a
    return [random.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
