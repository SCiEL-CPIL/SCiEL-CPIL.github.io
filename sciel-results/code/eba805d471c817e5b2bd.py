# EVOLVE-BLOCK-START
import random
import time

def solve(instance):
    start_time = time.time()
    time_limit = 295.0  # safety margin before 330s kill

    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)

    # ---------- Preprocessing: pure literal elimination + unit propagation ----------
    active_clauses = list(clauses)
    fixed = {}

    def preprocess(clause_list, fixed_assign):
        changed = True
        while changed:
            changed = False
            # pure literals
            var_polarity = {}
            for c in clause_list:
                for l in c:
                    v = abs(l) - 1
                    p = l > 0
                    if v not in fixed_assign:
                        var_polarity.setdefault(v, set()).add(p)
            for v, pols in var_polarity.items():
                if len(pols) == 1 and v not in fixed_assign:
                    fixed_assign[v] = pols.pop()
                    changed = True
            # unit propagation
            new_clauses = []
            for c in clause_list:
                remaining = []
                satisfied = False
                for l in c:
                    v = abs(l) - 1
                    p = l > 0
                    if v in fixed_assign:
                        if fixed_assign[v] == p:
                            satisfied = True
                            break
                        else:
                            continue
                    remaining.append(l)
                if satisfied:
                    continue
                new_clauses.append(remaining)
            for c in new_clauses:
                if len(c) == 0:
                    return None
                if len(c) == 1:
                    l = c[0]
                    v = abs(l) - 1
                    p = l > 0
                    if v not in fixed_assign:
                        fixed_assign[v] = p
                        changed = True
            clause_list = new_clauses
        return clause_list

    fixed_assign = {}
    remaining_clauses = preprocess(active_clauses, fixed_assign)

    if remaining_clauses is None:
        return [fixed_assign.get(i, random.random() < 0.5) for i in range(n)]

    # check all satisfied?
    all_sat = True
    for c in remaining_clauses:
        if len(c) == 0:
            all_sat = False
            break
        sat = any((fixed_assign.get(abs(l) - 1) == (l > 0)) for l in c)
        if not sat:
            all_sat = False
            break
    if all_sat:
        return [fixed_assign.get(i, random.random() < 0.5) for i in range(n)]

    # ---------- Build reduced problem over unfixed vars ----------
    unfixed_vars = [i for i in range(n) if i not in fixed_assign]
    var_map = {v: i for i, v in enumerate(unfixed_vars)}
    num_unfixed = len(unfixed_vars)

    clause_vars = []
    var_to_clauses = [[] for _ in range(num_unfixed)]
    for i, c in enumerate(remaining_clauses):
        cv = []
        for l in c:
            v = abs(l) - 1
            p = l > 0
            if v in var_map:
                li = var_map[v]
                cv.append((li, p))
                var_to_clauses[li].append(i)
        clause_vars.append(cv)
    m2 = len(clause_vars)

    if num_unfixed == 0:
        return [fixed_assign.get(i, random.random() < 0.5) for i in range(n)]

    # per-variable polarity in each clause (for fast damage), and occurrence count
    var_pol = [[None] * m2 for _ in range(num_unfixed)]
    for i, cv in enumerate(clause_vars):
        for li, p in cv:
            var_pol[li][i] = p

    def build_assignment():
        return [random.random() < 0.5 for _ in range(num_unfixed)]

    def full_assign(assignment):
        fa = [fixed_assign.get(i, False) for i in range(n)]
        for j, v in enumerate(unfixed_vars):
            fa[v] = assignment[j]
        return fa

    def init_sat_count(assignment):
        sat_count = [0] * m2
        for i, cv in enumerate(clause_vars):
            s = 0
            for li, p in cv:
                if assignment[li] == p:
                    s += 1
            sat_count[i] = s
        return sat_count

    def unsat_set_from(sat_count):
        return set(i for i in range(m2) if sat_count[i] == 0)

    def flip_var(assignment, var_idx, sat_count, unsat_set):
        assignment[var_idx] = not assignment[var_idx]
        for c_idx in var_to_clauses[var_idx]:
            p_c = var_pol[var_idx][c_idx]
            if p_c is None:
                continue
            was_unsat = (sat_count[c_idx] == 0)
            if assignment[var_idx] == p_c:
                sat_count[c_idx] += 1
                if was_unsat:
                    unsat_set.discard(c_idx)
            else:
                sat_count[c_idx] -= 1
                if sat_count[c_idx] == 0:
                    unsat_set.add(c_idx)

    # ---------- Clause-weighted WalkSAT ----------
    best_assignment = None
    best_unsat = float('inf')

    def weight_init(m2):
        return [1.0] * m2

    max_steps_per_restart = 2000
    max_restarts = 20000
    base_noise = 0.4
    stuck_threshold = 300

    while time.time() - start_time < time_limit and max_restarts > 0:
        max_restarts -= 1
        assignment = build_assignment()
        sat_count = init_sat_count(assignment)
        unsat_set = unsat_set_from(sat_count)
        if not unsat_set:
            return full_assign(assignment)

        weights = weight_init(m2)

        current_unsat = len(unsat_set)
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment[:]

        steps = 0
        no_improve = 0
        prev_unsat = current_unsat
        while steps < max_steps_per_restart and time.time() - start_time < time_limit:
            if not unsat_set:
                return full_assign(assignment)

            # weighted pick of an unsat clause
            total_w = 0.0
            wlist = []
            for i in unsat_set:
                w = weights[i]
                wlist.append(w)
                total_w += w
            r = random.random() * total_w
            acc = 0.0
            chosen = None
            for i, w in zip(unsat_set, wlist):
                acc += w
                if r <= acc:
                    chosen = i
                    break
            if chosen is None:
                chosen = next(iter(unsat_set))

            cv = clause_vars[chosen]
            # noise increases when stuck
            noise = min(base_noise + no_improve / 400.0, 0.9)

            if random.random() < noise:
                # random literal in the clause
                li, p = random.choice(cv)
            else:
                # minimum damage literal in the clause
                best_li = -1
                best_damage = float('inf')
                for li, p in cv:
                    damage = 0
                    for c_idx in var_to_clauses[li]:
                        if c_idx == chosen:
                            continue
                        pc = var_pol[li][c_idx]
                        if pc is None:
                            continue
                        if sat_count[c_idx] == 1 and assignment[li] == pc:
                            # this var is the only satisfier; flipping breaks it
                            damage += 1
                    if damage < best_damage:
                        best_damage = damage
                        best_li = li
                if best_li == -1:
                    best_li, _ = random.choice(cv)

            flip_var(assignment, best_li, sat_count, unsat_set)

            # update weights: raise unsat, decay sat
            for c_idx in var_to_clauses[best_li]:
                if sat_count[c_idx] == 0:
                    weights[c_idx] *= 1.1
                else:
                    weights[c_idx] *= 0.95

            steps += 1
            current_unsat = len(unsat_set)
            if current_unsat < prev_unsat:
                no_improve = 0
                prev_unsat = current_unsat
            else:
                no_improve += 1
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment[:]

            if no_improve > stuck_threshold:
                break

    # ---------- Final smooth polish on best_assignment ----------
    if best_assignment is not None and best_unsat > 0:
        remaining = time_limit - (time.time() - start_time)
        if remaining > 3.0:
            assignment = best_assignment[:]
            sat_count = init_sat_count(assignment)
            unsat_set = unsat_set_from(sat_count)
            if not unsat_set:
                return full_assign(assignment)
            # smooth objective: sum of soft-min per clause (lower is better)
            def smooth_obj(sat_count):
                s = 0.0
                for i in range(m2):
                    s += 1.0 / (1.0 + sat_count[i])
                return s
            base_obj = smooth_obj(sat_count)
            # greedy hill-climb over single flips minimizing smooth objective
            polish_steps = 0
            max_polish = 20000
            improve_streak = 0
            while polish_steps < max_polish and time.time() - start_time < time_limit - 2.0:
                # candidate: flip a variable from an unsat clause
                if not unsat_set:
                    return full_assign(assignment)
                cand_clause = random.choice(list(unsat_set))
                cand_var = random.choice(clause_vars[cand_clause])[0]

                old_obj = smooth_obj(sat_count)
                flip_var(assignment, cand_var, sat_count, unsat_set)
                new_obj = smooth_obj(sat_count)
                if new_obj < old_obj - 1e-12:
                    improve_streak += 1
                    if len(unsat_set) < best_unsat:
                        best_unsat = len(unsat_set)
                        best_assignment = assignment[:]
                else:
                    flip_var(assignment, cand_var, sat_count, unsat_set)  # undo
                    improve_streak = 0
                polish_steps += 1
                if improve_streak > 500:
                    break
    if best_assignment is not None:
        return full_assign(best_assignment)
    return [fixed_assign.get(i, random.random() < 0.5) for i in range(n)]

# EVOLVE-BLOCK-END
