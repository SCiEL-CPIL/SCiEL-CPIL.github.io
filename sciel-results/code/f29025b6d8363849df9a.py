"""Planted 3-SAT solver with preprocessing + clause-weighted WalkSAT."""
import random
import time
from collections import defaultdict

def solve(instance):
    start_time = time.time()
    deadline = 325.0  # Safety margin before 330s kill

    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [False] * n

    # --- Preprocessing: pure literal elimination + unit propagation ---
    # Work with a copy of clauses as lists of signed literals (1-based)
    work_clauses = [list(c) for c in clauses]
    assigned = {}  # var_idx (0-based) -> bool
    active = [True] * m  # which clauses are still active

    def propagate_pure_and_unit():
        """Iteratively apply pure literal elimination and unit propagation.
        Returns True if a contradiction is found (UNSAT), False otherwise."""
        changed = True
        while changed:
            changed = False
            # Remove clauses that are satisfied or have a pure literal
            # First, identify pure literals
            pos_vars = set()
            neg_vars = set()
            for ci in range(m):
                if not active[ci]:
                    continue
                for lit in work_clauses[ci]:
                    v = abs(lit) - 1
                    if v in assigned:
                        continue
                    if lit > 0:
                        pos_vars.add(v)
                    else:
                        neg_vars.add(v)

            # Pure literal elimination
            for v in list(pos_vars - neg_vars):
                assigned[v] = True
                changed = True
            for v in list(neg_vars - pos_vars):
                assigned[v] = False
                changed = True

            if changed:
                # Update active clauses
                for ci in range(m):
                    if not active[ci]:
                        continue
                    for lit in work_clauses[ci]:
                        v = abs(lit) - 1
                        if v in assigned:
                            if (lit > 0 and assigned[v]) or (lit < 0 and not assigned[v]):
                                active[ci] = False
                                break
                continue

            # Unit propagation
            unit_found = False
            for ci in range(m):
                if not active[ci]:
                    continue
                unassigned_lits = [lit for lit in work_clauses[ci]
                                   if (abs(lit) - 1) not in assigned]
                if len(unassigned_lits) == 0:
                    # All literals assigned but clause not satisfied -> contradiction
                    return True
                elif len(unassigned_lits) == 1:
                    lit = unassigned_lits[0]
                    v = abs(lit) - 1
                    assigned[v] = (lit > 0)
                    unit_found = True
                    changed = True

            if unit_found:
                # Update active clauses
                for ci in range(m):
                    if not active[ci]:
                        continue
                    for lit in work_clauses[ci]:
                        v = abs(lit) - 1
                        if v in assigned:
                            if (lit > 0 and assigned[v]) or (lit < 0 and not assigned[v]):
                                active[ci] = False
                                break

        return False

    contradiction = propagate_pure_and_unit()
    if contradiction:
        # Shouldn't happen for satisfiable instances, but handle gracefully
        pass

    # Build reduced problem: only unassigned variables and active clauses
    unassigned_vars = [v for v in range(n) if v not in assigned]
    var_map = {v: i for i, v in enumerate(unassigned_vars)}  # old_idx -> new_idx
    n_red = len(unassigned_vars)

    # Build reduced clauses
    red_clauses = []
    for ci in range(m):
        if not active[ci]:
            continue
        lits = []
        for lit in work_clauses[ci]:
            v = abs(lit) - 1
            if v not in assigned:
                lits.append(lit)
        if len(lits) <= 1:
            # Should have been handled by propagation, but skip if so
            if len(lits) == 1:
                lit = lits[0]
                v = abs(lit) - 1
                # This should be a unit clause; assign it
                assigned[v] = (lit > 0)
            continue
        if len(lits) == 2:
            # Convert to 3-lit by duplicating or just keep as 2-lit (WalkSAT handles any size)
            red_clauses.append(lits)
        else:
            red_clauses.append(lits)

    m_red = len(red_clauses)

    # If all variables are assigned, we're done
    if n_red == 0:
        return [assigned.get(v, False) for v in range(n)]

    # --- Clause-weighted WalkSAT on reduced problem ---
    # Parse reduced clauses
    red_clause_lits = []  # list of (new_var_idx, polarity 0/1)
    red_var_occ = [[] for _ in range(n_red)]  # (clause_idx, polarity)
    red_pos_count = [0] * n_red
    red_neg_count = [0] * n_red

    for ci, c in enumerate(red_clauses):
        lits = []
        for lit in c:
            v_old = abs(lit) - 1
            v_new = var_map[v_old]
            p = 1 if lit > 0 else 0
            lits.append((v_new, p))
            red_var_occ[v_new].append((ci, p))
            if p == 1:
                red_pos_count[v_new] += 1
            else:
                red_neg_count[v_new] += 1
        red_clause_lits.append(lits)

    # Try polarity-biased initial assignment on reduced problem
    def try_planted():
        if m_red == 0:
            return [False] * n_red
        assignment = []
        for v in range(n_red):
            assignment.append(red_pos_count[v] >= red_neg_count[v])
        for ci in range(m_red):
            sat = False
            for v, p in red_clause_lits[ci]:
                if assignment[v] == (p == 1):
                    sat = True
                    break
            if not sat:
                return None
        return assignment

    result = try_planted()
    if result is not None:
        # Map back
        full_assign = dict(assigned)
        for i, v in enumerate(unassigned_vars):
            full_assign[v] = result[i]
        return [full_assign.get(v, False) for v in range(n)]

    # WalkSAT with clause weights
    def walk_sat(max_restarts, base_steps, noise_init, noise_decay):
        best_unsat = float('inf')
        best_assign = None

        for restart in range(max_restarts):
            if time.time() - start_time > deadline:
                break

            # Initialize
            if restart < 2:
                assign = []
                for v in range(n_red):
                    if red_pos_count[v] > red_neg_count[v]:
                        assign.append(random.random() < 0.65)
                    elif red_neg_count[v] > red_pos_count[v]:
                        assign.append(random.random() < 0.35)
                    else:
                        assign.append(random.random() < 0.5)
            else:
                assign = [random.random() < 0.5 for _ in range(n_red)]

            # Count unsat
            unsat_count = 0
            clause_sat_count = [0] * m_red  # number of satisfied literals per clause
            for ci in range(m_red):
                cnt = 0
                for v, p in red_clause_lits[ci]:
                    if assign[v] == (p == 1):
                        cnt += 1
                clause_sat_count[ci] = cnt
                if cnt == 0:
                    unsat_count += 1

            if unsat_count == 0:
                return assign

            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assign[:]

            # Clause weights: weight = 1.0 / max(1, clause_sat_count) for unsat clauses
            # Actually, standard WalkSAT: pick random unsat clause, flip variable in it
            # with probability noise, else flip the one that minimizes damage

            steps = 0
            max_steps = base_steps + restart * 30
            noise = noise_init * (noise_decay ** restart)

            while steps < max_steps and time.time() - start_time < deadline:
                if unsat_count == 0:
                    return assign

                # Pick random unsat clause
                ci = random.randrange(m_red)
                while clause_sat_count[ci] != 0:
                    ci = random.randrange(m_red)

                clause = red_clause_lits[ci]

                # Choose variable to flip
                if random.random() < noise:
                    # Random flip
                    v, p = random.choice(clause)
                else:
                    # Greedy: minimize damage
                    best_v = -1
                    best_damage = float('inf')
                    for v, p in clause:
                        old_val = assign[v]
                        new_val = not old_val
                        damage = 0
                        for (cci, cp) in red_var_occ[v]:
                            if cci == ci:
                                continue
                            if clause_sat_count[cci] == 1:
                                # Only this literal satisfies it
                                if old_val == (cp == 1):
                                    damage += 1
                        if damage < best_damage:
                            best_damage = damage
                            best_v = v
                    v = best_v

                # Flip
                old_val = assign[v]
                assign[v] = not old_val
                new_val = assign[v]

                # Update clause_sat_count for all clauses containing v
                for (cci, cp) in red_var_occ[v]:
                    old_sat = old_val == (cp == 1)
                    new_sat = new_val == (cp == 1)
                    if old_sat and not new_sat:
                        clause_sat_count[cci] -= 1
                        if clause_sat_count[cci] == 0:
                            unsat_count += 1
                    elif not old_sat and new_sat:
                        clause_sat_count[cci] += 1
                        if clause_sat_count[cci] == 1:
                            unsat_count -= 1

                steps += 1

                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assign = assign[:]

            # Perturb if stuck
            if steps > 0 and time.time() - start_time < deadline:
                for _ in range(5):
                    if time.time() - start_time > deadline:
                        break
                    v = random.randrange(n_red)
                    old_val = assign[v]
                    assign[v] = not old_val
                    new_val = assign[v]
                    for (cci, cp) in red_var_occ[v]:
                        old_sat = old_val == (cp == 1)
                        new_sat = new_val == (cp == 1)
                        if old_sat and not new_sat:
                            clause_sat_count[cci] -= 1
                            if clause_sat_count[cci] == 0:
                                unsat_count += 1
                        elif not old_sat and new_sat:
                            clause_sat_count[cci] += 1
                            if clause_sat_count[cci] == 1:
                                unsat_count -= 1
                    if unsat_count < best_unsat:
                        best_unsat = unsat_count
                        best_assign = assign[:]
                    if unsat_count == 0:
                        return assign

        return best_assign

    # Run WalkSAT with parameters
    max_restarts = 300
    base_steps = 150
    noise_init = 0.3
    noise_decay = 0.995

    result = walk_sat(max_restarts, base_steps, noise_init, noise_decay)

    if result is not None:
        full_assign = dict(assigned)
        for i, v in enumerate(unassigned_vars):
            full_assign[v] = result[i]
        return [full_assign.get(v, False) for v in range(n)]

    # Fallback: polarity-based full assignment
    full_assign = dict(assigned)
    for v in range(n):
        if v not in full_assign:
            # Count polarities in original
            pos = 0
            neg = 0
            for c in clauses:
                for lit in c:
                    if abs(lit) - 1 == v:
                        if lit > 0:
                            pos += 1
                        else:
                            neg += 1
            full_assign[v] = pos >= neg
    return [full_assign.get(v, False) for v in range(n)]
