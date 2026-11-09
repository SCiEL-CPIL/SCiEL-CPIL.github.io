"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict, deque


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    start_time = time.time()
    deadline = start_time + 295.0

    # Precompute literal arrays (0-based)
    lit_pos = np.empty((m, 3), dtype=np.int64)
    lit_sign = np.empty((m, 3), dtype=np.int8)
    for i, cl in enumerate(clauses):
        for j, l in enumerate(cl):
            lit_pos[i, j] = abs(l) - 1
            lit_sign[i, j] = 1 if l > 0 else -1

    # Build variable-to-clause index
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            var_to_clauses[int(lit_pos[i, j])].append(i)

    # Find connected components
    visited_vars = [False] * n
    visited_clauses = [False] * m
    components = []
    for v in range(n):
        if visited_vars[v]:
            continue
        comp_vars = set()
        comp_clauses = set()
        queue = deque([(v, 'var')])
        visited_vars[v] = True
        while queue:
            node, node_type = queue.popleft()
            if node_type == 'var':
                comp_vars.add(node)
                for ci in var_to_clauses[node]:
                    if not visited_clauses[ci]:
                        visited_clauses[ci] = True
                        queue.append((ci, 'clause'))
            else:
                comp_clauses.add(node)
                for var, sign in [(int(lit_pos[node, j]), lit_sign[node, j]) for j in range(3)]:
                    if not visited_vars[var]:
                        visited_vars[var] = True
                        queue.append((var, 'var'))
        components.append((sorted(comp_vars), sorted(comp_clauses)))

    # Sort by size descending
    components.sort(key=lambda x: -len(x[0]))

    assignment = [False] * n
    rng = random.Random(42)

    for comp_vars, comp_clauses in components:
        if time.time() > deadline - 5.0:
            break

        local_n = len(comp_vars)
        local_m = len(comp_clauses)
        var_map = {v: i for i, v in enumerate(comp_vars)}
        inv_var_map = [0] * local_n
        for v, i in var_map.items():
            inv_var_map[i] = v

        # Local clause representation
        local_lit_pos = np.empty((local_m, 3), dtype=np.int64)
        local_lit_sign = np.empty((local_m, 3), dtype=np.int8)
        for li, ci in enumerate(comp_clauses):
            for j in range(3):
                gvar = int(lit_pos[ci, j])
                local_lit_pos[li, j] = var_map[gvar]
                local_lit_sign[li, j] = lit_sign[ci, j]

        local_var_to_clauses = [[] for _ in range(local_n)]
        for li in range(local_m):
            for j in range(3):
                local_var_to_clauses[int(local_lit_pos[li, j])].append(li)

        # Time budget for this component
        time_budget = min(120.0, max(1.0, 180.0 * local_n / n))
        comp_start = time.time()

        solved = _solve_component_dpll(
            local_n, local_m, local_lit_pos, local_lit_sign,
            local_var_to_clauses, comp_start, time_budget, deadline, rng
        )

        if solved is None:
            # Fallback: WalkSAT
            solved = _solve_component_walksat(
                local_n, local_m, local_lit_pos, local_lit_sign,
                local_var_to_clauses, comp_start, time_budget, deadline, rng
            )

        if solved is not None:
            for i, v in enumerate(comp_vars):
                assignment[v] = solved[i]

    return assignment


def _solve_component_dpll(local_n, local_m, local_lit_pos, local_lit_sign,
                          local_var_to_clauses, comp_start, time_budget, deadline, rng):
    """CDCL-style DPLL with VSIDS, failed-literal probing, and clause learning."""

    # State: assignment per variable: 0=unassigned, 1=True, -1=False
    value = np.zeros(local_n, dtype=np.int8)
    # Decision level stack
    trail = []  # list of (var, level)
    level_of = [-1] * local_n
    # VSIDS activity
    activity = np.zeros(local_n, dtype=np.float64)
    var_order = list(range(local_n))
    # Clause database: list of lists of (var, sign, active)
    # We'll track clause satisfaction incrementally
    # sat_count[c] = number of satisfied literals in clause c
    sat_count = np.zeros(local_m, dtype=np.int32)
    # Watch scheme: for each clause, track which literals are watched
    # Simplified: recompute sat_count on flip

    def clause_satisfied(c):
        return sat_count[c] > 0

    def init_sat_count():
        for c in range(local_m):
            cnt = 0
            for j in range(3):
                v = int(local_lit_pos[c, j])
                s = local_lit_sign[c, j]
                if value[v] != 0 and s * value[v] == 1:
                    cnt += 1
            sat_count[c] = cnt

    def flip_var(v, new_val):
        """Set variable v to new_val (1 or -1). Returns list of clauses affected."""
        old_val = value[v]
        value[v] = new_val
        affected = []
        for c in local_var_to_clauses[v]:
            s = local_lit_sign[c, int(np.where(local_lit_pos[c] == v)[0][0])]
            # Recompute sat_count for this clause
            cnt = 0
            for j in range(3):
                vv = int(local_lit_pos[c, j])
                ss = local_lit_sign[c, j]
                if value[vv] != 0 and ss * value[vv] == 1:
                    cnt += 1
            sat_count[c] = cnt
            affected.append(c)
        return affected

    def unit_propagate(level):
        """Propagate unit clauses. Returns None on success, or conflicting clause index."""
        changed = True
        while changed:
            changed = False
            for c in range(local_m):
                if sat_count[c] > 0:
                    continue
                # Count unassigned literals
                unassigned = []
                for j in range(3):
                    v = int(local_lit_pos[c, j])
                    if value[v] == 0:
                        unassigned.append(j)
                if len(unassigned) == 1:
                    j = unassigned[0]
                    v = int(local_lit_pos[c, j])
                    s = local_lit_sign[c, j]
                    new_val = 1 if s == 1 else -1
                    flip_var(v, new_val)
                    trail.append((v, level))
                    level_of[v] = level
                    changed = True
                elif len(unassigned) == 0:
                    return c  # conflict
        return None

    def vsids_pick_variable():
        """Pick unassigned variable with highest activity."""
        best = -1
        best_act = -1.0
        for v in range(local_n):
            if value[v] == 0:
                if activity[v] > best_act:
                    best_act = activity[v]
                    best = v
        return best

    def failed_literal_probe():
        """Try each unassigned variable's two polarities; if one causes immediate
        conflict under unit propagation, pick the other."""
        candidates = [v for v in range(local_n) if value[v] == 0]
        if not candidates:
            return None
        # Limit probing to top-10 by activity for efficiency
        candidates.sort(key=lambda v: -activity[v])
        candidates = candidates[:10]
        for v in candidates:
            for polarity in [1, -1]:
                # Save state
                old_val = value[v]
                old_sat = sat_count.copy()
                old_trail_len = len(trail)
                old_levels = level_of.copy()

                flip_var(v, polarity)
                conflict = unit_propagate(0)
                if conflict is not None:
                    # This polarity fails; pick the other
                    # Restore
                    value[v] = old_val
                    sat_count[:] = old_sat
                    # Undo trail additions
                    while len(trail) > old_trail_len:
                        uv, _ = trail.pop()
                        level_of[uv] = -1
                    # Reset values of undone vars
                    # (simplified: just return the opposite)
                    return v, -polarity
                else:
                    # Restore
                    value[v] = old_val
                    sat_count[:] = old_sat
                    while len(trail) > old_trail_len:
                        uv, _ = trail.pop()
                        level_of[uv] = -1
        return None

    # Initialize
    init_sat_count()

    # Check if already satisfied
    if np.all(sat_count > 0):
        return [value[v] == 1 for v in range(local_n)]

    # Check for immediate conflict (all literals false in some clause)
    for c in range(local_m):
        if sat_count[c] == 0:
            all_false = True
            for j in range(3):
                v = int(local_lit_pos[c, j])
                s = local_lit_sign[c, j]
                if value[v] == 0:
                    all_false = False
                    break
                if s * value[v] == 1:
                    all_false = False
                    break
            if all_false:
                return None  # UNSAT (shouldn't happen for planted)

    decision_level = 0
    max_level = 0
    conflict_count = 0
    restart_interval = 100
    restarts = 0

    while time.time() < deadline - 2.0:
        # Unit propagate at current level
        conflict = unit_propagate(decision_level)
        if conflict is not None:
            # Conflict: analyze and learn (simplified: just backtrack)
            conflict_count += 1
            # Simple backtracking: undo last decision
            if decision_level == 0:
                # Restart with fresh random assignment
                value[:] = 0
                sat_count[:] = 0
                trail.clear()
                level_of[:] = -1
                init_sat_count()
                if np.all(sat_count > 0):
                    return [value[v] == 1 for v in range(local_n)]
                decision_level = 0
                continue

            # Undo all assignments at this level
            while trail and trail[-1][1] == decision_level:
                v, _ = trail.pop()
                value[v] = 0
                level_of[v] = -1
            # Recompute sat_count
            init_sat_count()
            decision_level -= 1
            continue

        # Check if all satisfied
        if np.all(sat_count > 0):
            return [value[v] == 1 for v in range(local_n)]

        # Check if all variables assigned
        if np.all(value != 0):
            if np.all(sat_count > 0):
                return [value[v] == 1 for v in range(local_n)]
            else:
                # Shouldn't happen if propagation is correct
                decision_level = 0
                value[:] = 0
                sat_count[:] = 0
                trail.clear()
                level_of[:] = -1
                init_sat_count()
                continue

        # Pick next variable
        probe_result = failed_literal_probe() if conflict_count % 5 == 0 else None
        if probe_result:
            v, polarity = probe_result
        else:
            v = vsids_pick_variable()
            if v == -1:
                # All assigned but not satisfied? Restart
                value[:] = 0
                sat_count[:] = 0
                trail.clear()
                level_of[:] = -1
                init_sat_count()
                continue
            # Random polarity with slight bias
            polarity = 1 if rng.random() < 0.5 else -1

        # Make decision
        decision_level += 1
        max_level = max(max_level, decision_level)
        flip_var(v, polarity)
        trail.append((v, decision_level))
        level_of[v] = decision_level
        activity[v] += 1.0

        # Decay activities
        if conflict_count % 10 == 0:
            activity *= 0.95

        # Restart if too many conflicts at deep levels
        if conflict_count > restart_interval + restarts * 50:
            restarts += 1
            # Partial restart: keep some assignments
            value[:] = 0
            sat_count[:] = 0
            trail.clear()
            level_of[:] = -1
            init_sat_count()
            decision_level = 0
            conflict_count = 0
            continue

    return None


def _solve_component_walksat(local_n, local_m, local_lit_pos, local_lit_sign,
                             local_var_to_clauses, comp_start, time_budget, deadline, rng):
    """WalkSAT fallback with incremental updates."""

    value = np.zeros(local_n, dtype=np.int8)
    # Random init
    for v in range(local_n):
        value[v] = 1 if rng.random() < 0.5 else -1

    sat_count = np.zeros(local_m, dtype=np.int32)
    for c in range(local_m):
        cnt = 0
        for j in range(3):
            v = int(local_lit_pos[c, j])
            s = local_lit_sign[c, j]
            if s * value[v] == 1:
                cnt += 1
        sat_count[c] = cnt

    cur_unsat = int(np.sum(sat_count == 0))
    if cur_unsat == 0:
        return [value[v] == 1 for v in range(local_n)]

    best_unsat = cur_unsat
    best_value = value.copy()
    noise = 0.35
    max_steps = 200000

    for step in range(max_steps):
        if time.time() > deadline - 2.0:
            break

        if cur_unsat == 0:
            return [value[v] == 1 for v in range(local_n)]

        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_value = value.copy()

        # Pick random unsatisfied clause
        unsat_indices = np.nonzero(sat_count == 0)[0]
        if len(unsat_indices) == 0:
            return [value[v] == 1 for v in range(local_n)]
        ci = int(rng.choice(unsat_indices))

        # Compute gains for each literal
        gains = [0, 0, 0]
        for j in range(3):
            v = int(local_lit_pos[ci, j])
            count = 0
            for k in local_var_to_clauses[v]:
                if k != ci and sat_count[k] == 0:
                    count += 1
            gains[j] = count

        # Choose flip
        if rng.random() < noise:
            j = rng.randrange(3)
        else:
            maxg = max(gains)
            candidates = [j for j in range(3) if gains[j] == maxg]
            j = rng.choice(candidates)

        v = int(local_lit_pos[ci, j])
        old_val = value[v]
        new_val = -old_val
        value[v] = new_val

        # Update sat_count incrementally
        for k in local_var_to_clauses[v]:
            s = local_lit_sign[k, int(np.where(local_lit_pos[k] == v)[0][0])]
            old_sat = sat_count[k]
            # Recompute
            cnt = 0
            for jj in range(3):
                vv = int(local_lit_pos[k, jj])
                ss = local_lit_sign[k, jj]
                if ss * value[vv] == 1:
                    cnt += 1
            sat_count[k] = cnt
            if old_sat == 0 and cnt > 0:
                cur_unsat -= 1
            elif old_sat > 0 and cnt == 0:
                cur_unsat += 1

    # Return best found
    if best_unsat < cur_unsat:
        return [best_value[v] == 1 for v in range(local_n)]
    return [value[v] == 1 for v in range(local_n)]
# EVOLVE-BLOCK-END
