"""Planted 3-SAT solver (50 vars): fast DPLL + optimized WalkSAT polish."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import deque


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    # Safety margin: evaluator kills at 330s; use 320s internal deadline.
    t0 = time.time()
    deadline = t0 + 320.0

    # Convert to 0-based signed literals: pos => var+1, neg => -(var+1)
    # Keep original for verification; build (var, pol) for solver.
    clauses0 = []
    for c in clauses:
        cc = []
        for l in c:
            if l > 0:
                cc.append(l - 1)  # var index
            else:
                cc.append(-(l + 1))
        clauses0.append(cc)
    m = len(clauses0)

    # For each variable, list of (clause_idx, polarity)
    var_to_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clauses0):
        for lit in c:
            v = lit if lit > 0 else -lit
            pol = 1 if lit > 0 else -1
            var_to_clauses[v - 1].append((ci, pol))

    def satisfies(assignment, c0):
        for lit in c0:
            v = lit if lit > 0 else -lit
            val = assignment[v - 1]
            if (lit > 0 and val) or (lit < 0 and not val):
                return True
        return False

    def verify(assignment):
        for c0 in clauses0:
            ok = False
            for lit in c0:
                v = lit if lit > 0 else -lit
                val = assignment[v - 1]
                if (lit > 0 and val) or (lit < 0 and not val):
                    ok = True
                    break
            if not ok:
                return False
        return True

    # ---------------- DPLL (watched literals) ----------------
    # assignment: list of True/False/None
    # watches[ci] = [i, j] indices of literals in clause0[ci] that are watched
    # We'll implement a non-recursive DPLL with restarts and clause learning.

    def dpll_phase():
        if time.time() > deadline - 3.0:
            return None

        # Build watched literals for each clause
        # clause0[ci] is list of signed 0-based literals
        # We'll store watch positions as indices into the clause list.
        watches = []
        for c in clauses0:
            # Ensure at least 2 literals; if 1, handle unit directly later.
            watches.append([0, 1] if len(c) >= 2 else [0, 0])

        # Assignment arrays
        assignment = [None] * n
        level = [0] * n  # decision level when assigned
        trail = []       # list of vars assigned in order
        trail_lim = []   # decision level boundaries
        decisions = 0

        # Conflict analysis state
        learnt_clauses = []
        learnt_watches = []
        reason = [None] * n  # clause index that forced var (for learned clauses)

        # Heuristic: VSIDS-like activity
        activity = [1.0] * n
        inc = 1.0 / n

        # Unit queue (var, value)
        unitq = deque()

        def set_var(v, val, lvl):
            assignment[v] = val
            level[v] = lvl
            trail.append(v)
            # Update watches for clauses containing v
            for ci, pol in var_to_clauses[v]:
                c = clauses0[ci]
                w0, w1 = watches[ci]
                # Find positions of v in clause (there may be duplicates but 3-CNF unique)
                pos = None
                for idx, lit in enumerate(c):
                    vv = lit if lit > 0 else -lit
                    if vv == v + 1:
                        pos = idx
                        break
                if pos is None:
                    continue
                if pos == w0:
                    w0 = w1
                elif pos == w1:
                    w1 = w0
                # If the watched literal just became false, we need to find a new watch
                lit_at_pos = c[pos]
                vv2 = lit_at_pos if lit_at_pos > 0 else -lit_at_pos
                val2 = assignment[vv2 - 1]
                is_false_now = (lit_at_pos > 0 and val2 is False) or (lit_at_pos < 0 and val2 is True)
                if is_false_now:
                    # Search for a new watch literal that is not false
                    new_pos = None
                    for idx, lit in enumerate(c):
                        if idx == pos:
                            continue
                        vv3 = lit if lit > 0 else -lit
                        val3 = assignment[vv3 - 1]
                        is_false = (lit > 0 and val3 is False) or (lit < 0 and val3 is True)
                        if not is_false:
                            new_pos = idx
                            break
                    if new_pos is None:
                        # Clause is now unit or conflict
                        # Determine the single unassigned literal (if any)
                        for idx, lit in enumerate(c):
                            vv4 = lit if lit > 0 else -lit
                            val4 = assignment[vv4 - 1]
                            if val4 is None:
                                unitq.append((vv4 - 1, lit > 0))
                                return False  # signal unit propagation needed
                        return True  # conflict
                    # Swap
                    if new_pos == w0:
                        w0, w1 = pos, w1
                    else:
                        w0, w1 = w1, pos
                watches[ci] = [w0, w1]
            return False

        def propagate():
            nonlocal decisions, inc
            while unitq:
                v, val = unitq.popleft()
                if assignment[v] is not None:
                    if assignment[v] != val:
                        return False  # conflict
                    continue
                if not set_var(v, val, level[trail[-1]] if trail else 0):
                    return False
            return True

        def pick_var():
            # Choose variable with highest activity among unassigned
            best_v = -1
            best_a = -1.0
            for v in range(n):
                if assignment[v] is None:
                    if activity[v] > best_a:
                        best_a = activity[v]
                        best_v = v
            return best_v

        def bump_activity(v):
            nonlocal inc
            activity[v] += inc
            if activity[v] > 1e100:
                for i in range(n):
                    activity[i] /= 1e100
                inc /= 1e100

        # Initial unit propagation
        if not propagate():
            return None
        if time.time() > deadline - 3.0:
            return None

        max_conflicts = 200000
        conflicts = 0
        restarts = 0
        max_restarts = 50
        decisions_this_restart = 0

        # Main loop
        while True:
            if time.time() > deadline - 3.0:
                break
            if conflicts > max_conflicts and restarts >= max_restarts:
                break

            # Check if all clauses satisfied
            all_sat = True
            for ci in range(m):
                c = clauses0[ci]
                sat = False
                for lit in c:
                    v = lit if lit > 0 else -lit
                    val = assignment[v - 1]
                    if (lit > 0 and val is True) or (lit < 0 and val is False):
                        sat = True
                        break
                if not sat:
                    all_sat = False
                    break
            if all_sat:
                # Fill unassigned
                for i in range(n):
                    if assignment[i] is None:
                        assignment[i] = False
                return assignment[:]

            # Pick variable
            v = pick_var()
            if v == -1:
                break

            # Decision
            decisions_this_restart += 1
            # Branch on True first (heuristic)
            # We'll try a fixed order; if conflict, we'll backtrack
            # For simplicity, we do a simple DPLL without full CDCL backtracking:
            # We'll use a recursive-style with trail backtracking.

            # Backtracking DPLL (simpler and robust for 50 vars)
            def dpll_rec(lvl):
                nonlocal conflicts, inc
                if time.time() > deadline - 3.0:
                    return None
                # Unit propagation
                if not propagate():
                    return None
                # Check all satisfied
                all_sat = True
                for ci in range(m):
                    c = clauses0[ci]
                    sat = False
                    for lit in c:
                        vv = lit if lit > 0 else -lit
                        val = assignment[vv - 1]
                        if (lit > 0 and val is True) or (lit < 0 and val is False):
                            sat = True
                            break
                    if not sat:
                        all_sat = False
                        break
                if all_sat:
                    for i in range(n):
                        if assignment[i] is None:
                            assignment[i] = False
                    return assignment[:]
                # Pick var
                best_v = -1
                best_a = -1.0
                for i in range(n):
                    if assignment[i] is None:
                        if activity[i] > best_a:
                            best_a = activity[i]
                            best_v = i
                if best_v == -1:
                    return None
                # Bump activity
                bump_activity(best_v)
                # Try True
                old_trail_len = len(trail)
                assignment[best_v] = True
                level[best_v] = lvl
                trail.append(best_v)
                # Propagate
                ok = True
                for ci, pol in var_to_clauses[best_v]:
                    c = clauses0[ci]
                    # Check if clause satisfied
                    sat = False
                    for lit in c:
                        vv = lit if lit > 0 else -lit
                        val = assignment[vv - 1]
                        if (lit > 0 and val is True) or (lit < 0 and val is False):
                            sat = True
                            break
                    if sat:
                        continue
                    # Check if clause falsified
                    falsified = True
                    for lit in c:
                        vv = lit if lit > 0 else -lit
                        val = assignment[vv - 1]
                        if val is None:
                            falsified = False
                            break
                        if (lit > 0 and val is True) or (lit < 0 and val is False):
                            falsified = False
                            break
                    if falsified:
                        ok = False
                        break
                    # If unit, queue it
                    unassigned = []
                    for lit in c:
                        vv = lit if lit > 0 else -vv if False else (lit if lit > 0 else -lit)
                        val = assignment[vv - 1]
                        if val is None:
                            unassigned.append((vv - 1, lit > 0))
                    if len(unassigned) == 1:
                        unitq.append(unassigned[0])
                if ok and not propagate():
                    ok = False
                if ok:
                    res = dpll_rec(lvl + 1)
                    if res is not None:
                        return res
                # Backtrack
                del trail[old_trail_len:]
                assignment[best_v] = None
                level[best_v] = 0

                # Try False
                old_trail_len = len(trail)
                assignment[best_v] = False
                level[best_v] = lvl
                trail.append(best_v)
                ok = True
                for ci, pol in var_to_clauses[best_v]:
                    c = clauses0[ci]
                    sat = False
                    for lit in c:
                        vv = lit if lit > 0 else -lit
                        val = assignment[vv - 1]
                        if (lit > 0 and val is True) or (lit < 0 and val is False):
                            sat = True
                            break
                    if sat:
                        continue
                    falsified = True
                    for lit in c:
                        vv = lit if lit > 0 else -lit
                        val = assignment[vv - 1]
                        if val is None:
                            falsified = False
                            break
                        if (lit > 0 and val is True) or (lit < 0 and val is False):
                            falsified = False
                            break
                    if falsified:
                        ok = False
                        break
                    unassigned = []
                    for lit in c:
                        vv = lit if lit > 0 else -lit
                        val = assignment[vv - 1]
                        if val is None:
                            unassigned.append((vv - 1, lit > 0))
                    if len(unassigned) == 1:
                        unitq.append(unassigned[0])
                if ok and not propagate():
                    ok = False
                if ok:
                    res = dpll_rec(lvl + 1)
                    if res is not None:
                        return res
                # Backtrack
                del trail[old_trail_len:]
                assignment[best_v] = None
                level[best_v] = 0
                return None

            res = dpll_rec(0)
            if res is not None:
                return res
            # Restart
            restarts += 1
            # Reset assignment but keep some learned info (activity)
            assignment = [None] * n
            level = [0] * n
            trail = []
            unitq = deque()
            # Reinitialize watches
            for ci in range(m):
                watches[ci] = [0, 1] if len(clauses0[ci]) >= 2 else [0, 0]
            decisions_this_restart = 0
            if restarts >= max_restarts:
                break

        return None

    dpll_result = dpll_phase()
    if dpll_result is not None and verify(dpll_result):
        return dpll_result

    # ---------------- WalkSAT phase ----------------
    rng = random.Random(12345)

    # Precompute clause literal var indices and polarities for speed
    clause_vars = []
    for c in clauses0:
        vars_pol = []
        for lit in c:
            v = (lit if lit > 0 else -lit) - 1
            pol = 1 if lit > 0 else -1
            vars_pol.append((v, pol))
        clause_vars.append(vars_pol)

    def count_unsat(assignment):
        cnt = 0
        for cv in clause_vars:
            sat = False
            for v, pol in cv:
                if assignment[v] == (pol == 1):
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt

    best_assign = None
    best_unsat = float('inf')

    # Multiple restarts with increasing step budgets
    max_restarts = 800
    base_steps = 20000
    max_steps = 200000

    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break

        # Random initialization
        assignment = [rng.random() < 0.5 for _ in range(n)]
        unsat = count_unsat(assignment)
        if unsat < best_unsat:
            best_unsat = unsat
            best_assign = assignment[:]
            if unsat == 0:
                return assignment

        steps = min(base_steps + restart * 500, max_steps)

        # Track unsatisfied clause indices
        unsat_set = set()
        for ci, cv in enumerate(clause_vars):
            sat = False
            for v, pol in cv:
                if assignment[v] == (pol == 1):
                    sat = True
                    break
            if not sat:
                unsat_set.add(ci)

        # WalkSAT loop
        for step in range(steps):
            if time.time() > deadline - 5.0:
                break
            if not unsat_set:
                return assignment

            cur_unsat = len(unsat_set)
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assign = assignment[:]

            # Pick random unsatisfied clause
            ci = rng.choice(list(unsat_set))
            clause = clause_vars[ci]

            # Score each literal: net change in satisfied clauses
            best_var = -1
            best_score = -1e9
            best_flip = False

            for v, pol in clause:
                new_val = not assignment[v]
                score = 0
                for cj, cpol in var_to_clauses[v]:
                    c_j = clause_vars[cj]
                    # Current satisfaction
                    cur_sat = False
                    for vv, pp in c_j:
                        if assignment[vv] == (pp == 1):
                            cur_sat = True
                            break
                    # New satisfaction (flip v)
                    new_sat = False
                    for vv, pp in c_j:
                        if vv == v:
                            if new_val == (pp == 1):
                                new_sat = True
                                break
                        else:
                            if assignment[vv] == (pp == 1):
                                new_sat = True
                                break
                    if cur_sat and not new_sat:
                        score -= 1
                    elif not cur_sat and new_sat:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_var = v
                    best_flip = new_val

            # Exploration: with probability 0.3, pick random literal in clause
            if rng.random() < 0.3:
                v, pol = rng.choice(clause)
                new_val = not assignment[v]
            else:
                v = best_var
                new_val = best_flip

            if v < 0:
                continue

            old_val = assignment[v]
            assignment[v] = new_val

            # Update unsat_set incrementally
            for cj, cpol in var_to_clauses[v]:
                c_j = clause_vars[cj]
                was_sat = False
                for vv, pp in c_j:
                    if vv == v:
                        if old_val == (pp == 1):
                            was_sat = True
                            break
                    else:
                        if assignment[vv] == (pp == 1):
                            was_sat = True
                            break
                is_sat_now = False
                for vv, pp in c_j:
                    if vv == v:
                        if new_val == (pp == 1):
                            is_sat_now = True
                            break
                    else:
                        if assignment[vv] == (pp == 1):
                            is_sat_now = True
                            break
                if was_sat and not is_sat_now:
                    unsat_set.add(cj)
                elif not was_sat and is_sat_now:
                    unsat_set.discard(cj)

        if not unsat_set:
            return assignment

    if best_assign is not None and verify(best_assign):
        return best_assign
    if best_assign is not None:
        return best_assign
    return [False] * n
# EVOLVE-BLOCK-END
