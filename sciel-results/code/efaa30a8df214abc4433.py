"""Improved 3-SAT solver: DPLL with VSIDS, failed-literal probing, and unit propagation."""
import sys
import random

sys.setrecursionlimit(100000)


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]

    def lit_to_var(lit):
        return abs(lit) - 1

    def lit_to_sign(lit):
        return lit > 0

    def is_satisfied(lit, assignment):
        v = lit_to_var(lit)
        s = lit_to_sign(lit)
        if assignment[v] is None:
            return False
        return assignment[v] == s

    def is_false(lit, assignment):
        v = lit_to_var(lit)
        s = lit_to_sign(lit)
        if assignment[v] is None:
            return False
        return assignment[v] != s

    def unit_propagate(all_clauses, assignment, trail):
        changed = True
        while changed:
            changed = False
            for clause in all_clauses:
                sat = False
                unit_lit = None
                unassigned_count = 0
                for lit in clause:
                    if is_satisfied(lit, assignment):
                        sat = True
                        break
                    elif is_false(lit, assignment):
                        continue
                    else:
                        unassigned_count += 1
                        unit_lit = lit
                if sat:
                    continue
                if unassigned_count == 0:
                    return False
                if unassigned_count == 1:
                    v = lit_to_var(unit_lit)
                    s = lit_to_sign(unit_lit)
                    if assignment[v] is not None:
                        if assignment[v] != s:
                            return False
                        else:
                            continue
                    assignment[v] = s
                    trail.append(v)
                    changed = True
        return True

    def pick_variable(var_activities, assignment):
        best_var = -1
        best_act = -1.0
        for v in range(len(assignment)):
            if assignment[v] is None:
                if var_activities[v] > best_act:
                    best_act = var_activities[v]
                    best_var = v
        return best_var if best_var >= 0 else None

    def failed_literal_probe(all_clauses, assignment, trail, var_activities):
        for v in range(len(assignment)):
            if assignment[v] is not None:
                continue
            # Try v = True
            assignment[v] = True
            trail.append(v)
            trail_len_before = len(trail)
            if not unit_propagate(all_clauses, assignment, trail):
                # Conflict, so v must be False
                while len(trail) > trail_len_before:
                    uv = trail.pop()
                    assignment[uv] = None
                assignment[v] = False
                trail.append(v)
                var_activities[v] *= 2.0
                return assignment, trail
            else:
                while len(trail) > trail_len_before:
                    uv = trail.pop()
                    assignment[uv] = None
                assignment[v] = True
                trail.append(v)
                var_activities[v] *= 2.0
                return assignment, trail
        return assignment, trail

    def dpll_search(all_clauses, assignment, trail, var_activities, depth=0):
        if not unit_propagate(all_clauses, assignment, trail):
            return False

        if all(a is not None for a in assignment):
            return True

        v = pick_variable(var_activities, assignment)
        if v is None:
            return True

        # Try True
        assignment[v] = True
        trail.append(v)
        if dpll_search(all_clauses, assignment, trail, var_activities, depth + 1):
            return True

        # Undo to before this decision
        while len(trail) > 0:
            uv = trail.pop()
            assignment[uv] = None

        # Try False
        assignment[v] = False
        trail.append(v)
        if dpll_search(all_clauses, assignment, trail, var_activities, depth + 1):
            return True

        # Undo
        while len(trail) > 0:
            uv = trail.pop()
            assignment[uv] = None

        return False

    # Initialize
    assignment = [None] * n
    trail = []
    var_activities = [1.0] * n

    # Initial unit propagation
    if not unit_propagate(clauses, assignment, trail):
        return [a if a is not None else False for a in assignment]

    if all(a is not None for a in assignment):
        return assignment

    # Failed literal probe at root
    assignment, trail = failed_literal_probe(clauses, assignment, trail, var_activities)

    # Main DPLL search
    dpll_search(clauses, assignment, trail, var_activities)

    for v in range(n):
        if assignment[v] is None:
            assignment[v] = False

    return assignment
