"""DPLL-based exact solver for planted satisfiable 3-CNF (50 vars)."""
# EVOLVE-BLOCK-START
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290.0  # safety margin before 300s kill

    # Preprocess: convert each clause to a set of literals (dedup, drop tautologies)
    clean_clauses = []
    for c in clauses:
        s = set(c)
        # drop tautologies (contains both x and -x)
        if any(l in s and -l in s for l in s):
            continue
        if s:
            clean_clauses.append(s)

    # Build initial literal occurrence counts for pure literal elimination
    lit_pos = {}
    lit_neg = {}
    for c in clean_clauses:
        for l in c:
            if l > 0:
                lit_pos[l] = lit_pos.get(l, 0) + 1
            else:
                lit_neg[-l] = lit_neg.get(-l, 0) + 1

    # Assignment stack: list of (var, value) for backtracking
    # We'll use arrays: assign[var] in {None, True, False}
    assign = [None] * (n + 1)  # 1-based
    trail = []  # list of (var, value) in order

    # We'll represent clauses as lists of ints (literals), mutating in place.
    # For propagation we need to know, per variable, which clauses contain it (pos/neg).
    # Build occurrence lists for efficient unit propagation.
    # occ[var] = list of (clause_index, sign) where sign=+1 means clause has +var, -1 means -var
    occ = [[] for _ in range(n + 1)]
    for ci, c in enumerate(clean_clauses):
        for l in c:
            var = abs(l)
            sign = 1 if l > 0 else -1
            occ[var].append((ci, sign))

    # Active clause mask
    active = [True] * len(clean_clauses)

    # Counts of unassigned literals per clause (for unit detection)
    # We'll recompute as needed via a helper, but keep a counter for speed.
    # Let's maintain clause_unsat_count: number of literals not yet satisfied/false.
    # Actually simpler: maintain for each clause a count of literals that are "undecided"
    # and whether any is true. We recompute on each change.

    # To keep it robust and simple, we maintain:
    #   clause_sat[ci] = number of literals in clause ci currently TRUE
    #   clause_undecided[ci] = number of literals not yet assigned
    clause_sat = [0] * len(clean_clauses)
    clause_undecided = [0] * len(clean_clauses)
    for ci, c in enumerate(clean_clauses):
        clause_undecided[ci] = len(c)

    # We'll implement propagation with a queue of newly-assigned variables.

    def is_sat(ci):
        return clause_sat[ci] > 0

    def conflict():
        return any(clause_sat[ci] == 0 and clause_undecided[ci] == 0 for ci in range(len(clean_clauses)) if active[ci])

    # Pure literal elimination: a literal l is pure if no clause contains the opposite.
    def pure_literal_eliminate():
        # Returns list of (var, value) to assign, or None if no pure literals
        # We check current lit_pos/lit_neg counts considering only active clauses and unassigned vars.
        # To be safe and simple, recompute pure-ness from scratch each call.
        pos = {}
        neg = {}
        for ci, c in enumerate(clean_clauses):
            if not active[ci]:
                continue
            for l in c:
                var = abs(l)
                if assign[var] is not None:
                    continue
                if l > 0:
                    pos[var] = pos.get(var, 0) + 1
                else:
                    neg[var] = neg.get(var, 0) + 1
        assignments = []
        for var in range(1, n + 1):
            if assign[var] is not None:
                continue
            has_pos = pos.get(var, 0) > 0
            has_neg = neg.get(var, 0) > 0
            if has_pos and not has_neg:
                assignments.append((var, True))
            elif has_neg and not has_pos:
                assignments.append((var, False))
        return assignments

    def assign_var(var, value):
        # Assign var=value, update clause counts, return False on conflict
        assign[var] = value
        trail.append((var, value))
        for ci, sign in occ[var]:
            if not active[ci]:
                continue
            clause_undecided[ci] -= 1
            if sign == 1:
                # clause contains +var
                if value:
                    clause_sat[ci] += 1
            else:
                # clause contains -var
                if not value:
                    clause_sat[ci] += 1
        return True

    def unassign_var(var):
        # Undo last assignment of var (must be top of trail)
        value = assign[var]
        assign[var] = None
        trail.pop()
        for ci, sign in occ[var]:
            if not active[ci]:
                continue
            clause_undecided[ci] += 1
            if sign == 1:
                if value:
                    clause_sat[ci] -= 1
            else:
                if not value:
                    clause_sat[ci] -= 1

    # Unit propagation: repeatedly assign forced variables.
    # Returns False if conflict, True otherwise.
    def propagate():
        # Use a worklist of variables that may have created unit clauses.
        # Simple approach: loop until no new assignments.
        while True:
            # Check for conflict
            for ci in range(len(clean_clauses)):
                if active[ci] and clause_sat[ci] == 0 and clause_undecided[ci] == 0:
                    return False
            # Find a unit clause
            unit_var = None
            unit_val = None
            for ci in range(len(clean_clauses)):
                if not active[ci]:
                    continue
                if clause_sat[ci] > 0:
                    continue
                if clause_undecided[ci] == 1:
                    # find the unassigned literal
                    for l in clean_clauses[ci]:
                        var = abs(l)
                        if assign[var] is None:
                            unit_var = var
                            unit_val = (l > 0)
                            break
                    break
            if unit_var is None:
                return True
            if not assign_var(unit_var, unit_val):
                return False
        # unreachable

    # Variable ordering: pick unassigned variable appearing in most unit-ish/active clauses
    def choose_var():
        # Score each unassigned var by number of active clauses containing it (undecided)
        best_var = None
        best_score = -1
        for var in range(1, n + 1):
            if assign[var] is not None:
                continue
            score = 0
            for ci, sign in occ[var]:
                if active[ci] and clause_sat[ci] == 0:
                    score += 1
            if score > best_score:
                best_score = score
                best_var = var
        return best_var

    # Main DPLL search with recursion via explicit stack to control backtracking.
    # We'll do a simple recursive function with a step counter.

    import sys
    sys.setrecursionlimit(100000)

    steps = [0]
    MAX_STEPS = 20000000  # generous; deadline guard is primary

    def dpll():
        # Pure literal elimination
        ples = pure_literal_eliminate()
        if ples:
            for var, val in ples:
                if not assign_var(var, val):
                    return None
        if not propagate():
            return None
        # Check all satisfied?
        for ci in range(len(clean_clauses)):
            if active[ci] and clause_sat[ci] == 0 and clause_undecided[ci] == 0:
                return None
        # All variables assigned?
        all_assigned = all(assign[v] is not None for v in range(1, n + 1))
        if all_assigned:
            return True
        var = choose_var()
        if var is None:
            return True
        steps[0] += 1
        if steps[0] > MAX_STEPS or time.time() > deadline:
            return None
        # Try True first, then False
        for val in (True, False):
            if not assign_var(var, val):
                continue
            if not propagate():
                unassign_var(var)
                continue
            res = dpll()
            if res is True:
                return True
            unassign_var(var)
            if time.time() > deadline:
                return None
        return None

    result = dpll()

    # Build final assignment list (1-based -> 0-based booleans)
    if result is True:
        # All vars should be assigned
        out = [bool(assign[v]) for v in range(1, n + 1)]
        return out
    else:
        # Fallback: return current partial assignment padded
        out = []
        for v in range(1, n + 1):
            if assign[v] is None:
                out.append(False)
            else:
                out.append(bool(assign[v]))
        return out
# EVOLVE-BLOCK-END
