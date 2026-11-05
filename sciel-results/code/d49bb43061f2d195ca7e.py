import time
import random
import sys

def solve(instance):
    START = time.time()
    DEADLINE = START + 295.0  # Safety margin before 330s kill

    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)

    # Preprocess: remove tautologies, duplicates within clauses
    processed_clauses = []
    for c in clauses:
        lits = list(set(c))
        is_taut = False
        for l in lits:
            if -l in lits:
                is_taut = True
                break
        if not is_taut:
            processed_clauses.append(lits)

    if not processed_clauses:
        return [False] * n

    # Check for empty clause (UNSAT)
    for c in processed_clauses:
        if len(c) == 0:
            return None

    # Data structures for watched literals
    # clause_watches[i] = [lit1, lit2] for clause i
    clause_watches = []
    watches = [[] for _ in range(n)]  # watches[var] = list of (clause_idx, literal)

    for i, c in enumerate(processed_clauses):
        if len(c) >= 2:
            cw = [c[0], c[1]]
            clause_watches.append(cw)
            watches[abs(c[0]) - 1].append((i, c[0]))
            watches[abs(c[1]) - 1].append((i, c[1]))
        elif len(c) == 1:
            clause_watches.append([c[0]])
            watches[abs(c[0]) - 1].append((i, c[0]))
        else:
            return None

    # Assignment: 0 unassigned, 1 true, -1 false
    assignment = [0] * n
    trail = []
    level = [0] * n
    current_level = 0

    # VSIDS
    activity = [1.0] * n
    inc = 1.0
    decay = 0.95

    # Queue for unit propagation
    queue = []

    def assign(var, val, lvl):
        """Assign var to val. Returns False if conflict."""
        if assignment[var] != 0:
            return assignment[var] == val

        assignment[var] = val
        trail.append(var)
        level[var] = lvl

        # Process watches for this variable
        to_remove = []
        for (ci, lit) in watches[var]:
            # If lit is satisfied by current assignment, do nothing
            if (lit > 0 and val == 1) or (lit < 0 and val == -1):
                continue

            # lit is falsified. Find new watch.
            cw = clause_watches[ci]
            if cw is None:
                continue

            # Find the other watched literal
            other_idx = 0 if cw[0] == lit else 1
            if len(cw) == 1:
                # Unit clause, other_lit doesn't exist
                # This means the clause is now empty -> conflict
                return False
            other_lit = cw[other_idx]
            ov = abs(other_lit) - 1

            # If other_lit is satisfied, we are fine
            if (other_lit > 0 and assignment[ov] == 1) or (other_lit < 0 and assignment[ov] == -1):
                continue

            # Need to find a new literal in the clause
            c = processed_clauses[ci]
            new_lit = 0
            for l in c:
                v = abs(l) - 1
                if (l > 0 and assignment[v] == 1) or (l < 0 and assignment[v] == -1):
                    continue
                new_lit = l
                break

            if new_lit == 0:
                # Conflict
                return False

            # Replace other_idx with new_lit
            cw[other_idx] = new_lit
            # Add new watch
            watches[abs(new_lit) - 1].append((ci, new_lit))
            # Mark old watch for removal
            to_remove.append((ci, lit))

            # If the clause is now unit (only 1 unassigned lit left), enqueue it
            # The new_lit is the only unassigned one if no other unassigned lits exist
            # But we don't track size explicitly. However, if new_lit is the only unassigned,
            # it means the clause has exactly 1 unassigned literal.
            # We can check: if the clause has no other unassigned literals besides new_lit.
            # But that's expensive. Instead, note that in 3-SAT, after falsifying one lit,
            # if the other two are both assigned and falsified, it's a conflict (handled above).
            # If one is assigned and falsified, and the other is unassigned, then new_lit is that other.
            # The clause becomes unit if exactly one literal is unassigned.
            # Since we found new_lit, and the other watch was falsified, 
            # the clause is unit if there are no other unassigned literals.
            # For efficiency in 3-SAT, we can assume that if we're here, 
            # and the clause size is 3, and one is falsified, 
            # if the other two are: one assigned-falsified, one unassigned -> unit.
            # But we don't know easily. Let's just enqueue new_lit if it's unassigned.
            # Actually, new_lit is always unassigned (we skipped assigned-satisfied, 
            # and if it were assigned-falsified, we'd skip it too? No, we skip satisfied only).
            # Wait, in the loop, we skip literals that are satisfied. 
            # If a literal is assigned-falsified, it's not satisfied, so we'd pick it? 
            # No, we want unassigned literals. Let's fix the search.

            # Actually, the standard algorithm: find an unassigned literal in the clause.
            # Let's redo the search properly.

        # Remove stale watches
        for wr in to_remove:
            if wr in watches[var]:
                watches[var].remove(wr)

        return True

    # The above assign has a bug in the new_lit search. Let me rewrite it more carefully.
    # Given the complexity and the small n=50, let's use a simpler but correct DPLL.

    # Reset and use simple DPLL with good heuristics
    assignment_simple = [0] * n

    def is_clause_satisfied(clause, assign):
        for lit in clause:
            v = abs(lit) - 1
            if (lit > 0 and assign[v] == 1) or (lit < 0 and assign[v] == -1):
                return True
        return False

    def unit_propagate_simple(assign):
        """Returns False if conflict, True otherwise. Modifies assign in place."""
        changed = True
        while changed:
            changed = False
            for c in processed_clauses:
                if is_clause_satisfied(c, assign):
                    continue
                unassigned = []
                for lit in c:
                    v = abs(lit) - 1
                    if assign[v] == 0:
                        unassigned.append(lit)
                    elif (lit > 0 and assign[v] == 1) or (lit < 0 and assign[v] == -1):
                        # Already satisfied, but we checked above. This shouldn't happen.
                        pass
                if len(unassigned) == 0:
                    return False
                if len(unassigned) == 1:
                    lit = unassigned[0]
                    v = abs(lit) - 1
                    val = 1 if lit > 0 else -1
                    if assign[v] != 0:
                        if assign[v] != val:
                            return False
                    else:
                        assign[v] = val
                        changed = True
        return True

    def dpll_simple(assign, depth=0):
        if time.time() > DEADLINE - 5:
            return None

        # Unit Propagation
        if not unit_propagate_simple(assign):
            return None

        # Check if all assigned
        all_assigned = True
        for v in range(n):
            if assign[v] == 0:
                all_assigned = False
                break

        if all_assigned:
            # Verify
            for c in processed_clauses:
                if not is_clause_satisfied(c, assign):
                    return None
            return assign[:]

        # Select variable using VSIDS-like heuristic
        # Count occurrences in unsatisfied clauses
        var_counts = [0] * n
        for c in processed_clauses:
            if is_clause_satisfied(c, assign):
                continue
            for lit in c:
                v = abs(lit) - 1
                if assign[v] == 0:
                    var_counts[v] += 1

        best_var = -1
        best_count = -1
        for v in range(n):
            if assign[v] == 0 and var_counts[v] > best_count:
                best_count = var_counts[v]
                best_var = v

        if best_var == -1:
            return None

        # Branch True
        assign[best_var] = 1
        res = dpll_simple(assign, depth + 1)
        if res:
            return res
        assign[best_var] = 0

        # Branch False
        assign[best_var] = -1
        res = dpll_simple(assign, depth + 1)
        if res:
            return res
        assign[best_var] = 0

        return None

    # Run Simple DPLL
    result = dpll_simple(assignment_simple)

    if result:
        return [a == 1 for a in result]

    # Fallback: Simulated Annealing
    remaining_time = DEADLINE - time.time()
    if remaining_time < 10:
        # Not enough time, return random
        rng = random.Random(42)
        return [rng.random() < 0.5 for _ in range(n)]

    rng = random.Random(42)

    # Initialize with random assignment
    current_assign = [rng.random() < 0.5 for _ in range(n)]

    def count_unsat_sa(assign):
        cnt = 0
        for c in processed_clauses:
            sat = False
            for lit in c:
                v = abs(lit) - 1
                if (lit > 0 and assign[v]) or (lit < 0 and not assign[v]):
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt

    best_assign = current_assign[:]
    best_uc = count_unsat_sa(current_assign)

    # Simulated Annealing parameters
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999

    max_steps = int(remaining_time * 50000)  # ~50k steps per second estimate

    for step in range(max_steps):
        if time.time() > DEADLINE - 2:
            break
        if best_uc == 0:
            return best_assign

        # Pick a random variable to flip
        var = rng.randint(0, n - 1)
        old_val = current_assign[var]
        current_assign[var] = not current_val

        new_uc = count_unsat_sa(current_assign)
        delta = new_uc - best_uc

        # Accept if better or with probability exp(-delta/T)
        if delta < 0:
            best_uc = new_uc
            best_assign = current_assign[:]
        else:
            if rng.random() < pow(2.71828, -delta / T):
                pass  # Accept
            else:
                current_assign[var] = old_val  # Reject

        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min

        # Occasionally restart with a new temperature if stuck
        if step % 10000 == 0 and T <= T_min:
            T = 0.5
            current_assign = best_assign[:]

    return best_assign
