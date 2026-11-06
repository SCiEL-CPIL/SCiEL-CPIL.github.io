"""Improved planted 3-SAT solver using bitmask-accelerated WalkSAT with greedy refinement."""
# EVOLVE-BLOCK-START
import random
import time
from math import log, exp

DEADLINE_FRACTION = 0.95  # safety margin: stop at 95% of 330s


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Deadline guard
    t_start = time.time()
    deadline = t_start + 330 * DEADLINE_FRACTION

    # Precompute bitmasks: for each variable, which clauses contain its positive/negative literal
    # pos_mask[v]: bitmask of clauses where literal +v appears
    # neg_mask[v]: bitmask of clauses where literal -v appears
    # Also for each clause, store its literals for fast satisfaction checking
    # We'll use Python integers as arbitrary-precision bitmasks

    # Build occurrence maps
    pos_mask = [0] * n
    neg_mask = [0] * n
    # clause_literals: list of (var, is_positive) for each clause
    clause_vars = []  # clause_vars[c] = list of (var_idx, sign) where sign=True means positive

    for ci, clause in enumerate(clauses):
        bit = 1 << ci
        for lit in clause:
            v = abs(lit) - 1
            if lit > 0:
                pos_mask[v] |= bit
            else:
                neg_mask[v] |= bit
        clause_vars.append([(abs(l) - 1, l > 0) for l in clause])

    # Helper: given an assignment (list of bools), compute the set of unsatisfied clauses as a bitmask
    def get_unsat_mask(assignment):
        """Return bitmask of unsatisfied clauses."""
        unsat = 0
        for ci in range(m):
            cvars = clause_vars[ci]
            sat = False
            for v, pos in cvars:
                if assignment[v] == pos:
                    sat = True
                    break
            if not sat:
                unsat |= (1 << ci)
        return unsat

    # Faster: compute satisfied count using bitmasks
    # A clause is satisfied if for some literal in it, the variable is set accordingly.
    # Alternative: for each variable, the clauses it satisfies are:
    #   if assignment[v] == True:  pos_mask[v]
    #   if assignment[v] == False: neg_mask[v]
    # The union of all these over all variables gives the set of satisfied clauses.
    # unsat = all_clauses_mask ^ satisfied_mask

    all_mask = (1 << m) - 1

    def satisfied_mask(assignment):
        sm = 0
        for v in range(n):
            if assignment[v]:
                sm |= pos_mask[v]
            else:
                sm |= neg_mask[v]
        return sm

    def is_satisfied(assignment):
        return satisfied_mask(assignment) == all_mask

    # Greedy flip gain: flipping variable v changes which clauses are satisfied.
    # Before flip: satisfied clauses from v = pos_mask[v] if assignment[v] else neg_mask[v]
    # After flip:  satisfied clauses from v = neg_mask[v] if assignment[v] else pos_mask[v]
    # But other variables may also satisfy those clauses, so we need to check carefully.
    # Net gain = (clauses newly satisfied) - (clauses newly unsatisfied)
    # A clause becomes newly satisfied if it was unsatisfied and the flipped literal now satisfies it
    #   AND no other literal in that clause was already satisfying it (which it wasn't, since it was unsat)
    # A clause becomes newly unsatisfied if it was satisfied only by v's literal and no other literal satisfies it.

    def flip_gain(v, assignment):
        """Return net gain in satisfied clauses if we flip variable v."""
        current_val = assignment[v]
        if current_val:
            # Currently satisfying via pos_mask[v], after flip via neg_mask[v]
            newly_satisfied = neg_mask[v] & ~satisfied_mask(assignment)
            # Clauses that were satisfied only by v's positive literal:
            # They are in pos_mask[v] and NOT satisfied by any other variable
            # A clause c in pos_mask[v] is satisfied only by v if:
            #   satisfied_mask(assignment) has c, but if we remove v's contribution, c is not in the rest
            # This is complex. Let's use a simpler approach:
            # gain = popcount(neg_mask[v] & unsat_mask) - popcount(pos_mask[v] & unsat_after_flip_mask)
            # Too complex. Let's just compute directly for the clauses involved.
            pass

        # Simpler approach: compute gain by checking each clause in the union of pos_mask[v] and neg_mask[v]
        # But that's O(m) per flip. For 50 vars and maybe 200-500 clauses, it's manageable.

        # Let's use a direct approach: for each clause in the affected set, check if flip helps or hurts.
        affected = pos_mask[v] | neg_mask[v]
        gain = 0
        # Get current unsat mask
        cur_sat = satisfied_mask(assignment)
        cur_unsat = all_mask ^ cur_sat

        # Clauses that might change: those in affected
        # For each such clause, determine if it's currently sat/unsat and what it'll be after flip
        # We can iterate over set bits in affected
        aff = affected
        while aff:
            lsb = aff & (-aff)
            ci = lsb.bit_length() - 1
            aff ^= lsb
            cvars = clause_vars[ci]
            # Check current satisfaction
            cur_sat_c = False
            for vv, pos in cvars:
                if vv == v:
                    continue  # skip v, check others
                if assignment[vv] == pos:
                    cur_sat_c = True
                    break
            # After flip, v's contribution:
            new_v_val = not current_val
            v_satisfies_after = (new_v_val == [p for vv, p in cvars if vv == v][0]) if any(vv == v for vv, p in cvars) else False

            # Actually, v must be in this clause since ci is in affected
            v_pos_in_clause = None
            for vv, pos in cvars:
                if vv == v:
                    v_pos_in_clause = pos
                    break

            if v_pos_in_clause is None:
                continue

            # Currently: clause is sat if (assignment[v] == v_pos_in_clause) or cur_sat_c
            cur_clause_sat = (assignment[v] == v_pos_in_clause) or cur_sat_c
            # After flip: clause is sat if (not assignment[v] == v_pos_in_clause) or cur_sat_c
            new_clause_sat = (not assignment[v] == v_pos_in_clause) or cur_sat_c

            if new_clause_sat and not cur_clause_sat:
                gain += 1
            elif cur_clause_sat and not new_clause_sat:
                gain -= 1

        return gain

    # Initialize assignment
    rng = random.Random(42)
    assignment = [rng.random() < 0.5 for _ in range(n)]

    best_assignment = assignment[:]
    best_unsat_count = m - bin(satisfied_mask(assignment)).count('1')

    # WalkSAT with greedy refinement
    # Parameters tuned for planted 3-SAT with 50 variables
    walksat_restarts = 200
    max_steps_per_restart = 5000
    noise_param = 0.25  # probability of random flip vs greedy

    for restart in range(walksat_restarts):
        if time.time() > deadline:
            break

        # Random restart assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        cur_unsat = m - bin(satisfied_mask(assignment)).count('1')

        if cur_unsat == 0:
            return assignment

        if cur_unsat < best_unsat_count:
            best_unsat_count = cur_unsat
            best_assignment = assignment[:]

        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break

            cur_sat_mask = satisfied_mask(assignment)
            cur_unsat_mask = all_mask ^ cur_sat_mask
            cur_unsat = bin(cur_unsat_mask).count('1')

            if cur_unsat == 0:
                return assignment

            if cur_unsat < best_unsat_count:
                best_unsat_count = cur_unsat
                best_assignment = assignment[:]

            # Choose a variable to flip
            if rng.random() < noise_param:
                # Random flip
                v = rng.randint(0, n - 1)
            else:
                # Greedy: find flip with max gain
                best_gain = -1
                best_v = -1
                best_gain_v = -1
                for v in range(n):
                    g = flip_gain(v, assignment)
                    if g > best_gain or (g == best_gain and rng.random() < 0.5):
                        best_gain = g
                        best_v = v
                v = best_v

            assignment[v] = not assignment[v]

    # Final polish: try local search from best_assignment
    if time.time() < deadline:
        assignment = best_assignment[:]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            cur_unsat = m - bin(satisfied_mask(assignment)).count('1')
            if cur_unsat == 0:
                return assignment
            for v in range(n):
                g = flip_gain(v, assignment)
                if g > 0:
                    assignment[v] = not assignment[v]
                    improved = True
                    break

    return best_assignment
# EVOLVE-BLOCK-END
