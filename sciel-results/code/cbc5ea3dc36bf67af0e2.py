"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 285  # safety margin before 330s kill

    # Build clause list
    cnf = [tuple(c) for c in clauses]
    m = len(cnf)

    # Preprocess: unit propagation and pure literal elimination
    assignment = [None] * n
    cnf_work = [list(c) for c in cnf]
    changed = True
    while changed:
        changed = False
        # Remove satisfied clauses
        new_cnf = []
        for c in cnf_work:
            sat = False
            for lit in c:
                vi = abs(lit) - 1
                if assignment[vi] is not None and assignment[vi] == (lit > 0):
                    sat = True
                    break
            if not sat:
                new_cnf.append(c)
        cnf_work = new_cnf

        # Unit propagation
        for i, c in enumerate(cnf_work):
            if len(c) == 1:
                lit = c[0]
                vi = abs(lit) - 1
                if assignment[vi] is None:
                    assignment[vi] = (lit > 0)
                    changed = True
                elif assignment[vi] != (lit > 0):
                    return [False] * n  # contradiction, return dummy
            elif len(c) == 0:
                return [False] * n  # contradiction

        # Pure literal elimination
        lit_counts = [[0, 0] for _ in range(n)]
        for c in cnf_work:
            seen = set()
            for lit in c:
                vi = abs(lit) - 1
                if vi in seen:
                    continue
                seen.add(vi)
                if lit > 0:
                    lit_counts[vi][0] += 1
                else:
                    lit_counts[vi][1] += 1

        for vi in range(n):
            if assignment[vi] is not None:
                continue
            if lit_counts[vi][0] == 0 and lit_counts[vi][1] > 0:
                assignment[vi] = False
                changed = True
            elif lit_counts[vi][1] == 0 and lit_counts[vi][0] > 0:
                assignment[vi] = True
                changed = True

    # Check if all assigned
    if all(a is not None for a in assignment):
        final_assign = [bool(a) for a in assignment]
        # Verify
        ok = True
        for c in cnf:
            sat = False
            for lit in c:
                vi = abs(lit) - 1
                if final_assign[vi] == (lit > 0):
                    sat = True
                    break
            if not sat:
                ok = False
                break
        if ok:
            return final_assign

    # Build numpy bitmasks for fast evaluation
    # For each variable, create bitmask of clauses where it appears positively/negatively
    # pos_mask[vi] = bitmask of clauses containing +vi
    # neg_mask[vi] = bitmask of clauses containing -vi
    # We need to handle m potentially > 64, so use multiple uint64 words
    num_words = (m + 63) // 64

    pos_masks = [np.zeros(num_words, dtype=np.uint64) for _ in range(n)]
    neg_masks = [np.zeros(num_words, dtype=np.uint64) for _ in range(n)]

    for ci, c in enumerate(cnf):
        word_idx = ci // 64
        bit = ci % 64
        bitval = np.uint64(1) << np.uint64(bit)
        for lit in c:
            vi = abs(lit) - 1
            if lit > 0:
                pos_masks[vi][word_idx] |= bitval
            else:
                neg_masks[vi][word_idx] |= bitval

    # Also build clause literal info for local search
    clause_lits = []  # for each clause, list of (var_idx, sign)
    for c in cnf:
        clause_lits.append([(abs(lit) - 1, 1 if lit > 0 else -1) for lit in c])

    # Build var_to_clauses
    var_to_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(cnf):
        for lit in c:
            vi = abs(lit) - 1
            var_to_clauses[vi].append(ci)

    def is_satisfied(assign):
        """Check if assignment satisfies all clauses using numpy bitmasks."""
        # For each clause, check if any literal is satisfied
        # satisfied_mask = OR over all vars of (assign[vi] ? pos_mask[vi] : neg_mask[vi])
        # A clause is satisfied if its bit is set in satisfied_mask
        sat_mask = np.zeros(num_words, dtype=np.uint64)
        for vi in range(n):
            if assign[vi]:
                sat_mask |= pos_masks[vi]
            else:
                sat_mask |= neg_masks[vi]

        # Check all clauses are covered
        for w in range(num_words):
            if w == num_words - 1 and m % 64 != 0:
                # Last word: only check m%64 bits
                mask = (1 << (m % 64)) - 1
                if (sat_mask[w] & mask) != mask:
                    return False
            else:
                if sat_mask[w] != np.uint64(0xFFFFFFFFFFFFFFFF):
                    return False
        return True

    def count_unsat(assign):
        """Count unsatisfied clauses using numpy bitmasks."""
        sat_mask = np.zeros(num_words, dtype=np.uint64)
        for vi in range(n):
            if assign[vi]:
                sat_mask |= pos_masks[vi]
            else:
                sat_mask |= neg_masks[vi]

        unsat = 0
        for w in range(num_words):
            if w == num_words - 1 and m % 64 != 0:
                mask = (1 << (m % 64)) - 1
                unsat += bin(mask & ~sat_mask[w]).count('1')
            else:
                unsat += bin(~sat_mask[w] & 0xFFFFFFFFFFFFFFFF).count('1')
        return unsat

    def get_unsat_clauses(assign):
        """Get list of unsatisfied clause indices."""
        sat_mask = np.zeros(num_words, dtype=np.uint64)
        for vi in range(n):
            if assign[vi]:
                sat_mask |= pos_masks[vi]
            else:
                sat_mask |= neg_masks[vi]

        unsat_list = []
        for w in range(num_words):
            if w == num_words - 1 and m % 64 != 0:
                mask = (1 << (m % 64)) - 1
                unsat_bits = mask & ~sat_mask[w]
            else:
                unsat_bits = ~sat_mask[w] & 0xFFFFFFFFFFFFFFFF

            if unsat_bits:
                for b in range(64):
                    if w * 64 + b < m and (unsat_bits >> b) & 1:
                        unsat_list.append(w * 64 + b)
        return unsat_list

    def greedy_local_search(assign, max_steps, rng):
        """Greedy local search: flip variable that maximizes net gain in satisfied clauses."""
        cur = assign[:]
        cur_unsat = count_unsat(cur)
        min_unsat = cur_unsat
        min_assign = cur[:]

        for step in range(max_steps):
            if time.time() > deadline:
                break
            if cur_unsat == 0:
                return True, cur, 0

            # Find variable with best flip
            best_gain = -1
            best_var = -1
            best_ties = []

            for vi in range(n):
                # Calculate gain from flipping vi
                if cur[vi]:
                    # Currently True, would become False
                    # Lose satisfaction of clauses where +vi is the only satisfied literal
                    # Gain satisfaction of clauses where -vi is the only unsatisfied literal
                    # Simplified: compute net change
                    old_sat = pos_masks[vi]
                    new_sat = neg_masks[vi]
                    # Clauses that lose satisfaction: in old_sat but not in new_sat, and no other lit satisfies
                    # This is complex; use approximate: count clauses where this var is critical
                    # For simplicity, use a heuristic: degree-based
                    pass

                # Use a simpler approach: evaluate gain by checking affected clauses
                affected = var_to_clauses[vi]
                gain = 0
                for aci in affected:
                    # Current satisfaction of clause aci
                    old_sat = False
                    for vi2, sign2 in clause_lits[aci]:
                        if cur[vi2] == (sign2 > 0):
                            old_sat = True
                            break
                    # New satisfaction if we flip vi
                    new_val = not cur[vi]
                    new_sat = False
                    for vi2, sign2 in clause_lits[aci]:
                        if vi2 == vi:
                            if new_val == (sign2 > 0):
                                new_sat = True
                                break
                        else:
                            if cur[vi2] == (sign2 > 0):
                                new_sat = True
                                break
                    if not old_sat and new_sat:
                        gain += 1
                    elif old_sat and not new_sat:
                        gain -= 1

                if gain > best_gain:
                    best_gain = gain
                    best_var = vi
                    best_ties = [vi]
                elif gain == best_gain and gain > 0:
                    best_ties.append(vi)

            if best_var == -1 or best_gain <= 0:
                # No improving move; try random flip or break
                if rng.random() < 0.3:
                    vi = rng.randrange(n)
                    cur[vi] = not cur[vi]
                    cur_unsat = count_unsat(cur)
                else:
                    break
            else:
                if len(best_ties) > 1:
                    best_var = rng.choice(best_ties)
                cur[best_var] = not cur[best_var]
                cur_unsat = count_unsat(cur)

            if cur_unsat < min_unsat:
                min_unsat = cur_unsat
                min_assign = cur[:]

        return min_unsat == 0, (cur if min_unsat == 0 else min_assign), min_unsat

    def walksat_fast(assign, max_steps, p_bias, rng):
        """WalkSAT with numpy-accelerated clause evaluation."""
        cur = assign[:]
        cur_unsat = count_unsat(cur)
        min_unsat = cur_unsat
        min_assign = cur[:]

        for step in range(max_steps):
            if time.time() > deadline:
                break
            if cur_unsat == 0:
                return True, cur, 0

            # Get unsatisfied clauses
            unsat_clauses = get_unsat_clauses(cur)
            if not unsat_clauses:
                return True, cur, 0

            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            clause = cnf[ci]

            # Pick a literal from the clause
            lit = rng.choice(clause)
            vi = abs(lit) - 1

            # With probability p_bias, set to satisfy; else random flip
            if rng.random() < p_bias:
                cur[vi] = (lit > 0)
            else:
                cur[vi] = not cur[vi]

            cur_unsat = count_unsat(cur)

            if cur_unsat < min_unsat:
                min_unsat = cur_unsat
                min_assign = cur[:]

        return min_unsat == 0, (cur if min_unsat == 0 else min_assign), min_unsat

    # Initial assignment
    start_assign = []
    for i in range(n):
        if assignment[i] is not None:
            start_assign.append(bool(assignment[i]))
        else:
            start_assign.append(random.Random(i).random() < 0.5)

    if is_satisfied(start_assign):
        return start_assign

    best_assign = start_assign[:]
    best_unsat = count_unsat(start_assign)

    # Multi-restart with mixed strategies
    attempt = 0
    seeds = [42, 123, 7, 999, 5555, 12345, 67890, 24680, 13579, 98765]
    p_bias_values = [0.8, 0.75, 0.7, 0.85, 0.6, 0.9, 0.65, 0.78, 0.55, 0.82]

    while time.time() < deadline - 5:
        seed = seeds[attempt % len(seeds)]
        p_bias = p_bias_values[attempt % len(p_bias_values)]
        rng = random.Random(seed)
        steps = min(5000 + attempt * 1000, 50000)

        # Alternate between greedy and walksat
        if attempt % 2 == 0:
            success, result, min_unsat = greedy_local_search(best_assign, steps, rng)
        else:
            success, result, min_unsat = walksat_fast(best_assign, steps, p_bias, rng)

        if success:
            return result

        if min_unsat < best_unsat:
            best_unsat = min_unsat
            best_assign = result[:]

        attempt += 1

        # Periodic fresh restart
        if attempt % 10 == 0:
            fresh = [random.Random(attempt * 7).random() < 0.5 for _ in range(n)]
            fu = count_unsat(fresh)
            if fu < best_unsat:
                best_assign = fresh
                best_unsat = fu

    # Final check
    if is_satisfied(best_assign):
        return best_assign

    return best_assign

# EVOLVE-BLOCK-END
