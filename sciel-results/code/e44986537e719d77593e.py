"""
Optimized WalkSAT with fast clause tracking, adaptive noise, and final polish.
"""
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Pre-compute: for each variable, list of (clause_index, sign)
    # sign = 1 if literal is +var, -1 if literal is -var
    var_occurs = [[] for _ in range(n)]
    for ci, cl in enumerate(clauses):
        for lit in cl:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_occurs[v].append((ci, sign))

    # Convert to 0-based internal representation for speed
    # We'll work with literal indices: lit_idx = (var << 1) | (0 if positive else 1)
    # But let's keep it simple with the existing structure but optimize the inner loop.

    deadline = time.time() + 285  # 330s limit, keep 45s safety margin

    best_assignment = None
    best_unsat_count = m + 1

    rng = random.Random()

    # Adaptive parameters
    alpha = 0.35  # noise level
    max_steps_per_restart = 50000

    # We'll use a simple WalkSAT with good variable selection heuristic
    # Key optimization: maintain clause_sat array and update incrementally

    for restart in range(100000):  # Large budget, limited by time
        if time.time() > deadline - 2:
            break

        # Random initial assignment
        assignment = [rng.getrandbits(1) for _ in range(n)]

        # Compute initial clause satisfaction counts
        clause_sat = [0] * m
        for ci, cl in enumerate(clauses):
            s = 0
            for lit in cl:
                v = abs(lit) - 1
                sign = 1 if lit > 0 else -1
                desired = 1 if sign == 1 else 0
                if assignment[v] == desired:
                    s += 1
            clause_sat[ci] = s

        unsat_count = sum(1 for s in clause_sat if s == 0)
        if unsat_count == 0:
            return assignment

        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = list(assignment)

        # WalkSAT loop
        steps = 0
        max_steps = max_steps_per_restart
        assign = assignment
        c_sat = clause_sat
        var_occ = var_occurs
        n_vars = n

        # Track unsat clause indices for fast selection
        # We'll use a simple approach: pick random clause, check if unsat
        # But for speed, let's maintain a list of unsat clause indices
        # Actually, for 150 vars and ~300-400 clauses, random selection is fine

        while steps < max_steps:
            if time.time() > deadline - 2:
                break

            # Pick an unsat clause
            # Try random selection first, fallback to scan if many tries
            ci = -1
            for _ in range(10):
                ci = rng.randrange(m)
                if c_sat[ci] == 0:
                    break
            if c_sat[ci] > 0:
                # Scan to find an unsat clause
                unsat_list = [i for i in range(m) if c_sat[i] == 0]
                if not unsat_list:
                    break
                ci = rng.choice(unsat_list)

            cl = clauses[ci]

            # Compute scores for each literal in the clause
            # For each variable in the clause, compute gain - loss
            scores = []
            for lit in cl:
                v = abs(lit) - 1
                sign = 1 if lit > 0 else -1

                gain = 0
                loss = 0
                for (cci, sgn) in var_occ[v]:
                    if cci == ci:
                        continue
                    # Current contribution of variable v to clause cci
                    cur_desired = 1 if sgn == 1 else 0
                    cur_val = 1 if (assign[v] == cur_desired) else 0
                    # New contribution after flipping v
                    new_val = 1 - cur_val

                    if cur_val == 0 and new_val == 1:
                        # This flip would satisfy a currently unsat clause
                        if c_sat[cci] == 0:
                            gain += 1
                    elif cur_val == 1 and new_val == 0:
                        # This flip would break a currently sat clause
                        if c_sat[cci] == 1:
                            loss += 1

                score = gain - loss
                scores.append((score, v, sign))

            # Noisy selection
            r = rng.random()
            if r < alpha:
                idx = rng.randrange(len(cl))
                _, v, sign = scores[idx]
            else:
                # Pick max score with random tie-breaking
                max_score = max(s[0] for s in scores)
                top = [i for i, s in enumerate(scores) if s[0] == max_score]
                idx = rng.choice(top)
                _, v, sign = scores[idx]

            # Flip variable v
            old_val = assign[v]
            new_val = 1 - old_val
            assign[v] = new_val

            # Update clause_sat for all clauses containing v
            for (cci, sgn) in var_occ[v]:
                old_contrib = 1 if (old_val == (1 if sgn == 1 else 0)) else 0
                new_contrib = 1 if (new_val == (1 if sgn == 1 else 0)) else 0
                c_sat[cci] += new_contrib - old_contrib

            steps += 1

            # Periodic check
            if steps % 10 == 0:
                unsat_count = sum(1 for s in c_sat if s == 0)
                if unsat_count == 0:
                    break
                if unsat_count < best_unsat_count:
                    best_unsat_count = unsat_count
                    best_assignment = list(assign)

        # End of walksat loop
        unsat_count = sum(1 for s in c_sat if s == 0)
        if unsat_count == 0:
            return assignment

        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = list(assign)

        # Decay alpha slightly
        alpha = max(0.15, alpha * 0.99)

    # Final polish: try to fix remaining unsat clauses with targeted flips
    if best_assignment is not None:
        final_assign = list(best_assignment)
        final_c_sat = [0] * m
        for ci, cl in enumerate(clauses):
            s = 0
            for lit in cl:
                v = abs(lit) - 1
                sign = 1 if lit > 0 else -1
                desired = 1 if sign == 1 else 0
                if final_assign[v] == desired:
                    s += 1
            final_c_sat[ci] = s

        # Try multiple rounds of targeted fixes
        for _ in range(200):
            unsat_cis = [i for i in range(m) if final_c_sat[i] == 0]
            if not unsat_cis:
                break
            ci = rng.choice(unsat_cis)
            cl = clauses[ci]
            # Try flipping each var in the clause, pick best
            best_v = -1
            best_gain = -999
            for lit in cl:
                v = abs(lit) - 1
                gain = 0
                loss = 0
                for (cci, sgn) in var_occurs[v]:
                    if cci == ci:
                        continue
                    cur_desired = 1 if sgn == 1 else 0
                    cur_val = 1 if (final_assign[v] == cur_desired) else 0
                    new_val = 1 - cur_val
                    if cur_val == 0 and new_val == 1:
                        if final_c_sat[cci] == 0:
                            gain += 1
                    elif cur_val == 1 and new_val == 0:
                        if final_c_sat[cci] == 1:
                            loss += 1
                net = gain - loss
                if net > best_gain:
                    best_gain = net
                    best_v = v
            if best_v >= 0 and best_gain > 0:
                old_val = final_assign[best_v]
                final_assign[best_v] = 1 - old_val
                for (cci, sgn) in var_occurs[best_v]:
                    old_contrib = 1 if (old_val == (1 if sgn == 1 else 0)) else 0
                    new_contrib = 1 if ((1 - old_val) == (1 if sgn == 1 else 0)) else 0
                    final_c_sat[cci] += new_contrib - old_contrib

        final_unsat = sum(1 for s in final_c_sat if s == 0)
        if final_unsat == 0:
            return final_assign
        if final_unsat < best_unsat_count:
            best_unsat_count = final_unsat
            best_assignment = final_assign

    return best_assignment
