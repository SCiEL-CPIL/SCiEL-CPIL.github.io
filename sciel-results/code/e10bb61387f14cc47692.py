"""Synthetic planted 3-SAT solver: SA over Max-SAT objective + WalkSAT polish."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 285  # safety margin before 300s kill

    # ---- Precomputation for O(1) delta-score evaluation ----
    # For each variable i:
    #   pos_clauses[i]  = list of clause indices containing literal +i
    #   neg_clauses[i]  = list of clause indices containing literal -i
    # For each clause c:
    #   clause_vars[c] = list of (var_idx, polarity) where polarity 1 => +var
    #   clause_sat_count[c] = current number of satisfied literals (0..3)
    pos_clauses = [[] for _ in range(n)]
    neg_clauses = [[] for _ in range(n)]
    clause_vars = []
    for c_idx, cl in enumerate(clauses):
        vars_list = []
        for lit in cl:
            v = abs(lit) - 1
            vars_list.append((v, 1 if lit > 0 else -1))
            if lit > 0:
                pos_clauses[v].append(c_idx)
            else:
                neg_clauses[v].append(c_idx)
        clause_vars.append(vars_list)

    clause_sat_count = [0] * m  # current satisfied-literal count per clause

    def init_sat_count(assignment):
        for c_idx in range(m):
            cnt = 0
            for v, pol in clause_vars[c_idx]:
                if assignment[v] == (pol == 1):
                    cnt += 1
            clause_sat_count[c_idx] = cnt

    def total_satisfied():
        # A clause is satisfied iff sat_count >= 1
        cnt = 0
        for c in range(m):
            if clause_sat_count[c] >= 1:
                cnt += 1
        return cnt

    def delta_score(var_idx, new_val):
        """
        Compute change in total satisfied-clause count if we set variable var_idx to new_val.
        Assumes we know current value; we compute based on current clause_sat_count.
        Returns (delta, list_of_clauses_affected) -- but we only need delta for SA.
        We compute it without modifying state.
        """
        cur_val = assignment[var_idx]
        if cur_val == new_val:
            return 0
        # Determine which clauses are affected: those containing this variable
        # with the polarity that matches the OLD value (satisfied by old) or NEW value.
        # Actually: a clause's sat_count changes iff it contains this variable.
        # For each affected clause, compute old contribution and new contribution.
        # We need to iterate over clauses containing var_idx (either polarity).
        affected = pos_clauses[var_idx] + neg_clauses[var_idx]
        # Deduplicate (a clause can't contain both +v and -v in 3-SAT, but be safe)
        # Build a set for speed if needed; for n=50, lists are short.
        delta = 0
        seen = set()
        for c_idx in affected:
            if c_idx in seen:
                continue
            seen.add(c_idx)
            old_cnt = clause_sat_count[c_idx]
            # Compute new contribution of this variable to the clause
            # Find the literal for var_idx in this clause
            new_contrib = 0
            for v, pol in clause_vars[c_idx]:
                if v == var_idx:
                    if new_val == (pol == 1):
                        new_contrib = 1
                    else:
                        new_contrib = 0
                    break
            # Old contribution
            old_contrib = 0
            for v, pol in clause_vars[c_idx]:
                if v == var_idx:
                    if cur_val == (pol == 1):
                        old_contrib = 1
                    else:
                        old_contrib = 0
                    break
            new_cnt = old_cnt - old_contrib + new_contrib
            # Clause satisfied status change
            old_sat = 1 if old_cnt >= 1 else 0
            new_sat = 1 if new_cnt >= 1 else 0
            delta += new_sat - old_sat
        return delta

    # ---- Simulated Annealing over Max-SAT ----
    def simulated_annealing(start_assignment, rng, max_steps, deadline):
        nonlocal clause_sat_count
        assignment = start_assignment[:]
        init_sat_count(assignment)
        current_score = total_satisfied()
        best_assignment = assignment[:]
        best_score = current_score

        if best_score == m:
            return assignment, best_assignment, True

        start_temp = 1.0
        end_temp = 0.01
        # Geometric cooling
        ratio = (end_temp / start_temp) ** (1.0 / max(1, max_steps))

        for step in range(max_steps):
            if time.time() > deadline - 3:
                break

            temp = start_temp * (ratio ** step)

            # Pick a random variable to flip
            var_idx = rng.randint(0, n - 1)
            new_val = not assignment[var_idx]

            delta = delta_score(var_idx, new_val)

            if delta >= 0:
                # Accept: update state
                old_val = assignment[var_idx]
                assignment[var_idx] = new_val
                # Update clause_sat_count for affected clauses
                affected = pos_clauses[var_idx] + neg_clauses[var_idx]
                seen = set()
                for c_idx in affected:
                    if c_idx in seen:
                        continue
                    seen.add(c_idx)
                    # Recompute sat count for this clause
                    cnt = 0
                    for v, pol in clause_vars[c_idx]:
                        if assignment[v] == (pol == 1):
                            cnt += 1
                    clause_sat_count[c_idx] = cnt
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_assignment = assignment[:]
                    if best_score == m:
                        return assignment, best_assignment, True
            else:
                # Worsening: accept with probability exp(delta / temp)
                if temp > 0:
                    # delta is negative, so delta/temp is negative
                    prob = math.exp(delta / temp)
                    if rng.random() < prob:
                        old_val = assignment[var_idx]
                        assignment[var_idx] = new_val
                        affected = pos_clauses[var_idx] + neg_clauses[var_idx]
                        seen = set()
                        for c_idx in affected:
                            if c_idx in seen:
                                continue
                            seen.add(c_idx)
                            cnt = 0
                            for v, pol in clause_vars[c_idx]:
                                if assignment[v] == (pol == 1):
                                    cnt += 1
                            clause_sat_count[c_idx] = cnt
                        current_score += delta
                    # else: reject, state unchanged
        return assignment, best_assignment, best_score == m

    # ---- WalkSAT polish (O(1) damage via precomputed data) ----
    def get_unsat_clauses(assignment):
        return [c for c in range(m) if clause_sat_count[c] == 0]

    def walksat_polish(assignment, rng, max_steps, deadline):
        nonlocal clause_sat_count
        init_sat_count(assignment)
        unsat = get_unsat_clauses(assignment)
        if not unsat:
            return assignment, True

        best_assignment = assignment[:]
        best_unsat = len(unsat)

        for step in range(max_steps):
            if time.time() > deadline - 3:
                break
            if not unsat:
                return assignment, True

            if len(unsat) < best_unsat:
                best_unsat = len(unsat)
                best_assignment = assignment[:]

            # Pick random unsat clause
            c_idx = rng.choice(unsat)

            # For each literal in the clause, compute damage (how many other unsat-able clauses break)
            best_literal_var = None
            best_literal_pol = None
            best_damage = float('inf')

            for v, pol in clause_vars[c_idx]:
                # If we flip v, what's the damage?
                # Damage = number of clauses (other than c_idx) that are currently satisfied
                # ONLY by v with its current value, and would become unsatisfied.
                # Using precomputed data: for each clause containing v, check if it's satisfied
                # only by v.
                damage = 0
                affected = pos_clauses[v] + neg_clauses[v]
                seen = set()
                cur_val = assignment[v]
                for ac in affected:
                    if ac == c_idx or ac in seen:
                        continue
                    seen.add(ac)
                    # Is this clause currently satisfied?
                    if clause_sat_count[ac] == 0:
                        continue  # already unsat, flipping can't break it (it stays unsat or becomes sat)
                    # Check if it's satisfied ONLY by v
                    # Count how many literals are satisfied
                    # We know sat_count >= 1. If sat_count == 1, check if v is the one satisfying it.
                    # Actually, we need: after flipping v, does this clause become unsat?
                    # That happens iff currently sat_count[ac] == 1 AND v is the satisfying literal.
                    # OR more generally: new_sat_count = sat_count - (1 if v currently satisfies) + (1 if v new satisfies)
                    # If new_sat_count == 0, it breaks.
                    old_contrib = 1 if cur_val == (pol == 1) else 0
                    new_val = not cur_val
                    new_contrib = 1 if new_val == (pol == 1) else 0
                    new_cnt = clause_sat_count[ac] - old_contrib + new_contrib
                    if new_cnt == 0:
                        damage += 1

                if damage < best_damage:
                    best_damage = damage
                    best_literal_var = v
                    best_literal_pol = pol

            if best_literal_var is None:
                v, pol = rng.choice(clause_vars[c_idx])
                best_literal_var = v
                best_literal_pol = pol

            # Flip the variable
            assignment[best_literal_var] = not assignment[best_literal_var]

            # Update clause_sat_count for all affected clauses
            affected = pos_clauses[best_literal_var] + neg_clauses[best_literal_var]
            seen = set()
            for ac in affected:
                if ac in seen:
                    continue
                seen.add(ac)
                cnt = 0
                for v2, pol2 in clause_vars[ac]:
                    if assignment[v2] == (pol2 == 1):
                        cnt += 1
                clause_sat_count[ac] = cnt

            unsat = get_unsat_clauses(assignment)

        return best_assignment, best_unsat == 0

    # ---- Main search loop ----
    rng_main = random.Random(42)

    best_assignment = None
    best_score = -1

    # Phase 1: Simulated Annealing with multiple restarts
    num_sa_restarts = 20
    sa_steps_per_restart = 20000

    for restart in range(num_sa_restarts):
        if time.time() > deadline - 10:
            break

        rng = random.Random(restart * 7919 + 1)
        assignment = [rng.random() < 0.5 for _ in range(n)]

        sa_result, sa_best, sa_solved = simulated_annealing(
            assignment, rng, sa_steps_per_restart, deadline
        )

        if sa_solved:
            return sa_best

        # Track best
        init_sat_count(sa_best)
        score = total_satisfied()
        if score > best_score:
            best_score = score
            best_assignment = sa_best[:]

        # Phase 2: WalkSAT polish from SA result
        if time.time() > deadline - 5:
            break
        polish_rng = random.Random(restart * 104729 + 3)
        polish_result, polish_solved = walksat_polish(
            sa_best, polish_rng, 5000, deadline
        )
        if polish_solved:
            return polish_result

        init_sat_count(polish_result)
        p_score = total_satisfied()
        if p_score > best_score:
            best_score = p_score
            best_assignment = polish_result[:]

    # Final fallback: pure WalkSAT from random
    if time.time() < deadline - 5:
        rng_fallback = random.Random(999983)
        assignment = [rng_fallback.random() < 0.5 for _ in range(n)]
        result, solved = walksat_polish(assignment, rng_fallback, 20000, deadline)
        if solved:
            return result
        init_sat_count(result)
        score = total_satisfied()
        if score > best_score:
            best_score = score
            best_assignment = result[:]

    if best_assignment is not None:
        return best_assignment

    # Absolute fallback
    return [False] * n
# EVOLVE-BLOCK-END
