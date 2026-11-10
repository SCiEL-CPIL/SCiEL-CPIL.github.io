"""WalkSAT + smooth soft-min pressure polish for planted 3-SAT (50 vars).

Combines the incremental WalkSAT core (phase-saving restarts, clause weights,
adaptive P) with a smooth log-sum-exp surrogate for clause-satisfaction
pressure. The soft-min of a clause's per-literal "distance to satisfied"
gives a differentiable proxy for how 'close' an unsatisfied clause is to
being satisfied, which we use to bias clause selection and to drive a
final high-precision polish pass that can escape local minima pure-greedy
polish cannot.
"""
# EVOLVE-BLOCK-START
import time
import random
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290.0  # safety margin before 300s limit

    # Parse clauses to 0-based (var_index, is_positive) tuples
    parsed_clauses = []
    for c in clauses:
        parsed = []
        for l in c:
            vi = abs(l) - 1
            pos = (l > 0)
            parsed.append((vi, pos))
        parsed_clauses.append(parsed)
    m = len(parsed_clauses)

    # Build variable-to-clause index: var_to_clauses[v] = list of (clause_idx, is_positive)
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(parsed_clauses):
        for (vi, pos) in clause:
            var_to_clauses[vi].append((ci, pos))

    rng = random.Random(12345)

    # Initialize assignment
    assignment = [rng.random() < 0.5 for _ in range(n)]

    # Initialize clause satisfaction counts
    clause_sat = [0] * m
    for ci, clause in enumerate(parsed_clauses):
        for (vi, pos) in clause:
            if assignment[vi] == pos:
                clause_sat[ci] += 1

    # Track unsatisfied clauses incrementally
    unsat_clauses = set()
    for ci in range(m):
        if clause_sat[ci] == 0:
            unsat_clauses.add(ci)

    def is_satisfied():
        return len(unsat_clauses) == 0

    def flip_var(v):
        """Flip variable v and update clause_sat and unsat_clauses.
        Returns (newly_satisfied_count, newly_unsatisfied_count)."""
        old_val = assignment[v]
        new_val = not old_val
        assignment[v] = new_val

        newly_sat = 0
        newly_unsat = 0
        for (ci, pos) in var_to_clauses[v]:
            if new_val == pos:
                # This literal is now satisfied
                if clause_sat[ci] == 0:
                    unsat_clauses.discard(ci)
                    newly_sat += 1
                clause_sat[ci] += 1
            else:
                # This literal is now not satisfied
                if clause_sat[ci] == 1:
                    unsat_clauses.add(ci)
                    newly_unsat += 1
                clause_sat[ci] -= 1
        return newly_sat, newly_unsat

    def compute_gain(v):
        """Compute the net gain (satisfied - unsatisfied) if we flip variable v."""
        gain = 0
        for (ci, pos) in var_to_clauses[v]:
            if assignment[v] == pos:
                # Currently satisfied; flipping makes it unsatisfied
                if clause_sat[ci] == 1:
                    gain -= 1
            else:
                # Currently unsatisfied; flipping makes it satisfied
                if clause_sat[ci] == 0:
                    gain += 1
        return gain

    # ---- Smooth soft-min pressure surrogate ----
    # For an unsatisfied clause, each literal has a "distance" to being
    # satisfied: 0 if the literal would be satisfied by flipping that var
    # to the needed value, else a penalty. The soft-min (log-sum-exp of
    # negative distances) is a smooth surrogate for the hard-min, giving
    # a scalar "closeness-to-satisfiable" score per clause.
    def clause_pressure(ci, beta=3.0):
        """Smooth pressure: lower = clause is easier to satisfy.
        Uses log-sum-exp of -distance over literals in the clause."""
        clause = parsed_clauses[ci]
        # distance for literal (vi, pos): 0 if flipping vi to pos satisfies
        # this clause (i.e., it's currently unsatisfied, which it is since
        # ci in unsat_clauses), else large.
        # For an unsatisfied clause, ALL literals are currently false.
        # Flipping vi to pos makes that literal true -> clause satisfied.
        # So every literal has distance 0 in the "flip-to-satisfy" sense.
        # To differentiate, use the *cost* of the flip: how many other
        # clauses would break. This is the negative of compute_gain.
        neg_gains = []
        for (vi, pos) in clause:
            g = compute_gain(vi)
            # We want a "distance" that is small when the flip is cheap.
            # distance = max(0, -g) + small base
            d = max(0.0, -float(g)) + 0.1
            neg_gains.append(-d)
        # log-sum-exp for numerical stability
        mx = max(neg_gains)
        if mx == float('-inf'):
            return 0.0
        s = sum(math.exp(x - mx) for x in neg_gains)
        # soft-min = -logsumexp(-d) ; we return the pressure (lower=easier)
        pressure = -(mx + math.log(s))
        # scale by beta for sharper/softer
        return pressure * beta

    def select_clause(unsat_list, clause_weight, rng, beta=3.0):
        """Select an unsatisfied clause using a mix of weight and smooth pressure.
        Clauses with LOWER pressure (easier to fix) get higher selection prob,
        tempered by clause_weight for hard-clause emphasis."""
        if not unsat_list:
            return None
        if len(unsat_list) == 1:
            return unsat_list[0]
        # Compute combined score: weight / (pressure + eps)
        # Lower pressure -> higher score.
        scores = []
        for ci in unsat_list:
            p = clause_pressure(ci, beta)
            w = clause_weight[ci]
            # score high when weight high AND pressure low
            sc = w / (p + 1.0)
            scores.append(sc)
        total = sum(scores)
        if total <= 0:
            return rng.choice(unsat_list)
        r = rng.random() * total
        cum = 0.0
        for ci, sc in zip(unsat_list, scores):
            cum += sc
            if r <= cum:
                return ci
        return unsat_list[-1]

    # WalkSAT with adaptive parameters
    best_assignment = assignment[:]
    best_unsat_count = len(unsat_clauses)

    if best_unsat_count == 0:
        return assignment[:]

    # Parameters
    max_steps_per_restart = 20000
    max_restarts = 500

    # Clause weights for weighted selection
    clause_weight = [1.0] * m

    # Adaptive P: start high, decrease as we get closer
    P_start = 0.6
    P_min = 0.15

    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break

        # Reinitialize assignment with phase saving
        if restart == 0:
            pass  # Already initialized
        else:
            assignment = best_assignment[:]
            # Randomly flip some variables to escape local minima
            for v in range(n):
                if rng.random() < 0.25:
                    assignment[v] = not assignment[v]

            # Rebuild clause_sat and unsat_clauses
            clause_sat = [0] * m
            unsat_clauses = set()
            for ci, clause in enumerate(parsed_clauses):
                sat_count = 0
                for (vi, pos) in clause:
                    if assignment[vi] == pos:
                        sat_count += 1
                clause_sat[ci] = sat_count
                if sat_count == 0:
                    unsat_clauses.add(ci)

        current_unsat = len(unsat_clauses)
        if current_unsat == 0:
            return assignment[:]

        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]

        # Adaptive P based on unsat fraction
        unsat_frac = current_unsat / m if m > 0 else 0
        P = max(P_min, P_start - 0.4 * (1 - unsat_frac))

        # Adaptive beta for pressure: sharper when close to solution
        beta = 2.0 + 4.0 * (1 - unsat_frac)

        # WalkSAT loop
        stuck_counter = 0
        best_step_unsat = current_unsat
        best_step_assignment = assignment[:]

        for step in range(max_steps_per_restart):
            if time.time() > deadline - 5.0:
                break

            if not unsat_clauses:
                return assignment[:]

            current_unsat = len(unsat_clauses)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]

            if current_unsat < best_step_unsat:
                best_step_unsat = current_unsat
                best_step_assignment = assignment[:]
                stuck_counter = 0
            else:
                stuck_counter += 1

            # Choose an unsatisfied clause (weighted + smooth pressure)
            unsat_list = list(unsat_clauses)
            ci = select_clause(unsat_list, clause_weight, rng, beta)
            if ci is None:
                break

            clause = parsed_clauses[ci]

            # Choose a literal to flip
            if rng.random() < P:
                # Random literal from the clause
                (vi, pos) = rng.choice(clause)
            else:
                # Greedy: choose the literal that maximizes gain
                best_gain = -10
                best_vi = clause[0][0]
                for (vi, pos) in clause:
                    gain = compute_gain(vi)
                    if gain > best_gain:
                        best_gain = gain
                        best_vi = vi
                vi = best_vi

            # Flip the variable
            newly_sat, newly_unsat = flip_var(vi)

            # Update clause weights
            if newly_sat > newly_unsat:
                clause_weight[ci] *= 0.9
            else:
                clause_weight[ci] *= 1.3

            # Periodic weight decay
            if step % 500 == 0:
                for i in range(m):
                    clause_weight[i] *= 0.95
                    if clause_weight[i] < 0.1:
                        clause_weight[i] = 1.0

            # Early exit if satisfied
            if not unsat_clauses:
                return assignment[:]

            # If stuck for too long, break to restart
            if stuck_counter > 3000:
                break

        # After restart, if we made progress, keep the best step assignment
        if best_step_unsat < current_unsat:
            assignment = best_step_assignment[:]
            # Rebuild clause_sat and unsat_clauses
            clause_sat = [0] * m
            unsat_clauses = set()
            for ci, clause in enumerate(parsed_clauses):
                sat_count = 0
                for (vi, pos) in clause:
                    if assignment[vi] == pos:
                        sat_count += 1
                clause_sat[ci] = sat_count
                if sat_count == 0:
                    unsat_clauses.add(ci)

    # Final polish: smooth-pressure-guided high-precision pass.
    # Uses a softened objective (pressure) to break local minima that
    # pure-greedy polish cannot escape. Lower P (more greedy) but with
    # pressure-based clause selection.
    if time.time() < deadline - 10.0 and best_unsat_count > 0:
        assignment = best_assignment[:]
        clause_sat = [0] * m
        unsat_clauses = set()
        for ci, clause in enumerate(parsed_clauses):
            sat_count = 0
            for (vi, pos) in clause:
                if assignment[vi] == pos:
                    sat_count += 1
            clause_sat[ci] = sat_count
            if sat_count == 0:
                unsat_clauses.add(ci)

        # Simulated-annealing-like polish: allow occasional "bad" moves
        # guided by the smooth pressure, with a cooling schedule.
        P_polish = 0.05
        T_start = 2.0
        T_end = 0.05
        total_polish_steps = 50000
        for step in range(total_polish_steps):
            if time.time() > deadline - 2.0:
                break
            if not unsat_clauses:
                return assignment[:]

            current_unsat = len(unsat_clauses)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]

            # Cooling schedule
            frac = step / max(1, total_polish_steps - 1)
            T = T_start * (T_end / T_start) ** frac
            beta = 2.0 + 4.0 * (1 - current_unsat / m if m > 0 else 0)

            unsat_list = list(unsat_clauses)
            ci = select_clause(unsat_list, clause_weight, rng, beta)
            if ci is None:
                break
            clause = parsed_clauses[ci]

            if rng.random() < P_polish:
                (vi, pos) = rng.choice(clause)
            else:
                best_gain = -10
                best_vi = clause[0][0]
                for (vi, pos) in clause:
                    gain = compute_gain(vi)
                    if gain > best_gain:
                        best_gain = gain
                        best_vi = vi
                vi = best_vi

            # Try the flip; accept if it improves, or with prob exp(-delta/T)
            newly_sat, newly_unsat = flip_var(vi)
            delta = newly_unsat - newly_sat  # >0 means worse
            if delta > 0:
                # Possibly reject (annealing)
                accept = False
                if T > 1e-6:
                    try:
                        accept = rng.random() < math.exp(-delta / T)
                    except OverflowError:
                        accept = False
                if not accept:
                    # Undo the flip
                    flip_var(vi)

            if not unsat_clauses:
                return assignment[:]

    return best_assignment
# EVOLVE-BLOCK-END
