"""Stronger planted 3-SAT solver: MinCon + phase-aware WalkSAT with restarts and a wall-clock deadline guard."""
# EVOLVE-BLOCK-START
import time
import random


def _clause_sat_fast(clause_vars, clause_polarity, assign):
    """Check if clause is satisfied using precomputed vars and polarity."""
    for v, p in zip(clause_vars, clause_polarity):
        if assign[v] == p:
            return True
    return False


def _count_unsat_fast(clause_vars_list, clause_polarity_list, assign):
    count = 0
    for cv, cp in zip(clause_vars_list, clause_polarity_list):
        sat = False
        for v, p in zip(cv, cp):
            if assign[v] == p:
                sat = True
                break
        if not sat:
            count += 1
    return count


def _phase_seed(n, clauses, rng):
    """Estimate a planted assignment from literal phase bias, with random tie-breaking."""
    pos = [0] * n
    neg = [0] * n
    for c in clauses:
        for l in c:
            if l > 0:
                pos[l - 1] += 1
            else:
                neg[-l - 1] += 1
    assign = []
    for i in range(n):
        p, q = pos[i], neg[i]
        if p > q:
            assign.append(True)
        elif q > p:
            assign.append(False)
        else:
            assign.append(rng.random() < 0.5)
    return assign


def _mincon_step_fast(clause_vars_list, clause_polarity_list, assign, rng):
    """One MinCon step with precomputed clause structure."""
    n = len(assign)
    unsat_vars = [0] * n
    unsat_clauses = []
    for idx, (cv, cp) in enumerate(zip(clause_vars_list, clause_polarity_list)):
        sat = False
        for v, p in zip(cv, cp):
            if assign[v] == p:
                sat = True
                break
        if not sat:
            unsat_clauses.append(idx)
            for v in cv:
                unsat_vars[v] += 1
    if not unsat_clauses:
        return assign, True
    # Choose the unsat clause with the highest sum of unsat_vars
    best_clause_idx = None
    best_score = -1
    for ci in unsat_clauses:
        cv = clause_vars_list[ci]
        s = sum(unsat_vars[v] for v in cv)
        if s > best_score:
            best_score = s
            best_clause_idx = ci
    # Flip the variable in best clause with highest unsat count
    cv = clause_vars_list[best_clause_idx]
    best_v = max(cv, key=lambda v: unsat_vars[v])
    assign[best_v] = not assign[best_v]
    return assign, False


def _run_one_fast(clause_vars_list, clause_polarity_list, assign, rng, deadline, strategy):
    """Run a single local search to completion or deadline. Returns (assign, solved)."""
    n = len(assign)
    num_clauses = len(clause_vars_list)
    if strategy == "mincon":
        for _ in range(200000):
            if time.time() >= deadline:
                return assign, _count_unsat_fast(clause_vars_list, clause_polarity_list, assign) == 0
            assign, done = _mincon_step_fast(clause_vars_list, clause_polarity_list, assign, rng)
            if done:
                return assign, True
    else:
        # Phase-aware WalkSAT with precomputed structure
        for _ in range(200000):
            if time.time() >= deadline:
                return assign, _count_unsat_fast(clause_vars_list, clause_polarity_list, assign) == 0
            # Find unsat clauses
            unsat_indices = []
            for idx in range(num_clauses):
                cv = clause_vars_list[idx]
                cp = clause_polarity_list[idx]
                sat = False
                for v, p in zip(cv, cp):
                    if assign[v] == p:
                        sat = True
                        break
                if not sat:
                    unsat_indices.append(idx)
            if not unsat_indices:
                return assign, True
            ci = rng.choice(unsat_indices)
            cv = clause_vars_list[ci]
            cp = clause_polarity_list[ci]
            # Score each variable: prefer flipping a var that is in many unsat clauses
            best_v = None
            best_s = -1
            for v in cv:
                s = rng.random() * 0.1
                # Count how many unsat clauses contain this variable
                count = 0
                for ui in unsat_indices:
                    if v in clause_vars_list[ui]:
                        count += 1
                s += count * 0.01
                if s > best_s:
                    best_s = s
                    best_v = v
            assign[best_v] = not assign[best_v]
    return assign, _count_unsat_fast(clause_vars_list, clause_polarity_list, assign) == 0


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start = time.time()
    # Safety margin: stop 8s before the 330s kill
    deadline = start + 322.0

    # Precompute clause structure: list of (var_index, polarity) per clause
    clause_vars_list = []
    clause_polarity_list = []
    for c in clauses:
        cv = []
        cp = []
        for l in c:
            v = abs(l) - 1
            cv.append(v)
            cp.append(l > 0)
        clause_vars_list.append(cv)
        clause_polarity_list.append(cp)

    # Precompute a phase-bias seed (deterministic base) and use it
    base_rng = random.Random(0)
    phase_assign = _phase_seed(n, clauses, base_rng)

    best_assign = phase_assign[:]
    best_unsat = _count_unsat_fast(clause_vars_list, clause_polarity_list, best_assign)
    if best_unsat == 0:
        return best_assign

    # Portfolio: alternate strategies and seeds, growing effort, until deadline
    seed_counter = 0
    while time.time() < deadline:
        rng = random.Random(seed_counter)
        # Choose a starting assignment
        r = rng.random()
        if r < 0.5:
            assign = phase_assign[:]
        elif r < 0.75:
            assign = [rng.random() < 0.5 for _ in range(n)]
        else:
            # Flip a few random vars from phase seed
            assign = phase_assign[:]
            for _ in range(rng.randint(1, 5)):
                assign[rng.randrange(n)] = not assign[rng.randrange(n)]

        strategy = rng.choice(["mincon", "walk"])
        assign, solved = _run_one_fast(clause_vars_list, clause_polarity_list, assign, rng, deadline, strategy)
        if solved:
            return assign
        u = _count_unsat_fast(clause_vars_list, clause_polarity_list, assign)
        if u < best_unsat:
            best_unsat = u
            best_assign = assign[:]
        seed_counter += 1

    return best_assign
# EVOLVE-BLOCK-END
