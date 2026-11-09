"""Planted 3-SAT solver: DPLL+unit-prop + annealed WalkSAT with deadline guard."""
# EVOLVE-BLOCK-START
import random
import time


def _dpll_unit(instance):
    """Try DPLL with unit propagation. Returns (solved, assignment) or (False, None)."""
    n = instance["n"]
    clauses = instance["clauses"]
    assignment = [None] * n  # None = unassigned

    # Build clause list as list of literals (signed 1-based)
    # Use watched-literal style unit propagation
    # Simple approach: iterate until fixpoint
    changed = True
    while changed:
        changed = False
        # Unit propagation: find clauses with exactly one unassigned literal
        for i in range(len(clauses)):
            c = clauses[i]
            # Check if clause is already satisfied
            sat = False
            unassigned = []
            for l in c:
                v = assignment[abs(l) - 1]
                if v is not None:
                    if v == (l > 0):
                        sat = True
                        break
                else:
                    unassigned.append(l)
            if sat:
                continue
            if not unassigned:
                return (False, None)  # conflict
            if len(unassigned) == 1:
                l = unassigned[0]
                assignment[abs(l) - 1] = (l > 0)
                changed = True

    # Check all satisfied
    for c in clauses:
        sat = False
        for l in c:
            v = assignment[abs(l) - 1]
            if v is not None and v == (l > 0):
                sat = True
                break
        if not sat:
            return (False, None)

    # Fill remaining unassigned
    for i in range(n):
        if assignment[i] is None:
            assignment[i] = False
    return (True, assignment)


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    deadline = time.time() + 290  # safety margin before 300s

    # Step 1: Try DPLL with unit propagation
    solved, assignment = _dpll_unit(instance)
    if solved:
        return assignment

    # Step 2: Annealed WalkSAT with weighted variable selection
    rng = random.Random(42)

    # Build index: variable -> list of (clause_idx, literal)
    var_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clauses):
        for l in c:
            var_clauses[abs(l) - 1].append((ci, l))

    def eval_clause(c, assign):
        for l in c:
            if assign[abs(l) - 1] == (l > 0):
                return True
        return False

    def count_unsat(assign):
        cnt = 0
        for c in clauses:
            if not eval_clause(c, assign):
                cnt += 1
        return cnt

    best_assign = None
    best_unsat = float('inf')

    # Initial random assignment
    assign = [rng.random() < 0.5 for _ in range(n)]
    cur_unsat = count_unsat(assign)
    best_assign = assign[:]
    best_unsat = cur_unsat
    if cur_unsat == 0:
        return assign

    # Simulated annealing parameters
    T_start = 1.0
    T_end = 0.05
    alpha = 0.999  # cooling rate
    T = T_start

    # Weighted flip: for each variable, compute delta = (# clauses that would become satisfied) - (# that would become unsatisfied)
    # We pick among variables in unsat clauses, weighting by how many unsat clauses they fix

    max_iter = 500000
    it = 0
    no_improve = 0
    stagnation_limit = 5000

    while it < max_iter and time.time() < deadline:
        # Find unsat clauses
        unsat = [c for c in clauses if not eval_clause(c, assign)]
        cur_unsat = len(unsat)
        if cur_unsat == 0:
            return assign
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assign = assign[:]
            no_improve = 0
        else:
            no_improve += 1

        # Pick a random unsat clause
        clause = unsat[rng.randrange(len(unsat))]

        # Compute weighted delta for each literal in the clause
        # delta for flipping variable v: gain - loss
        candidates = []
        for l in clause:
            v = abs(l) - 1
            # Count how many unsat clauses this variable appears in and would fix
            gain = 0
            loss = 0
            for (ci, ll) in var_clauses[v]:
                c2 = clauses[ci]
                # Is this clause currently unsat?
                # We need to check: if we flip v, does c2 become sat or unsat?
                # Current state of c2:
                cur_sat = False
                for l2 in c2:
                    if assign[abs(l2) - 1] == (l2 > 0):
                        cur_sat = True
                        break
                if cur_sat:
                    # Flipping v might make it unsat (if v was the only satisfying literal)
                    # Check if any other literal satisfies c2
                    other_sat = False
                    for l2 in c2:
                        if l2 != ll and assign[abs(l2) - 1] == (l2 > 0):
                            other_sat = True
                            break
                    if not other_sat:
                        loss += 1
                else:
                    # Clause is unsat; flipping v to satisfy ll would satisfy c2
                    # Only if ll is the literal we're flipping to
                    if (l > 0 and not assign[v]) or (l < 0 and assign[v]):
                        # Flipping v would make ll true
                        gain += 1
            delta = gain - loss
            candidates.append((delta, v))

        # Weighted selection: prefer higher delta
        # Use softmax-like weighting
        max_d = max(d for d, _ in candidates)
        weights = []
        for d, _ in candidates:
            weights.append((d - max_d) / max(T, 0.01))
        # Convert to probabilities
        import math
        exps = [math.exp(w) for w in weights]
        total = sum(exps)
        probs = [e / total for e in exps]
        r = rng.random()
        cum = 0
        chosen_idx = 0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                chosen_idx = i
                break
        v = candidates[chosen_idx][1]
        assign[v] = not assign[v]

        it += 1
        # Cooling
        if T > T_end:
            T *= alpha

        # Restart if stagnating
        if no_improve > stagnation_limit:
            # Partial restart: keep best, randomize some
            assign = best_assign[:]
            for i in range(n):
                if rng.random() < 0.3:
                    assign[i] = not assign[i]
            cur_unsat = count_unsat(assign)
            no_improve = 0
            T = T_start * 0.5

    return best_assign
# EVOLVE-BLOCK-END
