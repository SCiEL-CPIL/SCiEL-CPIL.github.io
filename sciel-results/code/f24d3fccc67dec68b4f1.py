"""Synthetic planted 3-SAT solver (50 variables) - improved local search."""
import time
import random


def solve(instance):
    n = instance["n"]
    clauses_orig = instance["clauses"]
    deadline = time.time() + 320.0

    # Preprocess: unit propagation and pure literal elimination
    fixed = {}
    changed = True
    clauses = list(clauses_orig)
    while changed:
        changed = False
        new_clauses = []
        for clause in clauses:
            unassigned = [l for l in clause if abs(l) - 1 not in fixed]
            if not unassigned:
                if not any(fixed[abs(l) - 1] == (l > 0) for l in clause):
                    return [fixed.get(i, False) for i in range(n)]
                continue
            if len(unassigned) == 1:
                l = unassigned[0]
                fixed[abs(l) - 1] = (l > 0)
                changed = True
                continue
            new_clauses.append(clause)
        clauses = new_clauses
        if clauses:
            pos = set()
            neg = set()
            for clause in clauses:
                for l in clause:
                    var = abs(l) - 1
                    if var in fixed:
                        continue
                    if l > 0:
                        pos.add(var)
                    else:
                        neg.add(var)
            for var in list(pos - neg):
                fixed[var] = True
                changed = True
            for var in list(neg - pos):
                fixed[var] = False
                changed = True
            clauses = []
            for clause in clauses_orig:
                unassigned = [l for l in clause if abs(l) - 1 not in fixed]
                if not unassigned:
                    if not any(fixed[abs(l) - 1] == (l > 0) for l in clause):
                        return [fixed.get(i, False) for i in range(n)]
                    continue
                clauses.append(clause)

    m = len(clauses)
    if m == 0:
        return [fixed.get(i, False) for i in range(n)]

    rng = random.Random(42)

    def make_assignment(seed):
        r = random.Random(seed)
        return [fixed.get(i, r.random() < 0.5) for i in range(n)]

    def count_unsat(assignment):
        cnt = 0
        for clause in clauses:
            sat = False
            for l in clause:
                var = abs(l) - 1
                if assignment[var] == (l > 0):
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt

    best_assignment = make_assignment(0)
    best_unsat = count_unsat(best_assignment)
    if best_unsat == 0:
        return best_assignment

    step_budget = 50000
    max_restarts = 1000

    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        if restart == 0:
            assignment = best_assignment[:]
        else:
            assignment = make_assignment(rng.randint(0, 10**9))
        unsat = count_unsat(assignment)
        if unsat == 0:
            return assignment
        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = assignment[:]

        for step in range(step_budget):
            if time.time() > deadline:
                break
            # Find unsat clauses
            unsat_indices = []
            for ci, clause in enumerate(clauses):
                sat = False
                for l in clause:
                    var = abs(l) - 1
                    if assignment[var] == (l > 0):
                        sat = True
                        break
                if not sat:
                    unsat_indices.append(ci)
            if not unsat_indices:
                return assignment
            # Pick clause with fewest false literals (all are 3 for 3-SAT)
            # Just pick random unsat clause
            ci = rng.choice(unsat_indices)
            clause = clauses[ci]
            # Pick literal: try each, pick one that minimizes unsat count
            # For speed, pick random with some bias
            l = rng.choice(clause)
            var = abs(l) - 1
            assignment[var] = not assignment[var]
            unsat = count_unsat(assignment)
            if unsat == 0:
                return assignment
            if unsat < best_unsat:
                best_unsat = unsat
                best_assignment = assignment[:]

    return best_assignment
