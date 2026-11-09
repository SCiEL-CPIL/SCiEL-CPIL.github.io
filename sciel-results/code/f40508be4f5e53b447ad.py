import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    time_limit = 295  # Leave a small buffer to avoid timeout

    def walk_sat(restart_seed, max_flips=100000):
        rng = random.Random(restart_seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        def satisfied(clause):
            return any(assignment[abs(l) - 1] == (l > 0) for l in clause)
        
        def count_unsat(clause):
            return sum(1 for l in clause if assignment[abs(l) - 1] != (l > 0))
        
        # Precompute clause lengths for efficiency
        for _ in range(max_flips):
            unsat_clauses = [c for c in clauses if not satisfied(c)]
            if not unsat_clauses:
                return assignment
            # Choose an unsatisfied clause
            clause = rng.choice(unsat_clauses)
            # Choose a variable to flip with WalkSAT's heuristic
            p = 0.5  # Probability of choosing a variable that appears in many unsat clauses
            if rng.random() < p:
                # Choose a variable that appears in the most unsat clauses
                var_counts = {}
                for c in unsat_clauses:
                    for l in c:
                        var = abs(l)
                        var_counts[var] = var_counts.get(var, 0) + 1
                # Select from the clause, but weight by frequency
                weights = [var_counts.get(abs(l), 1) for l in clause]
                total = sum(weights)
                r = rng.random() * total
                cumulative = 0
                chosen_literal = clause[-1]
                for l, w in zip(clause, weights):
                    cumulative += w
                    if r <= cumulative:
                        chosen_literal = l
                        break
            else:
                chosen_literal = rng.choice(clause)
            
            var = abs(chosen_literal) - 1
            assignment[var] = not assignment[var]
        
        return None

    # Luby restart sequence
    def luby_sequence():
        k = 0
        while True:
            n = 1
            i = 0
            while i < (1 << k):
                yield n
                i += 1
            k += 1

    restarts = list(luby_sequence())
    max_restarts = len(restarts)
    
    for i in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        restart_seed = i
        result = walk_sat(restart_seed)
        if result is not None:
            return result
    
    # If all restarts fail, return the last assignment
    return assignment
