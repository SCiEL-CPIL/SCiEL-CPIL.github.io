# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Precompute clause structure
    # clauses_vars[ci] = list of 0-based variable indices
    # clauses_signs[ci] = list of +1/-1 signs
    clauses_vars = []
    clauses_signs = []
    for c in clauses:
        vs = []
        ss = []
        for l in c:
            vs.append(abs(l) - 1)
            ss.append(1 if l > 0 else -1)
        clauses_vars.append(vs)
        clauses_signs.append(ss)

    # var_to_clauses[v] = list of clause indices containing v
    var_to_clauses = [[] for _ in range(n)]
    for ci, vs in enumerate(clauses_vars):
        for v in vs:
            var_to_clauses[v].append(ci)

    # signs_in[v][ci] = sign of variable v in clause ci (0 if not present)
    # Using a list of lists for fast access
    signs_in = [[0] * m for _ in range(n)]
    for ci, vs in enumerate(clauses_vars):
        ss = clauses_signs[ci]
        for j, v in enumerate(vs):
            signs_in[v][ci] = ss[j]

    deadline = time.time() + 290.0
    rng = random.Random(0)

    # Initialize assignment with random values (1 or -1)
    assignment = [1 if rng.random() < 0.5 else -1 for _ in range(n)]
    
    # Initialize sat_count
    sat_count = [0] * m
    for ci in range(m):
        vs = clauses_vars[ci]
        ss = clauses_signs[ci]
        cnt = 0
        for j in range(len(vs)):
            if assignment[vs[j]] == ss[j]:
                cnt += 1
        sat_count[ci] = cnt

    # Check if solved
    if all(c > 0 for c in sat_count):
        return [a > 0 for a in assignment]

    def is_solved():
        for c in sat_count:
            if c == 0:
                return False
        return True

    max_steps_per_run = 10000
    steps_in_run = 0
    best_sat_count = sum(1 for c in sat_count if c > 0)
    best_assignment = assignment[:]

    while time.time() < deadline:
        if is_solved():
            return [a > 0 for a in assignment]

        # Find unsat clauses
        unsat_indices = [ci for ci in range(m) if sat_count[ci] == 0]
        if not unsat_indices:
            return [a > 0 for a in assignment]

        ci = rng.choice(unsat_indices)
        vs = clauses_vars[ci]
        
        best_var = -1
        best_delta = float('inf')
        
        # Evaluate delta for each variable in the unsat clause
        for v in vs:
            delta = 0
            old_val = assignment[v]
            # new_val will be -old_val
            
            for cj in var_to_clauses[v]:
                sgn = signs_in[v][cj]
                # Check if literal is currently satisfied
                is_sat_lit = (old_val == sgn)
                k = sat_count[cj]
                
                if is_sat_lit:
                    # Flipping makes it unsatisfied
                    # Clause becomes unsat if no other literal is satisfied
                    if k - 1 == 0:
                        delta += 1
                else:
                    # Flipping makes it satisfied
                    # Clause becomes sat if it was unsat (k==0)
                    if k == 0:
                        delta -= 1
            
            if delta < best_delta:
                best_delta = delta
                best_var = v

        # Randomization: 30% chance to pick a random variable from the clause
        if rng.random() < 0.3:
            v = rng.choice(vs)
        else:
            v = best_var

        if v == -1:
            # Should not happen
            break

        # Apply flip
        old_val = assignment[v]
        new_val = -old_val
        assignment[v] = new_val
        
        # Update sat_count
        for cj in var_to_clauses[v]:
            sgn = signs_in[v][cj]
            old_sat = (old_val == sgn)
            new_sat = (new_val == sgn)
            
            if old_sat and not new_sat:
                sat_count[cj] -= 1
            elif not old_sat and new_sat:
                sat_count[cj] += 1

        steps_in_run += 1
        
        # Track best state
        current_sat_count = sum(1 for c in sat_count if c > 0)
        if current_sat_count > best_sat_count:
            best_sat_count = current_sat_count
            best_assignment = assignment[:]

        # Restart logic
        if steps_in_run > max_steps_per_run:
            # Random restart
            assignment = [1 if rng.random() < 0.5 else -1 for _ in range(n)]
            sat_count = [0] * m
            for ci in range(m):
                vs = clauses_vars[ci]
                ss = clauses_signs[ci]
                cnt = 0
                for j in range(len(vs)):
                    if assignment[vs[j]] == ss[j]:
                        cnt += 1
                sat_count[ci] = cnt
            
            if all(c > 0 for c in sat_count):
                return [a > 0 for a in assignment]
                
            steps_in_run = 0

    # Return best found
    return [a > 0 for a in best_assignment]
# EVOLVE-BLOCK-END
