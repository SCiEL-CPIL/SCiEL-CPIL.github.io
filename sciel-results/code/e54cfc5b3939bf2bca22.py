import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Preprocessing
    # clauses are lists of signed 1-based literals
    # Convert to (var_idx, polarity)
    # var_idx 0-based, polarity True/False
    
    # Data structures
    clause_data = [] # list of lists of (var_idx, polarity)
    var_clauses = [[] for _ in range(n)] # list of lists of (clause_idx, polarity)
    
    for i, c in enumerate(clauses):
        lit_data = []
        for l in c:
            v = abs(l) - 1
            p = l > 0
            lit_data.append((v, p))
            var_clauses[v].append((i, p))
        clause_data.append(lit_data)
        
    m = len(clauses)
    
    best_unsat_count = float('inf')
    best_assignment = None
    
    # Adaptive restart parameters
    # Geometric distribution for restart length
    # p_restart = 0.01 implies mean restart length 100
    p_restart = 0.01
    
    # Weighted variable selection parameters
    # Temperature for softmax-like weighting? Or just use min-conflicts count as weight?
    # Let's use: weight = 1 / (conflicts + 1)
    # This favors variables with fewer conflicts.
    
    while time.time() < deadline:
        # Generate restart length
        # Geometric distribution: P(k) = (1-p)^(k-1) * p
        # We want to decide how many steps to take in this restart
        # Or simply check for restart at each step?
        # Efficient: decide max_steps for this restart.
        # But we can also just run until restart condition.
        # Let's use a fixed max_steps cap to avoid infinite loops if p is very small, 
        # but usually geometric is fine.
        
        # Initialize random assignment
        rng = random.Random()
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Calc initial sat counts
        sat_count = [0] * m
        unsat_set = set()
        
        for c_idx in range(m):
            cnt = 0
            for v, p in clause_data[c_idx]:
                if assignment[v] == p:
                    cnt += 1
            sat_count[c_idx] = cnt
            if cnt == 0:
                unsat_set.add(c_idx)
                
        unsat_count = len(unsat_set)
        
        if unsat_count == 0:
            return assignment
            
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
            
        # Walk loop
        # Determine restart length
        # Using geometric distribution:
        # r = rng.expovariate(p_restart) gives mean 1/p_restart
        # steps = int(r)
        # However, expovariate might be slow? No, it's fast.
        # But we need integer steps.
        # Alternatively, loop and break with prob p_restart.
        # This allows early exit if solved.
        
        while time.time() < deadline:
            if unsat_count == 0:
                return assignment
            
            # Check for restart
            if rng.random() < p_restart:
                break # Restart
            
            # Pick random unsat clause
            if not unsat_set:
                return assignment # Should be caught by unsat_count==0
            
            c_idx = next(iter(unsat_set)) # O(1) amortized? No, next(iter()) on set is O(1) in CPython for first element?
            # Actually, random.choice on a set requires list conversion O(N).
            # For M~600, list(unsat_set) is fast.
            # But we can just pick a random index from a list if we maintain one.
            # Given M is small, let's use list(unsat_set) for simplicity and true randomness.
            # Optimization: If unsat_count is small, list conversion is cheap.
            unsat_list = list(unsat_set)
            c_idx = unsat_list[rng.randrange(len(unsat_list))]

            # Evaluate flips for literals in c_idx
            # We want to pick a variable to flip.
            # Heuristic: Minimize number of newly unsatisfied clauses.
            # Let's calculate the "cost" (number of breaks) for each variable in the clause.
            # cost[v] = number of clauses containing v that are currently satisfied ONLY by v.
            
            costs = []
            for v, p in clause_data[c_idx]:
                cost = 0
                # Iterate over clauses containing v
                for c2_idx, p2 in var_clauses[v]:
                    if sat_count[c2_idx] == 1 and assignment[v] == p2:
                        # This clause is satisfied only by v. Flipping v will break it.
                        cost += 1
                costs.append(cost)
                
            # Select variable based on costs
            # Strategy: Weighted random selection where weight = 1 / (cost + 1)
            # This prefers low cost variables but allows exploration.
            # Alternatively, pick min cost with some noise.
            # Let's try: with prob 0.5 pick min cost, else pick weighted random.
            # Or simply weighted random always.
            
            total_weight = 0.0
            weights = []
            min_cost = float('inf')
            for c in costs:
                w = 1.0 / (c + 1.0)
                weights.append(w)
                total_weight += w
                if c < min_cost:
                    min_cost = c
                    
            # Hybrid selection:
            # 50% chance: pick a variable with min_cost
            # 50% chance: pick weighted random
            if rng.random() < 0.5:
                # Pick one of the min cost variables randomly
                min_vars = [i for i, c in enumerate(costs) if c == min_cost]
                v_idx = min_vars[rng.randrange(len(min_vars))]
            else:
                # Weighted random
                r = rng.random() * total_weight
                cumulative = 0.0
                v_idx = 0
                for i, w in enumerate(weights):
                    cumulative += w
                    if r <= cumulative:
                        v_idx = i
                        break
                else:
                    v_idx = len(weights) - 1
            
            v_to_flip = clause_data[c_idx][v_idx][0]
            
            # Apply flip
            old_val = assignment[v_to_flip]
            new_val = not old_val
            assignment[v_to_flip] = new_val
            
            # Update state
            for c2_idx, p2 in var_clauses[v_to_flip]:
                was_sat = (old_val == p2)
                
                if was_sat:
                    # Lost a satisfaction
                    sat_count[c2_idx] -= 1
                    if sat_count[c2_idx] == 0:
                        # Became unsat
                        unsat_set.add(c2_idx)
                        unsat_count += 1
                else:
                    # Gained a satisfaction
                    sat_count[c2_idx] += 1
                    if sat_count[c2_idx] == 1:
                        # Became sat
                        unsat_set.discard(c2_idx)
                        unsat_count -= 1
                            
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]

    # Fallback
    if best_assignment is not None:
        return best_assignment
    return assignment
