import time
import random
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Pre-process clauses
    # Convert 1-based signed literals to 0-based (var, sign)
    proc_clauses = []
    var_to_clauses = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses):
        proc_clause = []
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            proc_clause.append((v, s))
            var_to_clauses[v].append((i, s))
        proc_clauses.append(proc_clause)
        
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    best_energy = float('inf')
    best_assignment = [0] * n
    
    rng = random.Random(0)
    
    while time.time() < deadline:
        # Initialize random assignment
        assignment = [rng.randint(0, 1) for _ in range(n)]
        
        # Calculate initial weights and energy
        weights = [0] * m
        energy = 0
        for i, clause in enumerate(proc_clauses):
            w = 0
            for v, s in clause:
                if (s == 1 and assignment[v] == 0) or (s == -1 and assignment[v] == 1):
                    w += 1
            weights[i] = w
            energy += w
            
        if energy == 0:
            return [bool(x) for x in assignment]
            
        if energy < best_energy:
            best_energy = energy
            best_assignment = assignment[:]
            
        T = 1.0
        steps_since_improve = 0
        best_in_run = energy
        
        # Main search loop
        while time.time() < deadline:
            if energy == 0:
                return [bool(x) for x in assignment]
                
            if energy < best_in_run:
                best_in_run = energy
                steps_since_improve = 0
                if energy < best_energy:
                    best_energy = energy
                    best_assignment = assignment[:]
            else:
                steps_since_improve += 1
                
            if steps_since_improve > 3000:
                break # Restart
                
            # Pick an unsatisfied clause
            # Build list of unsat indices and their weights
            unsat_data = []
            total_w = 0
            for i in range(m):
                if weights[i] > 0:
                    unsat_data.append((i, weights[i]))
                    total_w += weights[i]
                    
            if not unsat_data:
                break # Should be caught by energy==0
                
            # Weighted random selection
            r = rng.uniform(0, total_w)
            acc = 0
            selected_ci = unsat_data[0][0]
            for i, w in unsat_data:
                acc += w
                if r <= acc:
                    selected_ci = i
                    break
            
            clause = proc_clauses[selected_ci]
            
            # Pick a literal in the clause
            v, s = rng.choice(clause)
            
            # Calculate delta E for flipping v
            delta_e = 0
            for ci2, s2 in var_to_clauses[v]:
                if (s2 == 1 and assignment[v] == 0) or (s2 == -1 and assignment[v] == 1):
                    # Currently False
                    delta_e -= 1
                else:
                    # Currently True
                    delta_e += 1
            
            # Acceptance
            accept = False
            if delta_e < 0:
                accept = True
            else:
                if T > 0:
                    prob = math.exp(-delta_e / T)
                    if rng.random() < prob:
                        accept = True
            
            if accept:
                # Flip v
                assignment[v] = 1 - assignment[v]
                
                # Update weights
                for ci2, s2 in var_to_clauses[v]:
                    # Check new state
                    if (s2 == 1 and assignment[v] == 1) or (s2 == -1 and assignment[v] == 0):
                        # Now True, was False
                        weights[ci2] -= 1
                        energy -= 1
                    else:
                        # Now False, was True
                        weights[ci2] += 1
                        energy += 1
                        
                # Cool
                T *= 0.999
                if T < 0.01:
                    T = 0.01
                    
            else:
                # Reject flip
                T *= 0.999
                if T < 0.01:
                    T = 0.01

    # Return best found
    return [bool(x) for x in best_assignment]
