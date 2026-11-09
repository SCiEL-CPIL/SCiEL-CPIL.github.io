import time
import random
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    start_time = time.time()
    deadline = start_time + 295
    
    # Precomputation: convert to 0-based, store as (var, sign)
    var_clauses = [[] for _ in range(n)]
    clauses_store = []
    
    for i, clause in enumerate(clauses):
        cl_store = []
        for l in clause:
            v = abs(l) - 1
            sign = 1 if l > 0 else -1
            cl_store.append((v, sign))
            var_clauses[v].append((i, sign))
        clauses_store.append(cl_store)
    
    m = len(clauses)
    
    def is_satisfied(assignment, clause_idx):
        for v, sign in clauses_store[clause_idx]:
            if assignment[v] == (sign == 1):
                return True
        return False
    
    def count_unsat(assignment):
        cnt = 0
        for i in range(m):
            if not is_satisfied(assignment, i):
                cnt += 1
        return cnt
    
    def walk_sat_simulated_annealing(rng, max_steps=50000, initial_temp=10.0, cooling=0.9995, min_temp=0.01):
        """
        WalkSAT with simulated annealing to escape local minima.
        """
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Compute initial unsat count
        unsat_count = count_unsat(assignment)
        if unsat_count == 0:
            return assignment
        
        best_unsat = unsat_count
        best_assignment = assignment.copy()
        
        temp = initial_temp
        no_improve = 0
        max_no_improve = 3000
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if unsat_count == 0:
                return assignment
            
            # Pick a random unsatisfied clause
            unsat_clauses = [i for i in range(m) if not is_satisfied(assignment, i)]
            if not unsat_clauses:
                return assignment
            
            cl_idx = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
            
            # Choose a variable to flip from this clause
            # With some probability, pick the best variable; otherwise random
            if rng.random() < 0.7:
                # Best improvement heuristic
                best_var = -1
                best_delta = -1000000
                for v, sign in clauses_store[cl_idx]:
                    # Compute delta in unsat count if we flip v
                    delta = 0
                    # This clause is currently unsat, so flipping v will satisfy it if v currently doesn't satisfy
                    if assignment[v] != (sign == 1):
                        delta -= 1  # This clause becomes satisfied
                    else:
                        delta += 1  # This clause stays unsat (but we're flipping a satisfying literal)
                    
                    # Check other clauses containing v
                    for c_idx, c_sign in var_clauses[v]:
                        if c_idx == cl_idx:
                            continue
                        currently_satisfies = (assignment[v] == (c_sign == 1))
                        if currently_satisfies:
                            # Flipping will make this clause potentially unsat
                            # Check if other literals in this clause satisfy it
                            other_satisfy = False
                            for v2, s2 in clauses_store[c_idx]:
                                if v2 == v:
                                    continue
                                if assignment[v2] == (s2 == 1):
                                    other_satisfy = True
                                    break
                            if not other_satisfy:
                                delta += 1
                        else:
                            # Flipping will satisfy this clause
                            delta -= 1
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_var = v
                if best_var != -1:
                    v_to_flip = best_var
                else:
                    v_to_flip = clauses_store[cl_idx][rng.randint(0, 2)][0]
            else:
                # Random variable from the clause
                v_to_flip = clauses_store[cl_idx][rng.randint(0, 2)][0]
            
            # Compute the actual delta for this flip
            old_unsat = unsat_count
            assignment[v_to_flip] = not assignment[v_to_flip]
            new_unsat = count_unsat(assignment)
            delta = new_unsat - old_unsat
            
            # Simulated annealing acceptance
            if delta <= 0:
                # Accept improvement or equal
                unsat_count = new_unsat
                if new_unsat < best_unsat:
                    best_unsat = new_unsat
                    best_assignment = assignment.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Accept worse move with probability exp(-delta/temp)
                if temp > min_temp:
                    prob = math.exp(-delta / temp)
                    if rng.random() < prob:
                        unsat_count = new_unsat
                        no_improve += 1
                    else:
                        # Reject, revert
                        assignment[v_to_flip] = not assignment[v_to_flip]
                        no_improve += 1
                else:
                    # Reject, revert
                    assignment[v_to_flip] = not assignment[v_to_flip]
                    no_improve += 1
            
            # Cool down
            temp = max(temp * cooling, min_temp)
            
            if no_improve > max_no_improve:
                break
        
        if best_unsat == 0:
            return best_assignment
        return None
    
    def dpll_with_propagation(rng, depth_limit=20):
        """
        DPLL with unit propagation for deeper search.
        """
        def unit_propagate(assignment, clauses_to_check):
            """
            Perform unit propagation. Returns (new_assignment, remaining_clauses, contradiction).
            """
            new_assignment = assignment.copy()
            remaining = list(clauses_to_check)
            changed = True
            while changed:
                changed = False
                new_remaining = []
                for clause in remaining:
                    # Count satisfied and unassigned literals
                    sat = False
                    unassigned = []
                    for v, sign in clause:
                        if new_assignment[v] is not None:
                            if new_assignment[v] == (sign == 1):
                                sat = True
                                break
                        else:
                            unassigned.append((v, sign))
                    
                    if sat:
                        continue  # Clause is satisfied
                    
                    if len(unassigned) == 0:
                        return None, None, True  # Contradiction
                    
                    if len(unassigned) == 1:
                        # Unit clause - propagate
                        v, sign = unassigned[0]
                        new_assignment[v] = (sign == 1)
                        changed = True
                    else:
                        new_remaining.append(clause)
                
                remaining = new_remaining
            
            return new_assignment, remaining, False
        
        def dpll_recursive(assignment, clauses_remaining, depth):
            if depth > depth_limit:
                return None
            
            # Check if all clauses are satisfied
            all_sat = True
            for clause in clauses_remaining:
                sat = False
                for v, sign in clause:
                    if assignment[v] is not None:
                        if assignment[v] == (sign == 1):
                            sat = True
                            break
                if not sat:
                    all_sat = False
                    break
            
            if all_sat:
                return assignment
            
            # Unit propagation
            new_assignment, remaining_clauses, contradiction = unit_propagate(assignment, clauses_remaining)
            if contradiction:
                return None
            
            if remaining_clauses is None or len(remaining_clauses) == 0:
                # Check if all are satisfied
                all_sat = True
                for clause in clauses_remaining:
                    sat = False
                    for v, sign in clause:
                        if new_assignment[v] is not None:
                            if new_assignment[v] == (sign == 1):
                                sat = True
                                break
                    if not sat:
                        all_sat = False
                        break
                if all_sat:
                    return new_assignment
                else:
                    return None
            
            # Pick a variable to branch on
            # Choose variable that appears in most remaining clauses
            var_counts = {}
            for clause in remaining_clauses:
                for v, sign in clause:
                    if new_assignment[v] is None:
                        var_counts[v] = var_counts.get(v, 0) + 1
            
            if not var_counts:
                # No unassigned variables but not all satisfied - shouldn't happen
                return None
            
            # Pick variable with highest count, random tie-break
            max_count = max(var_counts.values())
            candidates = [v for v, c in var_counts.items() if c == max_count]
            var = candidates[rng.randint(0, len(candidates) - 1)]
            
            # Try both assignments
            for val in [True, False]:
                test_assignment = new_assignment.copy()
                test_assignment[var] = val
                
                # Filter clauses
                filtered_clauses = []
                for clause in remaining_clauses:
                    sat = False
                    for v, sign in clause:
                        if test_assignment[v] is not None:
                            if test_assignment[v] == (sign == 1):
                                sat = True
                                break
                    if not sat:
                        filtered_clauses.append(clause)
                
                result = dpll_recursive(test_assignment, filtered_clauses, depth + 1)
                if result is not None:
                    return result
            
            return None
        
        # Initialize with None for all variables
        initial_assignment = [None] * n
        result = dpll_recursive(initial_assignment, clauses_store, 0)
        
        if result is not None:
            # Convert None to random values
            final = []
            for v in range(n):
                if result[v] is None:
                    final.append(rng.random() < 0.5)
                else:
                    final.append(result[v])
            return final
        
        return None
    
    # Strategy 1: Try DPLL with limited depth first (fast for easy instances)
    rng = random.Random(42)
    
    # Try DPLL with small depth limit
    sol = dpll_with_propagation(rng, depth_limit=15)
    if sol is not None:
        # Verify
        if count_unsat(sol) == 0:
            return sol
    
    # Strategy 2: WalkSAT with simulated annealing
    max_restarts = 30
    for r in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Vary parameters for diversity
        initial_temp = 5.0 + r * 0.5
        cooling = 0.9997 - r * 0.00001
        max_steps = 30000 + r * 1000
        
        sol = walk_sat_simulated_annealing(rng, max_steps=max_steps, initial_temp=initial_temp, cooling=cooling)
        if sol is not None:
            if count_unsat(sol) == 0:
                return sol
    
    # Strategy 3: Pure random restarts with basic WalkSAT (fallback)
    for r in range(20):
        if time.time() > deadline:
            break
        
        assignment = [rng.random() < 0.5 for _ in range(n)]
        unsat_count = count_unsat(assignment)
        
        for step in range(20000):
            if time.time() > deadline:
                break
            
            if unsat_count == 0:
                return assignment
            
            # Pick random unsat clause
            unsat_clauses = [i for i in range(m) if not is_satisfied(assignment, i)]
            if not unsat_clauses:
                return assignment
            
            cl_idx = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
            
            # Pick variable to flip
            v, sign = clauses_store[cl_idx][rng.randint(0, 2)]
            
            old_unsat = unsat_count
            assignment[v] = not assignment[v]
            new_unsat = count_unsat(assignment)
            
            # Accept if improvement, or with some probability if worse
            if new_unsat <= old_unsat or rng.random() < 0.1:
                unsat_count = new_unsat
            else:
                assignment[v] = not assignment[v]
    
    # Last resort: return random assignment
    return [rng.random() < 0.5 for _ in range(n)]
