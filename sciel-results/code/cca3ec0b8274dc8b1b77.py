import time
import random
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    start_time = time.time()
    # Safety margin: 295s to avoid hitting 330s hard limit
    deadline = start_time + 295.0
    
    # Precomputation
    # Convert clauses to 0-based indices and signs
    # clause_data[i] = [(var_idx, is_positive), ...]
    clause_data = []
    var_clauses = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses):
        cl_store = []
        for l in clause:
            v = abs(l) - 1
            sign = 1 if l > 0 else -1
            cl_store.append((v, sign))
            var_clauses[v].append(i)
        clause_data.append(cl_store)
    
    m = len(clauses)
    
    # Helper to check if a clause is satisfied given assignment
    def is_satisfied(assignment, clause_idx):
        for v, sign in clause_data[clause_idx]:
            if assignment[v] == (sign == 1):
                return True
        return False

    def get_unsat_clauses(assignment):
        return [i for i in range(m) if not is_satisfied(assignment, i)]

    def count_unsat(assignment, unsat_list=None):
        if unsat_list is not None:
            return len(unsat_list)
        return len(get_unsat_clauses(assignment))

    def dpll_limited(rng, depth_limit=10):
        """
        Lightweight DPLL with unit propagation.
        Returns assignment if solved, else None.
        """
        # Simple recursive DPLL with unit propagation
        # For n=50, depth 10 is very fast
        
        def propagate(assignment, clauses_remaining):
            """
            assignment: list of bool or None
            clauses_remaining: list of clause indices
            Returns (new_assignment, new_clauses, contradiction)
            """
            new_assign = assignment[:]
            new_clauses = list(clauses_remaining)
            changed = True
            while changed:
                changed = False
                next_clauses = []
                for ci in new_clauses:
                    sat = False
                    unassigned = []
                    for v, sign in clause_data[ci]:
                        if new_assign[v] is not None:
                            if new_assign[v] == (sign == 1):
                                sat = True
                                break
                        else:
                            unassigned.append((v, sign))
                    
                    if sat:
                        continue
                    if not unassigned:
                        return None, None, True
                    if len(unassigned) == 1:
                        v, sign = unassigned[0]
                        new_assign[v] = (sign == 1)
                        changed = True
                    else:
                        next_clauses.append(ci)
                new_clauses = next_clauses
            
            return new_assign, new_clauses, False

        def dpll_rec(assignment, clauses_remaining, depth):
            if depth > depth_limit:
                return None
            
            # Check if all satisfied
            if not clauses_remaining:
                return assignment
            
            new_assign, rem_clauses, contra = propagate(assignment, clauses_remaining)
            if contra:
                return None
            
            if not rem_clauses:
                return new_assign
            
            # Pick variable
            # Heuristic: variable appearing in most remaining clauses
            var_counts = {}
            for ci in rem_clauses:
                for v, _ in clause_data[ci]:
                    if new_assign[v] is None:
                        var_counts[v] = var_counts.get(v, 0) + 1
            
            if not var_counts:
                # All assigned but clauses remain? Should be handled by propagation
                return None
                
            max_count = max(var_counts.values())
            candidates = [v for v, c in var_counts.items() if c == max_count]
            var = candidates[rng.randint(0, len(candidates) - 1)]
            
            # Try True then False
            for val in [True, False]:
                trial_assign = new_assign[:]
                trial_assign[var] = val
                
                # Filter clauses
                filtered = []
                for ci in rem_clauses:
                    sat = False
                    for v, sign in clause_data[ci]:
                        if trial_assign[v] is not None:
                            if trial_assign[v] == (sign == 1):
                                sat = True
                                break
                    if not sat:
                        filtered.append(ci)
                        
                res = dpll_rec(trial_assign, filtered, depth + 1)
                if res is not None:
                    # Fill unassigned with random
                    final = []
                    for i in range(n):
                        if res[i] is None:
                            final.append(rng.random() < 0.5)
                        else:
                            final.append(res[i])
                    return final
            return None

        initial = [None] * n
        return dpll_rec(initial, list(range(m)), 0)

    # Strategy 1: Quick DPLL
    rng = random.Random(42)
    sol = dpll_limited(rng, depth_limit=12)
    if sol is not None:
        if count_unsat(sol) == 0:
            return sol

    # Strategy 2: Weighted WalkSAT
    # This is the structural change: Clause weighting + Min-Conflicts heuristic
    
    def weighted_walksat(rng, max_steps, noise_prob=0.2, weight_decay=0.95):
        """
        WalkSAT with clause weighting.
        """
        assignment = [rng.random() < 0.5 for _ in range(n)]
        unsat_clauses = get_unsat_clauses(assignment)
        
        if not unsat_clauses:
            return assignment
        
        # Weights for each clause
        weights = [1.0] * m
        # Track unsat status for quick updates
        # We will recompute unsat_clauses list after each flip for simplicity, 
        # as m is small (~50-100). O(m) update is fine.
        
        best_unsat = len(unsat_clauses)
        best_assignment = assignment.copy()
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if not unsat_clauses:
                return assignment
            
            # Select clause with probability proportional to weight
            # Normalize weights for selection
            total_weight = sum(weights[i] for i in unsat_clauses)
            if total_weight == 0:
                total_weight = 1.0
                for i in unsat_clauses:
                    weights[i] = 1.0
            
            r = rng.random() * total_weight
            cumulative = 0
            selected_clause = unsat_clauses[0]
            for ci in unsat_clauses:
                cumulative += weights[ci]
                if r <= cumulative:
                    selected_clause = ci
                    break
            
            # Increase weight of selected clause
            weights[selected_clause] += 1.0
            
            # Select variable to flip
            # Min-Conflicts: pick var that satisfies most other unsat clauses
            clause_vars = clause_data[selected_clause]
            
            best_var = -1
            best_score = -1
            scores = []
            
            for v, sign in clause_vars:
                score = 0
                # Check other unsat clauses
                for other_ci in unsat_clauses:
                    if other_ci == selected_clause:
                        continue
                    # Check if flipping v satisfies other_ci
                    # Current assignment: assignment[v]
                    # New assignment: not assignment[v]
                    # Literal in other_ci for v: (v, sign_other)
                    # Satisfied if (not assignment[v]) == (sign_other == 1)
                    for v2, sign2 in clause_data[other_ci]:
                        if v2 == v:
                            if (not assignment[v]) == (sign2 == 1):
                                score += 1
                            break
                scores.append(score)
                if score > best_score:
                    best_score = score
                    best_var = v
            
            # Apply noise
            if rng.random() < noise_prob:
                v_to_flip = clause_vars[rng.randint(0, 2)][0]
            else:
                if best_var == -1:
                    v_to_flip = clause_vars[rng.randint(0, 2)][0]
                else:
                    # Pick randomly among top scorers if tie? Or just best.
                    # Let's add a bit of randomness among top 2 if close
                    if best_score > 0:
                        top_vars = [v for v, s in zip([v for v, _ in clause_vars], scores) if s == best_score]
                        v_to_flip = top_vars[rng.randint(0, len(top_vars) - 1)]
                    else:
                        v_to_flip = clause_vars[rng.randint(0, 2)][0]
            
            # Flip
            assignment[v_to_flip] = not assignment[v_to_flip]
            
            # Update unsat clauses and weights
            # We need to update weights for clauses that became sat/unsat
            # Simplest: recompute unsat_clauses
            old_unsat_set = set(unsat_clauses)
            new_unsat_clauses = get_unsat_clauses(assignment)
            new_unsat_set = set(new_unsat_clauses)
            
            # Decay weights for unsat clauses
            for ci in new_unsat_clauses:
                weights[ci] *= weight_decay
            
            # Reset weight for clauses that became satisfied? 
            # Or just let them decay. 
            # Clauses that were unsat and are now sat: weight should drop significantly
            for ci in old_unsat_set:
                if ci not in new_unsat_set:
                    weights[ci] = 1.0 # Reset
            
            # Clauses that were sat and are now unsat: weight stays or increases slightly?
            # Usually reset to 1.0 or keep low.
            for ci in new_unsat_set:
                if ci not in old_unsat_set:
                    weights[ci] = 1.0
            
            unsat_clauses = new_unsat_clauses
            
            if len(unsat_clauses) < best_unsat:
                best_unsat = len(unsat_clauses)
                best_assignment = assignment.copy()
                
            if not unsat_clauses:
                return assignment
        
        if best_unsat == 0:
            return best_assignment
        return None

    # Run Weighted WalkSAT with restarts
    max_restarts = 100
    base_steps = 5000
    for r in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Adjust parameters
        # More noise in early restarts, less later? Or constant?
        noise = 0.3 if r < 10 else 0.2
        steps = base_steps + r * 500
        
        sol = weighted_walksat(rng, steps, noise_prob=noise)
        if sol is not None:
            if count_unsat(sol) == 0:
                return sol

    # Fallback: Return best effort or random
    # Try one last long run with pure random WalkSAT if time permits
    if time.time() < deadline - 10:
        assignment = [rng.random() < 0.5 for _ in range(n)]
        unsat_clauses = get_unsat_clauses(assignment)
        while time.time() < deadline - 1 and unsat_clauses:
            ci = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
            v, sign = clause_data[ci][rng.randint(0, 2)]
            assignment[v] = not assignment[v]
            unsat_clauses = get_unsat_clauses(assignment)
        if not unsat_clauses:
            return assignment
        return assignment

    return [rng.random() < 0.5 for _ in range(n)]
