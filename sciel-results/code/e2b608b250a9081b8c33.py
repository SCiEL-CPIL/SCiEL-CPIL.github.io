import time
import random
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 295  # Safety margin before 330s kill
    
    # Precomputation: Convert to 0-based indices and store signs
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

    def fast_dpll(rng, time_limit=15.0):
        """
        Optimized DPLL with Unit Propagation and Pure Literal Elimination.
        Uses efficient clause tracking.
        """
        dpll_start = time.time()
        
        # Use lists for active clauses and their status
        # active_clauses: list of clause indices that are not yet satisfied
        # assignment: list of None/True/False
        assignment = [None] * n
        
        # Track which clauses are active (not satisfied)
        active = set(range(m))
        
        # Helper to check clause status
        def clause_status(cl_idx, assignment):
            unassigned = 0
            for v, sign in clauses_store[cl_idx]:
                if assignment[v] is not None:
                    if assignment[v] == (sign == 1):
                        return 1  # satisfied
                else:
                    unassigned += 1
            return -1 if unassigned == 0 else 0  # -1: falsified, 0: open
        
        def propagate():
            """Apply unit propagation and pure literal elimination."""
            nonlocal active, assignment
            
            changed = True
            while changed:
                changed = False
                
                # Unit propagation
                new_active = set()
                for cl_idx in active:
                    status = clause_status(cl_idx, assignment)
                    if status == 1:
                        continue  # satisfied, remove
                    elif status == -1:
                        return False  # contradiction
                    
                    # Count unassigned literals
                    unassigned_lits = []
                    for v, sign in clauses_store[cl_idx]:
                        if assignment[v] is None:
                            unassigned_lits.append((v, sign))
                    
                    if len(unassigned_lits) == 1:
                        v, sign = unassigned_lits[0]
                        assignment[v] = (sign == 1)
                        changed = True
                    else:
                        new_active.add(cl_idx)
                
                active = new_active
                if not active:
                    return True
                
                # Pure literal elimination
                var_signs = {}
                for cl_idx in active:
                    for v, sign in clauses_store[cl_idx]:
                        if assignment[v] is None:
                            if v not in var_signs:
                                var_signs[v] = set()
                            var_signs[v].add(sign)
                
                pure_vars = []
                for v, signs in var_signs.items():
                    if len(signs) == 1:
                        pure_vars.append((v, list(signs)[0]))
                
                if pure_vars:
                    changed = True
                    for v, sign in pure_vars:
                        assignment[v] = (sign == 1)
                    
                    # Re-filter active clauses
                    new_active = set()
                    for cl_idx in active:
                        status = clause_status(cl_idx, assignment)
                        if status != 1:
                            if status == -1:
                                return False
                            new_active.add(cl_idx)
                    active = new_active
                    if not active:
                        return True
            
            return True
        
        def dpll_recursive(depth=0):
            if time.time() - dpll_start > time_limit:
                return None
            
            if not propagate():
                return None
            
            if not active:
                # All clauses satisfied
                final = assignment.copy()
                for i in range(n):
                    if final[i] is None:
                        final[i] = rng.random() < 0.5
                return final
            
            # Variable selection: pick variable with highest frequency in active clauses
            var_freq = {}
            for cl_idx in active:
                for v, sign in clauses_store[cl_idx]:
                    if assignment[v] is None:
                        var_freq[v] = var_freq.get(v, 0) + 1
            
            if not var_freq:
                final = assignment.copy()
                for i in range(n):
                    if final[i] is None:
                        final[i] = rng.random() < 0.5
                return final
            
            max_freq = max(var_freq.values())
            candidates = [v for v, c in var_freq.items() if c == max_freq]
            var = candidates[rng.randint(0, len(candidates) - 1)]
            
            # Try both values
            for val in [True, False]:
                old_assignment = assignment.copy()
                old_active = active.copy()
                
                assignment[var] = val
                
                # Filter clauses
                new_active = set()
                for cl_idx in active:
                    status = clause_status(cl_idx, assignment)
                    if status == 1:
                        continue
                    elif status == -1:
                        assignment[:] = old_assignment
                        active = old_active
                        break
                    new_active.add(cl_idx)
                else:
                    active = new_active
                    result = dpll_recursive(depth + 1)
                    if result is not None:
                        return result
                    # Backtrack
                    assignment[:] = old_assignment
                    active = old_active
                    continue
                # Contradiction, backtrack
                assignment[:] = old_assignment
                active = old_active
            
            return None
        
        return dpll_recursive(0)

    def local_search(rng, time_limit, initial_temp=5.0, cooling=0.9999, min_temp=0.01):
        """
        Improved local search with smooth surrogate and adaptive noise.
        """
        ls_start = time.time()
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Pre-compute unsat set
        unsat_clauses = set()
        for i in range(m):
            if not is_satisfied(assignment, i):
                unsat_clauses.add(i)
        
        current_unsat = len(unsat_clauses)
        if current_unsat == 0:
            return assignment
        
        best_unsat = current_unsat
        best_assignment = assignment.copy()
        temp = initial_temp
        no_improve = 0
        max_no_improve = 5000
        
        # Smooth surrogate: use soft-min of clause violations
        # For each unsat clause, contribution is 1.0; for sat, 0.0
        # Smooth version: sum of exp(-alpha * sat_count) for each clause
        alpha = 2.0
        
        def smooth_score(assignment):
            score = 0.0
            for i in range(m):
                sat_count = 0
                for v, sign in clauses_store[i]:
                    if assignment[v] == (sign == 1):
                        sat_count += 1
                if sat_count == 0:
                    score += 1.0
                else:
                    score += math.exp(-alpha * sat_count)
            return score
        
        while time.time() - ls_start < time_limit:
            if current_unsat == 0:
                return assignment
            
            # Pick a random unsatisfied clause
            unsat_list = list(unsat_clauses)
            if not unsat_list:
                break
            cl_idx = unsat_list[rng.randint(0, len(unsat_list) - 1)]
            
            # Evaluate flipping each variable in the clause
            best_var = -1
            best_delta = float('inf')
            candidates = []
            
            for v, sign in clauses_store[cl_idx]:
                # Calculate delta in unsat count if we flip v
                delta = 0
                
                # The chosen clause becomes satisfied
                delta -= 1
                
                # Check other clauses containing v
                for c_idx, c_sign in var_clauses[v]:
                    if c_idx == cl_idx:
                        continue
                    
                    c_is_sat = is_satisfied(assignment, c_idx)
                    
                    if c_is_sat:
                        # Check if v is the unique satisfier
                        unique = True
                        for v2, s2 in clauses_store[c_idx]:
                            if v2 == v:
                                continue
                            if assignment[v2] == (s2 == 1):
                                unique = False
                                break
                        if unique:
                            delta += 1
                    else:
                        # Flipping v satisfies this clause
                        delta -= 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_var = v
                    candidates = [v]
                elif delta == best_delta:
                    candidates.append(v)
            
            # With some probability, use smooth score to guide choice
            use_smooth = rng.random() < 0.3
            if use_smooth and candidates:
                # Pick candidate that minimizes smooth score after flip
                best_smooth = float('inf')
                best_smooth_var = candidates[0]
                for v in candidates:
                    old_val = assignment[v]
                    assignment[v] = not old_val
                    s = smooth_score(assignment)
                    assignment[v] = old_val
                    if s < best_smooth:
                        best_smooth = s
                        best_smooth_var = v
                best_var = best_smooth_var
            
            if best_var == -1:
                best_var = candidates[rng.randint(0, len(candidates) - 1)] if candidates else rng.randint(0, n - 1)
            
            # Add noise: with probability p, flip a random variable from the clause
            noise_prob = 0.1 + 0.2 * (temp / initial_temp)
            if rng.random() < noise_prob:
                v, sign = clauses_store[cl_idx][rng.randint(0, 2)]
                best_var = v
            
            # Perform flip
            old_count = current_unsat
            assignment[best_var] = not assignment[best_var]
            
            # Update unsat set
            for c_idx, c_sign in var_clauses[best_var]:
                was_unsat = c_idx in unsat_clauses
                now_sat = is_satisfied(assignment, c_idx)
                if was_unsat and now_sat:
                    unsat_clauses.discard(c_idx)
                elif not was_unsat and not now_sat:
                    unsat_clauses.add(c_idx)
            
            new_count = len(unsat_clauses)
            delta = new_count - old_count
            
            # Acceptance
            if delta <= 0:
                current_unsat = new_count
                if new_count < best_unsat:
                    best_unsat = new_count
                    best_assignment = assignment.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                if temp > min_temp:
                    prob = math.exp(-delta / temp)
                    if rng.random() < prob:
                        current_unsat = new_count
                        no_improve += 1
                    else:
                        # Reject
                        assignment[best_var] = not assignment[best_var]
                        for c_idx, c_sign in var_clauses[best_var]:
                            was_unsat_after = c_idx in unsat_clauses
                            now_sat_after = is_satisfied(assignment, c_idx)
                            if was_unsat_after and now_sat_after:
                                unsat_clauses.discard(c_idx)
                            elif not was_unsat_after and not now_sat_after:
                                unsat_clauses.add(c_idx)
                        current_unsat = old_count
                        no_improve += 1
                else:
                    # Reject
                    assignment[best_var] = not assignment[best_var]
                    for c_idx, c_sign in var_clauses[best_var]:
                        was_unsat_after = c_idx in unsat_clauses
                        now_sat_after = is_satisfied(assignment, c_idx)
                        if was_unsat_after and now_sat_after:
                            unsat_clauses.discard(c_idx)
                        elif not was_unsat_after and not now_sat_after:
                            unsat_clauses.add(c_idx)
                    current_unsat = old_count
                    no_improve += 1
            
            # Cooling
            temp = max(temp * cooling, min_temp)
            
            if no_improve > max_no_improve:
                break
        
        if best_unsat == 0:
            return best_assignment
        return None

    # Strategy Execution
    rng = random.Random(42)
    
    # 1. Try DPLL first
    dpll_time = 15.0
    sol = fast_dpll(rng, time_limit=dpll_time)
    
    if sol is not None and count_unsat(sol) == 0:
        return sol
    
    # 2. Local search with multiple restarts
    elapsed = time.time() - start_time
    remaining = deadline - time.time()
    if remaining <= 0:
        return [rng.random() < 0.5 for _ in range(n)]
    
    # Distribute time among restarts with increasing budget
    restarts = 8
    time_per_restart = remaining / restarts
    
    for r in range(restarts):
        if time.time() > deadline - 2:
            break
        
        # Adaptive parameters
        base_temp = 6.0
        temp = base_temp * (1.0 - 0.5 * r / restarts) + 1.0
        cooling = 0.9998
        t_limit = time_per_restart * (1.0 + 0.1 * r)  # Slightly increase later restarts
        
        sol = local_search(rng, t_limit, initial_temp=temp, cooling=cooling)
        
        if sol is not None and count_unsat(sol) == 0:
            return sol
    
    # 3. Final attempt: use best from last local search
    # Just return a random assignment as fallback
    return [rng.random() < 0.5 for _ in range(n)]

if __name__ == "__main__":
    pass
