"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard: 330s limit, use 290s for safety
    start_time = time.time()
    timeout = 290.0
    
    m = len(clauses)
    if m == 0:
        return [True] * n
        
    # Preprocess clauses into numpy arrays for fast evaluation
    # clause_data: shape (m, 3), values are +/-1 (sign)
    # clause_vars: shape (m, 3), values are var indices (0-based)
    clause_data = np.empty((m, 3), dtype=np.int8)
    clause_vars = np.empty((m, 3), dtype=np.int32)
    
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_data[i, j] = 1 if lit > 0 else -1
            clause_vars[i, j] = abs(lit) - 1
            
    # Build variable-to-clauses mapping for potential future use or analysis
    # Not strictly needed for the core loop if we stick to clause-centric search
    
    def is_satisfied(assignment):
        """
        Return boolean array of satisfied clauses.
        assignment: (n,) bools
        """
        # Get the values of the variables in each clause position
        # clause_vars is (m, 3) indices
        vals = assignment[clause_vars]  # (m, 3) bools
        
        # For each literal, it's true if (assignment[var] == (sign == 1))
        # sign == 1 means positive literal, so we want assignment[var] to be True
        # sign == -1 means negative literal, so we want assignment[var] to be False
        sign_bool = (clause_data == 1)  # (m, 3) bools
        
        # True if vals matches sign_bool
        literal_true = (vals == sign_bool)  # (m, 3)
        
        # Clause is satisfied if any literal is true
        return np.any(literal_true, axis=1)  # (m,) bools

    def count_unsat_clauses_for_vars(assignment, unsat_indices):
        """
        For each variable, count how many unsatisfied clauses it appears in.
        This helps in selecting which variable to flip.
        """
        counts = np.zeros(n, dtype=np.int32)
        if len(unsat_indices) == 0:
            return counts
        
        # Extract variables from unsatisfied clauses
        unsat_vars = clause_vars[unsat_indices] # (k, 3)
        
        # Use np.bincount for fast counting
        # Flatten the array and count occurrences
        flat_vars = unsat_vars.flatten()
        counts = np.bincount(flat_vars, minlength=n)
        
        return counts

    def walk_sat(restart_assignment, max_flips, seed, p=0.55):
        """
        Run WalkSAT with the given initial assignment and max flips.
        p is the probability of choosing a variable that minimizes the number of 
        unsatisfied clauses (greedy) vs random (exploration).
        """
        rng = random.Random(seed)
        assignment = restart_assignment.copy()
        
        # Pre-compute satisfied status
        sat = is_satisfied(assignment)
        if np.all(sat):
            return assignment.tolist()
            
        for step in range(max_flips):
            if time.time() - start_time > timeout:
                return None
            
            # Identify unsatisfied clauses
            unsat_indices = np.where(~sat)[0]
            if len(unsat_indices) == 0:
                return assignment.tolist()
            
            # Pick a random unsatisfied clause
            clause_idx = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]
            
            # Select a variable from this clause to flip
            # Heuristic: 
            # 1. Calculate the impact of flipping each variable in the clause.
            #    Flipping a variable in an unsatisfied clause will satisfy it.
            #    However, it might unsatisfy other clauses where this variable is the *only* true literal.
            #    
            #    A simpler and effective heuristic for WalkSAT:
            #    - With probability p, pick the variable that appears in the most OTHER unsatisfied clauses.
            #    - With probability 1-p, pick a random variable from the clause.
            
            clause_vars_list = clause_vars[clause_idx]
            
            if rng.random() < p:
                # Greedy: pick variable in this clause that is in the most other unsat clauses
                # We need counts of unsat clauses per variable.
                # To avoid recalculating counts for all vars every step (expensive), 
                # we can approximate or just count for the 3 vars in the current clause.
                
                # Let's count how many unsatisfied clauses each of the 3 vars appears in.
                # This is O(k) where k is num unsat clauses. 
                # To make this faster, we can pre-compute a global count if we update it incrementally, 
                # but that's complex. 
                # Alternative: Just count occurrences in the current list of unsat indices for these 3 specific vars.
                
                # Optimization: 
                # If we maintain a global array `var_unsat_count`, updating it after each flip is O(degree).
                # Let's try a simpler approach first: 
                # Count occurrences of the 3 candidates in the unsat_indices set.
                
                # For performance, let's just pick the one with the highest "frequency" in the unsat set.
                # Since unsat_indices can be large, a full bincount might be expensive if done every step.
                # However, n=150, m is likely ~450-600. 
                # Let's stick to a lightweight heuristic: 
                # Just pick a random var from the clause, but with a bias?
                
                # Actually, the standard WalkSAT heuristic is:
                # Pick a var from the clause.
                # The "Walk" is just random.
                # The "SAT" part is usually handled by restarts or better initial assignments.
                
                # Let's try a slightly better heuristic: 
                # Count how many unsatisfied clauses each of the 3 variables appears in.
                # We can do this by iterating through unsat_indices? No, that's O(m).
                
                # Let's use a global counter approach for speed.
                # We'll maintain `var_unsat_count` outside the loop.
                
                pass # Logic handled by outer scope variables if we refactor, but let's keep it simple inside.
                
                # Simple heuristic: 
                # If we don't want to pay O(m) per step, we can just pick randomly.
                # But let's try to be a bit smarter. 
                # We'll compute the count for just the 3 variables by checking their presence in unsat_indices.
                # This is inefficient if done naively.
                
                # Let's stick to pure random for now but with a very high restart rate, 
                # OR use the `count_unsat_clauses_for_vars` function but only if the unsat set is small?
                
                # Actually, for n=150, m~500, O(m) per step is 500 operations. 
                # 500 * 10000 steps = 5,000,000 ops. Python is slow, but NumPy is fast.
                # Let's try using the bincount approach but only if we can do it efficiently.
                
                # Let's revert to a simpler, faster heuristic that is known to work well:
                # Pick a random variable from the clause.
                # The power comes from the restarts.
                
                var_to_flip = clause_vars_list[rng.randint(0, 2)]
            else:
                var_to_flip = clause_vars_list[rng.randint(0, 2)]
                
            # Wait, the heuristic above was identical. Let's implement the "min-conflicts" style for the 3 vars.
            # We want to minimize the number of unsatisfied clauses AFTER the flip.
            # Flipping var v in clause C (unsat) satisfies C.
            # It may unsatisfy other clauses where v is the only true literal.
            
            # Let's calculate the "delta" for each of the 3 vars.
            # This requires knowing which clauses are "critical" for v (i.e., satisfied only by v).
            
            # This is getting complex. Let's stick to the standard WalkSAT:
            # 1. Pick random unsat clause.
            # 2. Pick random var from it.
            # 3. Flip.
            
            # The previous code did this. Why was it stuck? 
            # Maybe the restart strategy or the initial assignment?
            
            # Let's try a different heuristic: 
            # Pick the variable from the clause that appears in the MOST unsatisfied clauses.
            # We will compute this count efficiently using NumPy.
            
            # We need the count of unsat clauses per variable.
            # Let's maintain a global `var_unsat_count` array.
            
            pass # Placeholder for logic below
            
            # --- Revised Loop Logic ---
            
            # 1. Get unsat indices
            unsat_indices = np.where(~sat)[0]
            if len(unsat_indices) == 0:
                return assignment.tolist()
            
            # 2. Pick a random unsat clause
            clause_idx = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]
            clause_vars_list = clause_vars[clause_idx]
            
            # 3. Select variable to flip
            # Heuristic: Minimize conflicts.
            # For each var in the clause, count how many OTHER unsat clauses it appears in.
            # We'll use a global count array `var_unsat_count` that we update.
            
            # To avoid O(m) update every step, we can just pick randomly with high probability,
            # or use a cheaper heuristic.
            
            # Let's try: Pick the var with the highest `var_unsat_count`.
            # If tie, random.
            
            # We need to update `var_unsat_count` after each flip.
            # This is O(degree) per flip.
            
            # Let's implement this properly.
            
            # Get the 3 vars
            v1, v2, v3 = clause_vars_list
            
            # Pick the one with the highest count
            # If counts are equal, pick random
            c1 = var_unsat_count[v1]
            c2 = var_unsat_count[v2]
            c3 = var_unsat_count[v3]
            
            max_c = max(c1, c2, c3)
            candidates = []
            if c1 == max_c: candidates.append(v1)
            if c2 == max_c: candidates.append(v2)
            if c3 == max_c: candidates.append(v3)
            
            var_to_flip = candidates[rng.randint(0, len(candidates)-1)]
            
            # 4. Flip the variable
            assignment[var_to_flip] = not assignment[var_to_flip]
            
            # 5. Update satisfaction status and counts
            # This is the expensive part.
            # We need to update `sat` and `var_unsat_count`.
            
            # Update `sat` for clauses affected by var_to_flip
            # We need a mapping from var to clauses.
            
            # Let's build `var_to_clauses` once outside.
            
            affected_clauses = var_to_clauses[var_to_flip]
            
            # For each affected clause, check if it's now satisfied.
            # And update `var_unsat_count` for all vars in those clauses?
            # No, `var_unsat_count` should reflect the CURRENT state of unsat clauses.
            
            # Updating `var_unsat_count` incrementally is tricky because a clause going from SAT to UNSAT 
            # or UNSAT to SAT affects the counts of ALL 3 vars in that clause.
            
            # Let's just recompute `sat` and `var_unsat_count` from scratch every step?
            # O(m) for sat, O(m) for counts.
            # 2 * 500 * 10000 = 10,000,000 ops. Might be too slow in Python.
            
            # Alternative: 
            # Don't maintain `var_unsat_count`. Just pick randomly.
            # The "Walk" in WalkSAT is random.
            
            # Let's go back to Random Flip, but with a better restart policy.
            
            # Actually, let's try a "Soft" heuristic:
            # Pick a var from the clause.
            # Bias towards vars that are in more unsat clauses.
            # We can approximate this by just picking randomly, but if we have time, 
            # we can do a small local search.
            
            # Let's stick to the simplest robust method:
            # 1. Random unsat clause.
            # 2. Random var from clause.
            # 3. Flip.
            # 4. Check if solved.
            
            # The key is the restarts.
            
            pass # This block is messy. I will rewrite the function cleanly.
            
        return None

    # --- Clean Implementation ---
    
    # Build var_to_clauses mapping
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_to_clauses[v].append(i)
            
    def walk_sat_optimized(restart_assignment, max_flips, seed):
        rng = random.Random(seed)
        assignment = restart_assignment.copy()
        
        # Initial check
        sat = is_satisfied(assignment)
        if np.all(sat):
            return assignment.tolist()
            
        for step in range(max_flips):
            if time.time() - start_time > timeout:
                return None
            
            unsat_indices = np.where(~sat)[0]
            if len(unsat_indices) == 0:
                return assignment.tolist()
            
            # Pick a random unsatisfied clause
            clause_idx = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]
            
            # Pick a random variable from that clause
            # To improve on pure random, we can use a "weighted" random choice
            # where weight is inversely proportional to the number of times the var 
            # has been flipped recently? Or just pure random.
            
            # Let's try: Pick the variable that appears in the most unsatisfied clauses.
            # To do this efficiently without O(m) per step, we can just pick randomly 
            # but with a small probability of "greedy" choice if we pre-compute counts?
            
            # Let's stick to Random for simplicity and speed, relying on restarts.
            var_to_flip = clause_vars[clause_idx, rng.randint(0, 2)]
            
            # Flip
            assignment[var_to_flip] = not assignment[var_to_flip]
            
            # Update sat for affected clauses
            # Only clauses containing var_to_flip can change status
            for c_idx in var_to_clauses[var_to_flip]:
                # Check if clause c_idx is satisfied
                # clause_vars[c_idx] are the 3 vars
                # clause_data[c_idx] are the signs
                v1, v2, v3 = clause_vars[c_idx]
                s1, s2, s3 = clause_data[c_idx]
                
                # literal 1 true if (assignment[v1] == (s1==1))
                lit1 = (assignment[v1] == (s1 == 1))
                lit2 = (assignment[v2] == (s2 == 1))
                lit3 = (assignment[v3] == (s3 == 1))
                
                new_sat = lit1 or lit2 or lit3
                sat[c_idx] = new_sat
                
            # Check if all satisfied
            # np.all(sat) is O(m). We can keep a count of unsat clauses.
            # Let's maintain `unsat_count`.
            
            # Optimization: Maintain unsat_count
            # But we didn't initialize it. Let's just use np.all for now, it's C-optimized.
            
            if np.all(sat):
                return assignment.tolist()
                
        return None

    # Main search loop
    best_result = None
    rng = random.Random(0)
    restart_count = 0
    base_flips = 1000
    max_restarts = 10000
    
    while time.time() - start_time < timeout - 5 and restart_count < max_restarts:
        # Create a random initial assignment
        initial = np.array([rng.random() < 0.5 for _ in range(n)])
        
        # Run WalkSAT
        # Increase flip budget slightly with restarts? Or keep it constant?
        # Luby sequence is good for restarts.
        # Let's use a simple geometric increase or fixed budget.
        # Fixed budget of 2000 flips per restart is usually enough for small instances.
        max_flips = 2000 + (restart_count % 10) * 500
        
        result = walk_sat_optimized(initial, max_flips, seed=restart_count * 12345 + 67890)
        
        if result is not None:
            best_result = result
            break
        
        restart_count += 1
        
    if best_result is not None:
        return best_result
    
    # Fallback
    return [rng.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
