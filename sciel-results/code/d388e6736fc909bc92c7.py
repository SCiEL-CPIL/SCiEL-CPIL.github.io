"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def solve_walksat(vals, time_budget=30.0, seed=42):
        """
        Solve Max Sum-Free Subset using WalkSAT-style local search.
        Variables: x_i = 1 if vals[i] is in the set, 0 otherwise.
        Clauses: For every triple (i, j, k) such that vals[i] + vals[j] = vals[k],
                 the clause is (NOT x_i OR NOT x_j OR NOT x_k).
                 This is a 3-CNF clause, but we can handle it in WalkSAT by 
                 flipping a variable in a violated clause.
                 Wait, standard WalkSAT is for 2-SAT (clauses of size 2).
                 For 3-CNF, it's Minisat-style. 
                 However, the objective is to MAXIMIZE sum(x_i) subject to clauses.
                 This is Max-SAT.
                 
                 Alternative view:
                 We want to maximize the number of 1s.
                 Constraint: No triple (i,j,k) with vals[i]+vals[j]=vals[k] can all be 1.
                 
                 Local Search Move:
                 Start with a random assignment or a greedy one.
                 Score = Number of 1s - Penalty * (Number of violated clauses).
                 Or just try to fix violations while maintaining/growing size.
                 
                 Let's use a "Repair" strategy:
                 1. Start with a large random set or greedy set.
                 2. While there are violations:
                      Pick a random violated triple (i,j,k).
                      Randomly choose one of i,j,k to flip to 0 (exclude).
                 3. Then try to add elements back (greedy or random).
                 
                 This is essentially a randomized greedy repair.
                 To make it "WalkSAT-like", we can use a probability to flip a variable 
                 that appears in many violated clauses (min-unsat heuristic).
                 
                 Let's implement a more robust local search:
                 State: bitmask of included elements.
                 Objective: Maximize popcount(mask) such that no clause is violated.
                 
                 Move: 
                 - If current state is valid: Try to flip a 0 to 1. If it causes violation, revert or accept with penalty?
                   No, we must maintain validity. So we can only add if it doesn't violate.
                   This is just greedy.
                 - To escape local optima: 
                   Allow flipping a 1 to 0 even if it doesn't fix a violation? No, that reduces size.
                   Instead: "Kick" - randomly flip some variables (including 1->0 and 0->1) to change the landscape, then repair.
                 
                 Strategy:
                 1. Greedy initialization.
                 2. Loop until time up:
                    a. Try to add any possible element (greedy).
                    b. If stuck (no more can be added), perform a "kick":
                       - Randomly remove k elements (k=1,2,3).
                       - Try to add random elements to fill the gap.
                       - If size increases, keep. Else, maybe keep if it opens up more additions?
                       - Simpler: Randomly remove 1-3 elements, then run greedy repair to max.
                       - If new size > old size, accept.
                       - If new size == old size, accept with some probability (plateau walking).
                       - If new size < old size, reject.
                 
                 This is a form of Simulated Annealing / Randomized Hill Climbing.
                 The "WalkSAT" inspiration is using clause weights or frequency to guide the removal.
                 Let's track the "pressure" of each variable: how many violated clauses it participates in.
                 When stuck, remove a variable with high pressure? But if it's valid, there are no violated clauses.
                 
                 The issue with greedy is that it gets stuck in local maxima.
                 Example: A and B conflict with C. Greedy picks A, then B, then can't pick C.
                 If C was "better" (allows more other elements), we should have picked C.
                 
                 So, "Kick" strategy:
                 1. Identify a random element x in the current set.
                 2. Remove x.
                 3. Try to add x again? No.
                 4. Try to add a random element y not in the set.
                 5. If y can be added, do it.
                 6. This is a 1-swap (remove x, add y).
                 7. If size stays same, we might have improved the "structure" to allow more additions later.
                 8. After a swap, run greedy to fill any new opportunities.
                 
                 Let's implement "Swap + Greedy Repair".
                 """
        n = len(vals)
        if n == 0:
            return []
        
        vals_set = set(vals)
        index_of_val = {v: i for i, v in enumerate(vals)}
        
        # Precompute conflicts:
        # conflict_mask[i]: bitmask of j such that vals[i] + vals[j] is in vals_set.
        # If we add i, we must ensure no j in S is in conflict_mask[i] AND no k in S is double_mask[i].
        # Wait, the check "can_add" is:
        # (S & sum_mask[i]) == 0 AND (S & double_mask[i]) == 0.
        # Where sum_mask[i] contains j such that vals[i]+vals[j] in vals_set.
        # Note: if vals[i]+vals[j] = vals[k], then k is in sum_mask[i] AND i is in sum_mask[j].
        # Actually, if we add i, we check if any existing a in S has a+i in S.
        # So we need to check if (S & mask_of_a_such_that_a_plus_i_in_vals) is empty.
        # Let's precompute `incompat[i]`: bitmask of j such that vals[i] + vals[j] is in vals_set.
        # If i is in S, and j is in S, and vals[i]+vals[j] is in vals_set, that's a violation?
        # No, the violation is if vals[i]+vals[j] = vals[k] AND k is in S.
        # So the conflict is ternary.
        
        # However, for the "can_add" check when adding i to S:
        # We need to ensure that for all a in S, a+i is NOT in S.
        # This is equivalent to: (S & mask_i) == 0, where mask_i is the set of a such that a+i is in vals_set.
        # Let's precompute `sum_partner_mask[i]`: bitmask of a such that vals[a] + vals[i] is in vals_set.
        # Then `can_add(S, i)` is `(S & sum_partner_mask[i]) == 0` AND `(S & double_mask[i]) == 0`.
        # Note: `double_mask[i]` is subset of `sum_partner_mask[i]`?
        # If 2*vals[i] = vals[k], then vals[i]+vals[i]=vals[k]. So i is a partner of i?
        # sum_partner_mask[i] would contain i if 2*vals[i] is in vals_set.
        # But S does not contain i yet. So we check S & mask.
        # If k is in S, and 2*vals[i]=vals[k], then we have a conflict.
        # Is k in sum_partner_mask[i]?
        # sum_partner_mask[i] contains a such that vals[a]+vals[i] in vals_set.
        # If a=k, vals[k]+vals[i] = 2*vals[i]+vals[i] = 3*vals[i]. Not necessarily in vals_set.
        # So `sum_partner_mask` defined as "a such that a+i is in vals_set" does NOT capture the double case directly if we just check `S & mask`.
        # Wait. If we add i, we need to check if there exists a in S such that a+i in S.
        # This is exactly: Is there a in S such that a is in sum_partner_mask[i]?
        # No. sum_partner_mask[i] is the set of a such that a+i is in vals_set.
        # If a is in S AND a is in sum_partner_mask[i], then a+i is in vals_set.
        # But we need a+i to be IN S.
        # So the condition is: (S & sum_partner_mask[i]) is NOT sufficient.
        # We need: For all a in S, (a+i) is NOT in S.
        # This is equivalent to: (S & (S + i)) is empty? No.
        # Let's stick to the O(n) check or precompute a different mask.
        
        # Actually, let's precompute `forbidden[i]`: bitmask of j such that vals[i] + vals[j] is in vals_set.
        # This is symmetric.
        # If we have set S, and we want to add i:
        # We need to ensure that for all j in S, (i+j) is NOT in S.
        # This is hard to check with a single bitmask of S.
        
        # Let's use the O(n) check in the inner loop for now, as n is small (<=240).
        # can_add(S_set, x):
        #   if 2*x in S_set: return False
        #   for a in S_set:
        #       if a+x in S_set: return False
        #   return True
        
        # To speed up, we can precompute `sums_in_vals[x]`? No, x is from vals.
        
        # Let's optimize can_add using a set of S.
        # S_set is a set of values.
        
        def can_add_fast(S_set, x):
            if 2 * x in S_set:
                return False
            # Optimization: if x is large, a+x might be out of range? No, range is fixed.
            for a in S_set:
                if a + x in S_set:
                    return False
            return True

        # Initialization: Greedy
        # Try multiple random orders for greedy init
        best_size = 0
        best_S = set()
        
        rng = random.Random(seed)
        
        # Pre-compute a list of all elements
        all_vals = list(vals)
        
        # Try 5 greedy initializations with different orderings
        for _ in range(5):
            S = set()
            # Random shuffle
            shuffled = all_vals[:]
            rng.shuffle(shuffled)
            for x in shuffled:
                if can_add_fast(S, x):
                    S.add(x)
            if len(S) > best_size:
                best_size = len(S)
                best_S = S
        
        # Local Search: Swap + Repair
        # Time budget management
        end_time = start_time + time_budget
        
        while time.time() < end_time:
            # Try to add any element greedily
            improved = True
            while improved:
                improved = False
                # Try to add a random element not in S
                # To be efficient, pick random candidates
                for _ in range(100):
                    x = rng.choice(all_vals)
                    if x not in best_S:
                        if can_add_fast(best_S, x):
                            best_S.add(x)
                            best_size += 1
                            improved = True
                            break # Restart the inner while to check others
                
            if time.time() > end_time:
                break
                
            # If we couldn't add anything, we are in a local optimum.
            # Perform a "kick": remove 1-3 random elements, then repair.
            # We want to try to escape the local optimum.
            
            # Choose kick strength
            kick_size = rng.randint(1, 3)
            
            # Create a copy of current set
            S_temp = set(best_S)
            
            # Remove kick_size random elements
            to_remove = []
            for _ in range(kick_size):
                if not S_temp:
                    break
                x = rng.choice(list(S_temp))
                to_remove.append(x)
                S_temp.remove(x)
                
            # Now try to repair S_temp by adding random elements
            # We want to see if we can get a larger set.
            # Run a greedy repair on S_temp
            repair_improved = True
            while repair_improved:
                repair_improved = False
                for _ in range(50):
                    x = rng.choice(all_vals)
                    if x not in S_temp:
                        if can_add_fast(S_temp, x):
                            S_temp.add(x)
                            repair_improved = True
                            break
            
            # Compare
            if len(S_temp) > best_size:
                best_S = S_temp
                best_size = len(S_temp)
            elif len(S_temp) == best_size:
                # Plateau: accept with some probability to explore
                # Or check if the structure is "better"? Hard to measure.
                # Just accept 10% of the time to randomize the landscape
                if rng.random() < 0.1:
                    best_S = S_temp
            else:
                # Reject
                pass
                
        # Convert back to list
        return list(best_S)

    def solve_one(vals, time_budget=35.0):
        n = len(vals)
        start = time.time()
        
        # Strategy 1: Odd numbers (always sum-free)
        odd = [x for x in vals if x % 2 != 0]
        best = odd
        best_size = len(odd)
        
        # Strategy 2: WalkSAT / Local Search
        remaining = time_budget - (time.time() - start)
        if remaining > 2.0:
            try:
                cand = solve_walksat(vals, time_budget=remaining - 1.0, seed=42)
                if len(cand) > best_size:
                    best = cand
                    best_size = len(cand)
            except Exception:
                pass

        # Final safety check
        if not is_sum_free(best):
            # Fallback to odd
            best = odd
        
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2 != 0])
            continue
        
        vals = instance(size, seed)
        
        # Allocate time
        if size <= 60:
            budget = 30.0
        elif size <= 120:
            budget = 45.0
        else:
            budget = 60.0
        
        remaining = DEADLINE - elapsed - 5
        budget = min(budget, remaining)
        
        result = solve_one(vals, time_budget=budget)
        results.append(result)

    return results
# EVOLVE-BLOCK-END
